package aibridge

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"strings"
	"sync"
	"syscall"
	"time"

	"cdr.dev/slog"
)

var ErrEventStreamClosed = errors.New("event stream closed")

const pingInterval = time.Second * 10

type event []byte

type eventStream struct {
	ctx    context.Context
	logger slog.Logger

	pingPayload []byte

	closeOnce    sync.Once
	shutdownOnce sync.Once
	doneOnce     sync.Once
	eventsCh     chan event

	// startedMu protects the started flag.
	startedMu sync.Mutex
	started   bool
	// doneCh is closed when the start loop exits.
	doneCh chan struct{}
}

// newEventStream creates a new SSE stream, with an optional payload which is used to send pings every [pingInterval].
func newEventStream(ctx context.Context, logger slog.Logger, pingPayload []byte) *eventStream {
	return &eventStream{
		ctx:    ctx,
		logger: logger,

		pingPayload: pingPayload,

		eventsCh: make(chan event, 128), // Small buffer to unblock senders; once full, senders will block.
		doneCh:   make(chan struct{}),
	}
}

// start handles sending Server-Sent Event to the client.
func (s *eventStream) start(w http.ResponseWriter, r *http.Request) {
	// Atomically signal that streaming has started
	s.startedMu.Lock()
	if s.started {
		// Another goroutine is already running.
		s.startedMu.Unlock()
		return
	}
	s.started = true
	s.startedMu.Unlock()

	defer func() {
		// Signal completion on exit so senders don't block indefinitely after closure.
		s.doneOnce.Do(func() {
			close(s.doneCh)
		})
	}()

	ctx := r.Context()

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")

	// Send initial flush to ensure connection is established.
	if err := flush(w); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// Send periodic pings to keep connections alive.
	// The upstream provider may also send their own pings, but we can't rely on this.
	tick := time.NewTicker(pingInterval)
	defer tick.Stop()

	for {
		var (
			ev   event
			open bool
		)

		select {
		case <-s.ctx.Done():
			return
		case <-ctx.Done():
			s.logger.Debug(ctx, "request context canceled", slog.Error(ctx.Err()))
			return
		case ev, open = <-s.eventsCh:
			if !open {
				return
			}
		case <-tick.C:
			ev = s.pingPayload
			if ev == nil {
				continue
			}
		}

		_, err := w.Write(ev)
		if err != nil {
			if isConnError(err) {
				s.logger.Debug(ctx, "client disconnected during SSE write", slog.Error(err))
			} else {
				s.logger.Warn(ctx, "failed to write SSE event", slog.Error(err))
			}
			return
		}
		if err := flush(w); err != nil {
			s.logger.Warn(ctx, "failed to flush", slog.Error(err))
			return
		}

		// Reset the timer once we've flushed some data to the stream, since it's already fresh.
		// No need to ping in that case.
		tick.Reset(pingInterval)
	}
}

// Send enqueues an event in a non-blocking fashion, but if the channel is full
// then it will block.
func (s *eventStream) Send(ctx context.Context, payload []byte) error {
	// Save an unnecessary marshaling if possible.
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-s.ctx.Done():
		return s.ctx.Err()
	case <-s.doneCh:
		return ErrEventStreamClosed
	default:
	}

	return s.sendRaw(ctx, payload)
}

func (s *eventStream) sendRaw(ctx context.Context, payload []byte) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-s.ctx.Done():
		return s.ctx.Err()
	case <-s.doneCh:
		return ErrEventStreamClosed
	case s.eventsCh <- payload:
		return nil
	}
}

// Shutdown gracefully shuts down the stream, sending any supplementary events downstream if required.
// ONLY call this once all events have been submitted.
func (s *eventStream) Shutdown(shutdownCtx context.Context) error {
	var shutdownErr error

	s.shutdownOnce.Do(func() {
		s.logger.Debug(shutdownCtx, "shutdown initiated", slog.F("outstanding_events", len(s.eventsCh)))

		// Now it is safe to close the events channel; the start() loop will exit
		// after draining remaining events and receivers will stop ranging.
		close(s.eventsCh)
	})

	// Atomically check if start() was called and close doneCh if it wasn't.
	s.startedMu.Lock()
	if !s.started {
		// start() was never called, close doneCh ourselves so we don't block forever.
		s.doneOnce.Do(func() {
			close(s.doneCh)
		})
		s.startedMu.Unlock()
		return nil
	}
	// start() was called (or is about to be called), it will close doneCh when it finishes.
	s.startedMu.Unlock()

	// Wait for start() to complete. We must ALWAYS wait for doneCh to prevent
	// races with http.ResponseWriter cleanup, even if contexts are cancelled.
	select {
	case <-shutdownCtx.Done():
		shutdownErr = fmt.Errorf("shutdown timeout with %d outstanding events: %w", len(s.eventsCh), shutdownCtx.Err())
	case <-s.ctx.Done():
		shutdownErr = fmt.Errorf("shutdown cancelled with %d outstanding events: %w", len(s.eventsCh), s.ctx.Err())
	case <-s.doneCh:
		// Goroutine has finished.
		return shutdownErr
	}

	// If we got here due to context cancellation/timeout, we MUST still wait for doneCh
	// to ensure the goroutine has stopped using the ResponseWriter before the HTTP handler returns.
	<-s.doneCh
	return shutdownErr
}

func (s *eventStream) isRunning() bool {
	s.startedMu.Lock()
	defer s.startedMu.Unlock()
	return s.started
}

// isConnError checks if an error is related to client disconnection or context cancellation.
func isConnError(err error) bool {
	if err == nil {
		return false
	}

	if errors.Is(err, io.EOF) {
		return true
	}

	if errors.Is(err, syscall.ECONNRESET) || errors.Is(err, syscall.EPIPE) || errors.Is(err, net.ErrClosed) {
		return true
	}

	errStr := err.Error()
	return strings.Contains(errStr, "broken pipe") ||
		strings.Contains(errStr, "connection reset by peer")
}

func isUnrecoverableError(err error) bool {
	if errors.Is(err, context.Canceled) {
		return true
	}

	return isConnError(err)
}

func flush(w http.ResponseWriter) (err error) {
	flusher, ok := w.(http.Flusher)
	if !ok || flusher == nil {
		return errors.New("SSE not supported")
	}

	defer func() {
		if r := recover(); r != nil {
			// Likely a broken connection, don't spam the logs.
		}
	}()

	flusher.Flush()
	return nil
}
