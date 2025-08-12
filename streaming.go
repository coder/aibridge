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
	eventsCh     chan event

	// doneCh is closed when the run loop exits.
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

// run handles sending Server-Sent Event to the client.
func (s *eventStream) run(w http.ResponseWriter, r *http.Request) {
	// Signal completion on exit so senders don't block indefinitely after closure.
	defer close(s.doneCh)

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
	s.shutdownOnce.Do(func() {
		s.logger.Debug(shutdownCtx, "shutdown initiated", slog.F("outstanding_events", len(s.eventsCh)))

		// Now it is safe to close the events channel; the run loop will exit
		// after draining remaining events and receivers will stop ranging.
		close(s.eventsCh)
	})

	select {
	case <-shutdownCtx.Done():
		// If shutdownCtx completes, shutdown likely exceeded its timeout.
		return fmt.Errorf("shutdown ended prematurely with %d outstanding events: %w", len(s.eventsCh), shutdownCtx.Err())
	case <-s.ctx.Done():
		return fmt.Errorf("shutdown ended prematurely with %d outstanding events: %w", len(s.eventsCh), s.ctx.Err())
	case <-s.doneCh:
		return nil
	}
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
