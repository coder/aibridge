package aibridge

import (
	"context"
	"fmt"
	"sync"
	"time"

	"cdr.dev/slog"
)

var _ Recorder = &RecorderWrapper{}

// RecorderWrapper is a convenience struct which implements RecorderClient and resolves a client before calling each method.
// It also sets the start/creation time of each record.
type RecorderWrapper struct {
	logger   slog.Logger
	clientFn func() (Recorder, error)
}

func (r *RecorderWrapper) RecordInterception(ctx context.Context, req *InterceptionRecord) error {
	client, err := r.clientFn()
	if err != nil {
		return fmt.Errorf("acquire client: %w", err)
	}

	req.StartedAt = time.Now()
	if err = client.RecordInterception(ctx, req); err == nil {
		return nil
	}

	r.logger.Warn(ctx, "failed to record interception", slog.Error(err), slog.F("interception_id", req.ID))
	return err
}

func (r *RecorderWrapper) RecordPromptUsage(ctx context.Context, req *PromptUsageRecord) error {
	client, err := r.clientFn()
	if err != nil {
		return fmt.Errorf("acquire client: %w", err)
	}

	req.CreatedAt = time.Now()
	if err = client.RecordPromptUsage(ctx, req); err == nil {
		return nil
	}

	r.logger.Warn(ctx, "failed to record prompt usage", slog.Error(err), slog.F("interception_id", req.InterceptionID))
	return err
}

func (r *RecorderWrapper) RecordTokenUsage(ctx context.Context, req *TokenUsageRecord) error {
	client, err := r.clientFn()
	if err != nil {
		return fmt.Errorf("acquire client: %w", err)
	}

	req.CreatedAt = time.Now()
	if err = client.RecordTokenUsage(ctx, req); err == nil {
		return nil
	}

	r.logger.Warn(ctx, "failed to record token usage", slog.Error(err), slog.F("interception_id", req.InterceptionID))
	return err
}

func (r *RecorderWrapper) RecordToolUsage(ctx context.Context, req *ToolUsageRecord) error {
	client, err := r.clientFn()
	if err != nil {
		return fmt.Errorf("acquire client: %w", err)
	}

	req.CreatedAt = time.Now()
	if err = client.RecordToolUsage(ctx, req); err == nil {
		return nil
	}

	r.logger.Warn(ctx, "failed to record tool usage", slog.Error(err), slog.F("interception_id", req.InterceptionID))
	return err
}

func NewRecorder(logger slog.Logger, clientFn func() (Recorder, error)) *RecorderWrapper {
	return &RecorderWrapper{logger: logger, clientFn: clientFn}
}

var _ Recorder = &AsyncRecorder{}

// AsyncRecorder calls [Recorder] methods asynchronously and logs any errors which may occur.
type AsyncRecorder struct {
	logger  slog.Logger
	wrapped Recorder
	timeout time.Duration

	wg sync.WaitGroup
}

func NewAsyncRecorder(logger slog.Logger, wrapped Recorder, timeout time.Duration) *AsyncRecorder {
	return &AsyncRecorder{logger: logger, wrapped: wrapped, timeout: timeout}
}

// RecordInterception must NOT be called asynchronously.
// If an interception cannot be recorded, the whole request should fail.
func (a *AsyncRecorder) RecordInterception(ctx context.Context, req *InterceptionRecord) error {
	panic("RecordInterception must not be called asynchronously")
}

func (a *AsyncRecorder) RecordPromptUsage(_ context.Context, req *PromptUsageRecord) error {
	a.wg.Add(1)
	go func() {
		defer a.wg.Done()
		timedCtx, cancel := context.WithTimeout(context.Background(), a.timeout)
		defer cancel()

		err := a.wrapped.RecordPromptUsage(timedCtx, req)
		if err != nil {
			a.logger.Warn(timedCtx, "failed to record usage", slog.F("type", "prompt"), slog.Error(err), slog.F("payload", req))
		}
	}()

	return nil // Caller is not interested in error.
}

func (a *AsyncRecorder) RecordTokenUsage(_ context.Context, req *TokenUsageRecord) error {
	a.wg.Add(1)
	go func() {
		defer a.wg.Done()
		timedCtx, cancel := context.WithTimeout(context.Background(), a.timeout)
		defer cancel()

		err := a.wrapped.RecordTokenUsage(timedCtx, req)
		if err != nil {
			a.logger.Warn(timedCtx, "failed to record usage", slog.F("type", "token"), slog.Error(err), slog.F("payload", req))
		}
	}()

	return nil // Caller is not interested in error.
}

func (a *AsyncRecorder) RecordToolUsage(_ context.Context, req *ToolUsageRecord) error {
	a.wg.Add(1)
	go func() {
		defer a.wg.Done()
		timedCtx, cancel := context.WithTimeout(context.Background(), a.timeout)
		defer cancel()

		err := a.wrapped.RecordToolUsage(timedCtx, req)
		if err != nil {
			a.logger.Warn(timedCtx, "failed to record usage", slog.F("type", "tool"), slog.Error(err), slog.F("payload", req))
		}
	}()

	return nil // Caller is not interested in error.
}

func (a *AsyncRecorder) Wait() {
	a.wg.Wait()
}
