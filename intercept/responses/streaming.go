package responses

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"time"

	"cdr.dev/slog/v3"
	"github.com/coder/aibridge/intercept/eventstream"
	"github.com/coder/aibridge/mcp"
	"github.com/coder/aibridge/recorder"
	"github.com/google/uuid"
	"go.opentelemetry.io/otel/attribute"
)

type StreamingResponsesInterceptor struct {
	responsesInterceptionBase
}

func NewStreamingInterceptor(id uuid.UUID, req *ResponsesNewParamsWrapper, reqPayload []byte, baseURL, key string) *StreamingResponsesInterceptor {
	return &StreamingResponsesInterceptor{
		responsesInterceptionBase: responsesInterceptionBase{
			id:         id,
			req:        req,
			reqPayload: reqPayload,
			baseURL:    baseURL,
			key:        key,
		},
	}
}

func (i *StreamingResponsesInterceptor) Setup(logger slog.Logger, recorder recorder.Recorder, mcpProxy mcp.ServerProxier) {
	i.responsesInterceptionBase.Setup(logger.Named("streaming"), recorder, mcpProxy)
}

func (i *StreamingResponsesInterceptor) Streaming() bool {
	return true
}

func (i *StreamingResponsesInterceptor) TraceAttributes(r *http.Request) []attribute.KeyValue {
	return i.responsesInterceptionBase.baseTraceAttributes(r, true)
}

func (i *StreamingResponsesInterceptor) ProcessRequest(w http.ResponseWriter, r *http.Request) error {
	ctx, cancel := context.WithCancel(r.Context())
	defer cancel()
	r = r.WithContext(ctx) // Rewire context for SSE cancellation.

	if err := i.validateRequest(ctx, w); err != nil {
		return err
	}

	streamCtx, streamCancel := context.WithCancelCause(ctx)
	defer streamCancel(errors.New("deferred"))

	// events will either terminate when shutdown after interaction with upstream completes, or when streamCtx is done.
	events := eventstream.NewEventStream(streamCtx, i.logger.Named("sse-sender"), nil)
	go events.Start(w, r)
	defer func() {
		_ = events.Shutdown(streamCtx) // Catch-all in case it doesn't get shutdown after stream completes.
	}()

	var respBody deltaBuffer

	srv := NewResponsesService(i.baseURL, i.key, i.logger)
	opts := i.requestOptions(&respBody)
	stream := srv.NewStreaming(ctx, i.req.ResponseNewParams, opts...)
	defer stream.Close()

	for stream.Next() {
		if err := events.Send(ctx, respBody.readDelta()); err != nil {
			i.logger.Warn(ctx, "failed to relay chunk", slog.Error(err))
			err = fmt.Errorf("relay chunk: %w", err)
			stream.Close()
			break
		}
	}

	upstreamErr, earlyExit := i.handleUpstreamError(ctx, stream.Err(), w)
	if earlyExit {
		return upstreamErr
	}

	// Sometimes stream.Next() returns before respBody buffer is filled
	lastRead, err := io.ReadAll(&respBody)
	if err != nil {
		i.logger.Warn(ctx, "failed to read upstream response", slog.Error(err))
		return fmt.Errorf("failed to read upstream response: %w", err)
	}
	events.Send(ctx, lastRead)

	// Give the events stream 30 seconds (TODO: configurable) to gracefully shutdown.
	shutdownCtx, shutdownCancel := context.WithTimeout(ctx, time.Second*30)
	defer shutdownCancel()
	if err := events.Shutdown(shutdownCtx); err != nil {
		i.logger.Warn(ctx, "event stream shutdown", slog.Error(err))
	}

	return stream.Err()
}
