package responses

import (
	"context"
	"fmt"
	"net/http"
	"time"

	"cdr.dev/slog/v3"
	"github.com/coder/aibridge/intercept/eventstream"
	"github.com/coder/aibridge/mcp"
	"github.com/coder/aibridge/recorder"
	"github.com/google/uuid"
	"go.opentelemetry.io/otel/attribute"
)

const (
	streamShutdownTimeout = time.Second * 30 // TODO: configurable
)

type StreamingResponsesInterceptor struct {
	responsesInterceptionBase
}

func NewStreamingInterceptor(id uuid.UUID, req *ResponsesNewParamsWrapper, reqPayload []byte, baseURL string, key string, model string) *StreamingResponsesInterceptor {
	return &StreamingResponsesInterceptor{
		responsesInterceptionBase: responsesInterceptionBase{
			id:         id,
			req:        req,
			reqPayload: reqPayload,
			baseURL:    baseURL,
			apiKey:     key,
			model:      model,
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

	events := eventstream.NewEventStream(ctx, i.logger.Named("sse-sender"), nil)
	go events.Start(w, r)
	defer func() {
		// Catch-all in case it doesn't get shutdown after stream completes.
		shutdownCtx, shutdownCancel := context.WithTimeout(ctx, streamShutdownTimeout)
		defer shutdownCancel()
		_ = events.Shutdown(shutdownCtx)
	}()

	var respPayload deltaBuffer

	srv := i.newResponsesService()
	opts := i.requestOptions(&respPayload, w)
	stream := srv.NewStreaming(ctx, i.req.ResponseNewParams, opts...)
	defer stream.Close()

	for stream.Next() {
		if err := events.Send(ctx, respPayload.readDelta()); err != nil {
			i.logger.Warn(ctx, "failed to relay chunk", slog.Error(err))
			err = fmt.Errorf("relay chunk: %w", err)
			stream.Close()
			break
		}
	}

	upstreamErr, failFast := i.handleUpstreamError(ctx, stream.Err(), w)
	if failFast {
		return upstreamErr
	}

	if err := respPayload.drainCloser(); err != nil {
		i.logger.Warn(ctx, "failed to drain original response body", slog.Error(err))
	}
	events.Send(ctx, respPayload.readDelta())

	return stream.Err()
}
