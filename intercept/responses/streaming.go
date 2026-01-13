package responses

import (
	"context"
	"errors"
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
		shutdownCtx, shutdownCancel := context.WithTimeout(ctx, streamShutdownTimeout)
		defer shutdownCancel()
		_ = events.Shutdown(shutdownCtx)
	}()

	var respFwd responseForwarder

	srv := i.newResponsesService()
	opts := i.requestOptions(&respFwd)
	stream := srv.NewStreaming(ctx, i.req.ResponseNewParams, opts...)
	defer stream.Close()

	upstreamErr := stream.Err()
	if upstreamErr != nil {
		// events stream should never be initialized
		if events.IsStreaming() {
			i.logger.Warn(ctx, "event stream was initialized when no response was received from upstream")
			return upstreamErr
		}

		// no response received from upstream, return custom error
		if !respFwd.responseReceived.Load() {
			i.sendCustomErr(ctx, w, http.StatusInternalServerError, upstreamErr)
			return upstreamErr
		}

		// forward received response as-is
		err := respFwd.forwardResp(w)
		return errors.Join(upstreamErr, err)
	}

	for stream.Next() {
		if err := events.Send(ctx, respFwd.buff.readDelta()); err != nil {
			err = fmt.Errorf("failed to relay chunk: %w", err)
			return err
		}
	}

	b, err := respFwd.readAll()
	if err != nil {
		return errors.Join(upstreamErr, fmt.Errorf("failed to read response body: %w", upstreamErr))
	}

	err = events.Send(ctx, b)
	return errors.Join(err, stream.Err())
}
