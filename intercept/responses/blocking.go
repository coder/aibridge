package responses

import (
	"errors"
	"fmt"
	"net/http"

	"cdr.dev/slog/v3"
	"github.com/coder/aibridge/mcp"
	"github.com/coder/aibridge/recorder"
	"github.com/google/uuid"
	"go.opentelemetry.io/otel/attribute"
)

type BlockingResponsesInterceptor struct {
	responsesInterceptionBase
}

func NewBlockingInterceptor(id uuid.UUID, req *ResponsesNewParamsWrapper, reqPayload []byte, baseURL, key string) *BlockingResponsesInterceptor {
	return &BlockingResponsesInterceptor{responsesInterceptionBase: responsesInterceptionBase{
		id:         id,
		req:        req,
		reqPayload: reqPayload,
		baseURL:    baseURL,
		key:        key,
	}}
}

func (i *BlockingResponsesInterceptor) Setup(logger slog.Logger, recorder recorder.Recorder, mcpProxy mcp.ServerProxier) {
	i.responsesInterceptionBase.Setup(logger.Named("blocking"), recorder, mcpProxy)
}

func (i *BlockingResponsesInterceptor) Streaming() bool {
	return false
}

func (i *BlockingResponsesInterceptor) TraceAttributes(r *http.Request) []attribute.KeyValue {
	return i.responsesInterceptionBase.baseTraceAttributes(r, false)
}

func (i *BlockingResponsesInterceptor) ProcessRequest(w http.ResponseWriter, r *http.Request) error {
	ctx := r.Context()
	if err := i.validateRequest(ctx, w); err != nil {
		return err
	}

	var respPayload deltaBuffer
	srv := NewResponsesService(i.baseURL, i.key)

	opts := i.requestOptions(&respPayload)
	_, upstreamErr := srv.New(ctx, i.req.ResponseNewParams, opts...)

	upstreamErr, earlyExit := i.handleUpstreamError(ctx, upstreamErr, w)
	if earlyExit {
		return upstreamErr
	}

	if err := respPayload.drain(); err != nil {
		i.logger.Warn(ctx, "failed to drain original response body", slog.Error(err))
	}
	w.Header().Set("Content-Type", "application/json")
	_, err := w.Write(respPayload.readDelta())
	if err != nil {
		i.logger.Warn(ctx, "failed to write response", slog.Error(err))
		return errors.Join(fmt.Errorf("failed to write response: %w", err), upstreamErr)
	}

	return upstreamErr
}
