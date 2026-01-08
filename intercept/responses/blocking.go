package responses

import (
	"errors"
	"fmt"
	"io"
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

	var respBody deltaBuffer
	srv := NewResponsesService(i.baseURL, i.key, i.logger)

	opts := i.requestOptions(&respBody)
	_, err := srv.New(ctx, i.req.ResponseNewParams, opts...)

	upstreamErr, earlyExit := i.handleUpstreamError(ctx, err, w)
	if earlyExit {
		return upstreamErr
	}

	w.Header().Set("Content-Type", "application/json")
	out, err := io.ReadAll(&respBody)
	if err != nil {
		i.logger.Warn(ctx, "failed to read upstream response", slog.Error(err))
		return errors.Join(fmt.Errorf("failed to read upstream response: %w", err), upstreamErr)
	}

	_, err = w.Write(out)
	if err != nil {
		i.logger.Warn(ctx, "failed to write response", slog.Error(err))
		return errors.Join(fmt.Errorf("failed to write response: %w", err), upstreamErr)
	}

	return upstreamErr
}
