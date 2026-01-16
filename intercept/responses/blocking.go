package responses

import (
	"errors"
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

func NewBlockingInterceptor(id uuid.UUID, req *ResponsesNewParamsWrapper, reqPayload []byte, baseURL string, key string, model string) *BlockingResponsesInterceptor {
	return &BlockingResponsesInterceptor{
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

	srv := i.newResponsesService()
	var respCopy responseCopier

	opts := i.requestOptions(&respCopy)
	response, upstreamErr := srv.New(ctx, i.req.ResponseNewParams, opts...)

	// response could be nil eg. fixtures/openai/responses/blocking/wrong_response_format.txtar
	if response != nil {
		i.recordUserPrompt(ctx, response.ID)
		i.recordToolUsage(ctx, response)
	}

	if upstreamErr != nil && !respCopy.responseReceived.Load() {
		// no response received from upstream, return custom error
		i.sendCustomErr(ctx, w, http.StatusInternalServerError, upstreamErr)
	}

	err := respCopy.forwardResp(w)

	return errors.Join(upstreamErr, err)
}
