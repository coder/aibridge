package responses

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"time"

	"cdr.dev/slog/v3"
	"github.com/coder/aibridge/config"
	"github.com/coder/aibridge/mcp"
	"github.com/coder/aibridge/recorder"
	"github.com/google/uuid"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/responses"
	"github.com/tidwall/sjson"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
)

type BlockingResponsesInterceptor struct {
	responsesInterceptionBase
}

func NewBlockingInterceptor(id uuid.UUID, req *ResponsesNewParamsWrapper, reqPayload []byte, cfg config.OpenAI, model string, tracer trace.Tracer) *BlockingResponsesInterceptor {
	return &BlockingResponsesInterceptor{
		responsesInterceptionBase: responsesInterceptionBase{
			id:         id,
			req:        req,
			reqPayload: reqPayload,
			cfg:        cfg,
			model:      model,
			tracer:     tracer,
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

	i.injectTools()
	i.disableParallelToolCalls()

	var (
		response    *responses.Response
		upstreamErr error
		respCopy    responseCopier
	)

	for {
		srv := i.newResponsesService()
		respCopy = responseCopier{}

		opts := i.requestOptions(&respCopy)
		opts = append(opts, option.WithRequestTimeout(time.Second*600))
		response, upstreamErr = srv.New(ctx, i.req.ResponseNewParams, opts...)

		if upstreamErr != nil {
			break
		}

		// response could be nil eg. fixtures/openai/responses/blocking/wrong_response_format.txtar
		if response == nil {
			break
		}

		// Record prompt usage on first successful response.
		i.recordUserPrompt(ctx, response.ID)
		i.recordTokenUsage(ctx, response)

		// Check if there any injected tools to invoke.
		pending := i.getPendingInjectedToolCalls(ctx, response)
		if len(pending) == 0 {
			// No injected tools, record non-injected tool usage.
			i.recordNonInjectedToolUsage(ctx, response)

			// No injected function calls need to be invoked, flow is complete.
			break
		}

		shouldLoop, err := i.handleInnerAgenticLoop(ctx, pending, response)
		if err != nil {
			i.sendCustomErr(ctx, w, http.StatusInternalServerError, err)
			shouldLoop = false
		}

		if !shouldLoop {
			break
		}
	}

	if upstreamErr != nil && !respCopy.responseReceived.Load() {
		// no response received from upstream, return custom error
		i.sendCustomErr(ctx, w, http.StatusInternalServerError, upstreamErr)
	}

	err := respCopy.forwardResp(w)

	return errors.Join(upstreamErr, err)
}

// handleInnerAgenticLoop orchestrates the inner agentic loop whereby injected tools
// are invoked and their results are sent back to the model.
// This is in contrast to regular tool calls which will be handled by the client
// in its own agentic loop.
func (i *BlockingResponsesInterceptor) handleInnerAgenticLoop(ctx context.Context, pending []responses.ResponseFunctionToolCall, response *responses.Response) (bool, error) {
	// Invoke any injected function calls.
	// The Responses API refers to what we call "tools" as "functions", so we keep the terminology
	// consistent in this package.
	// See https://platform.openai.com/docs/guides/function-calling
	results, err := i.handleInjectedToolCalls(ctx, pending, response)
	if err != nil {
		return false, fmt.Errorf("failed to handle injected tool calls: %w", err)
	}

	// No tool results means no tools were invocable, so the flow is complete.
	if len(results) == 0 {
		return false, nil
	}

	// We'll use the tool results to issue another request to provide the model with.
	i.prepareRequestForAgenticLoop(response)
	i.req.Input.OfInputItemList = append(i.req.Input.OfInputItemList, results...)

	// TODO: we should avoid re-marshaling Input, but since it changes from a string to
	// a list in this loop, we have to.
	// See responsesInterceptionBase.requestOptions for more details about marshaling issues.
	i.reqPayload, err = sjson.SetBytes(i.reqPayload, "input", i.req.Input)
	if err != nil {
		i.logger.Error(ctx, "failure to marshal new input in inner agentic loop", slog.Error(err))
		// TODO: what should be returned under this condition?
		return false, nil
	}

	return true, nil
}
