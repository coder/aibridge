package responses

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"time"

	"cdr.dev/slog/v3"
	"github.com/coder/aibridge/config"
	aibcontext "github.com/coder/aibridge/context"
	"github.com/coder/aibridge/intercept"
	"github.com/coder/aibridge/mcp"
	"github.com/coder/aibridge/recorder"
	"github.com/coder/aibridge/tracing"
	"github.com/google/uuid"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/responses"
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

func (i *BlockingResponsesInterceptor) ProcessRequest(w http.ResponseWriter, r *http.Request) (outErr error) {
	ctx, span := i.tracer.Start(r.Context(), "Intercept.ProcessRequest", trace.WithAttributes(tracing.InterceptionAttributesFromContext(r.Context())...))
	defer tracing.EndSpanErr(span, &outErr)

	if err := i.validateRequest(ctx, w); err != nil {
		return err
	}

	i.injectTools()
	i.disableParallelToolCalls()

	var (
		response        *responses.Response
		upstreamErr     error
		respCopy        responseCopier
		firstResponseID string
	)

	prompt, promptFound, err := i.lastUserPrompt(ctx)
	if err != nil {
		i.logger.Warn(ctx, "failed to get user prompt", slog.Error(err))
	}
	shouldLoop := true

	for shouldLoop {
		srv := i.newResponsesService()
		respCopy = responseCopier{}

		opts := i.requestOptions(&respCopy)
		opts = append(opts, option.WithRequestTimeout(time.Second*600))
		if actor := aibcontext.ActorFromContext(r.Context()); actor != nil && i.cfg.SendActorHeaders {
			opts = append(opts, intercept.ActorHeadersAsOpenAIOpts(actor)...)
		}

		response, upstreamErr = i.newResponse(ctx, srv, opts)

		if upstreamErr != nil || response == nil {
			break
		}

		if firstResponseID == "" {
			firstResponseID = response.ID
		}

		i.recordTokenUsage(ctx, response)

		// Check if there any injected tools to invoke.
		pending := i.getPendingInjectedToolCalls(response)
		shouldLoop, err = i.handleInnerAgenticLoop(ctx, pending, response)
		if err != nil {
			i.sendCustomErr(ctx, w, http.StatusInternalServerError, err)
			shouldLoop = false
		}
	}

	if promptFound {
		i.recordUserPrompt(ctx, firstResponseID, prompt)
	}
	i.recordNonInjectedToolUsage(ctx, response)

	if upstreamErr != nil && !respCopy.responseReceived.Load() {
		// no response received from upstream, return custom error
		i.sendCustomErr(ctx, w, http.StatusInternalServerError, upstreamErr)
		return fmt.Errorf("failed to connect to upstream: %w", upstreamErr)
	}

	err = respCopy.forwardResp(w)
	return errors.Join(upstreamErr, err)
}

func (i *BlockingResponsesInterceptor) newResponse(ctx context.Context, srv responses.ResponseService, opts []option.RequestOption) (_ *responses.Response, outErr error) {
	ctx, span := i.tracer.Start(ctx, "Intercept.ProcessRequest.Upstream", trace.WithAttributes(tracing.InterceptionAttributesFromContext(ctx)...))
	defer tracing.EndSpanErr(span, &outErr)

	return srv.New(ctx, i.req.ResponseNewParams, opts...)
}
