package aibridge

import (
	"context"
	"encoding/json"
	"net/http"

	"github.com/coder/aibridge/mcp"
	"github.com/coder/aibridge/tracing"
	"github.com/google/uuid"
	"github.com/openai/openai-go/v2"
	"github.com/openai/openai-go/v2/option"
	"github.com/openai/openai-go/v2/shared"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"

	"cdr.dev/slog"
)

type OpenAIChatInterceptionBase struct {
	id      uuid.UUID
	req     *ChatCompletionNewParamsWrapper
	baseURL string
	key     string

	tracer trace.Tracer
	logger slog.Logger

	recorder Recorder
	mcpProxy mcp.ServerProxier
}

func (i *OpenAIChatInterceptionBase) newCompletionsService(baseURL, key string) openai.ChatCompletionService {
	opts := []option.RequestOption{option.WithAPIKey(key), option.WithBaseURL(baseURL)}

	return openai.NewChatCompletionService(opts...)
}

func (i *OpenAIChatInterceptionBase) ID() uuid.UUID {
	return i.id
}

func (i *OpenAIChatInterceptionBase) Setup(logger slog.Logger, recorder Recorder, mcpProxy mcp.ServerProxier) {
	i.logger = logger
	i.recorder = recorder
	i.mcpProxy = mcpProxy
}

func (s *OpenAIChatInterceptionBase) baseTraceAttributes(r *http.Request, streaming bool) []attribute.KeyValue {
	return []attribute.KeyValue{
		attribute.String(tracing.RequestPath, r.URL.Path),
		attribute.String(tracing.InterceptionID, s.id.String()),
		attribute.String(tracing.InitiatorID, actorFromContext(r.Context()).id),
		attribute.String(tracing.Provider, ProviderOpenAI),
		attribute.String(tracing.Model, s.Model()),
		attribute.Bool(tracing.Streaming, streaming),
	}
}

func (i *OpenAIChatInterceptionBase) Model() string {
	if i.req == nil {
		return "coder-aibridge-unknown"
	}

	return string(i.req.Model)
}

func (i *OpenAIChatInterceptionBase) newErrorResponse(err error) map[string]any {
	return map[string]any{
		"error":   true,
		"message": err.Error(),
	}
}

func (i *OpenAIChatInterceptionBase) injectTools() {
	if i.req == nil || i.mcpProxy == nil {
		return
	}

	// Inject tools.
	for _, tool := range i.mcpProxy.ListTools() {
		fn := openai.ChatCompletionToolUnionParam{
			OfFunction: &openai.ChatCompletionFunctionToolParam{
				Function: shared.FunctionDefinitionParam{
					Name:        tool.ID,
					Strict:      openai.Bool(false), // TODO: configurable.
					Description: openai.String(tool.Description),
					Parameters: openai.FunctionParameters{
						"type":       "object",
						"properties": tool.Params,
						// "additionalProperties": false, // Only relevant when strict=true.
					},
				},
			},
		}

		// Otherwise the request fails with "None is not of type 'array'" if a nil slice is given.
		if len(tool.Required) > 0 {
			// Must list ALL properties when strict=true.
			fn.OfFunction.Function.Parameters["required"] = tool.Required
		}

		i.req.Tools = append(i.req.Tools, fn)
	}
}

// writeUpstreamError marshals and writes a given error.
func (i *OpenAIChatInterceptionBase) writeUpstreamError(w http.ResponseWriter, oaiErr *OpenAIErrorResponse) {
	if oaiErr == nil {
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(oaiErr.StatusCode)

	out, err := json.Marshal(oaiErr)
	if err != nil {
		i.logger.Warn(context.Background(), "failed to marshal upstream error", slog.Error(err), slog.F("error_payload", slog.F("%+v", oaiErr)))
		// Response has to match expected format.
		_, _ = w.Write([]byte(`{
	"error": {
		"type": "error",
		"message":"error marshaling upstream error",
		"code": "server_error"
	},
}`))
	} else {
		_, _ = w.Write(out)
	}
}
