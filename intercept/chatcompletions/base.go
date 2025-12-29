package chatcompletions

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"strings"

	"github.com/coder/aibridge/config"
	aibcontext "github.com/coder/aibridge/context"
	"github.com/coder/aibridge/mcp"
	"github.com/coder/aibridge/recorder"
	"github.com/coder/aibridge/tracing"
	"github.com/google/uuid"
	"github.com/openai/openai-go/v2"
	"github.com/openai/openai-go/v2/option"
	"github.com/openai/openai-go/v2/shared"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"

	"cdr.dev/slog"
)

type interceptionBase struct {
	id      uuid.UUID
	req     *ChatCompletionNewParamsWrapper
	baseURL string
	key     string

	logger slog.Logger
	tracer trace.Tracer

	recorder recorder.Recorder
	mcpProxy mcp.ServerProxier
}

func (i *interceptionBase) newCompletionsService(baseURL string, key string) openai.ChatCompletionService {
	opts := []option.RequestOption{option.WithAPIKey(key), option.WithBaseURL(baseURL)}

	return openai.NewChatCompletionService(opts...)
}

func (i *interceptionBase) ID() uuid.UUID {
	return i.id
}

func (i *interceptionBase) Setup(logger slog.Logger, recorder recorder.Recorder, mcpProxy mcp.ServerProxier) {
	i.logger = logger
	i.recorder = recorder
	i.mcpProxy = mcpProxy
}

func (s *interceptionBase) baseTraceAttributes(r *http.Request, streaming bool) []attribute.KeyValue {
	return []attribute.KeyValue{
		attribute.String(tracing.RequestPath, r.URL.Path),
		attribute.String(tracing.InterceptionID, s.id.String()),
		attribute.String(tracing.InitiatorID, aibcontext.ActorFromContext(r.Context()).ID),
		attribute.String(tracing.Provider, config.ProviderOpenAI),
		attribute.String(tracing.Model, s.Model()),
		attribute.Bool(tracing.Streaming, streaming),
	}
}

func (i *interceptionBase) Model() string {
	if i.req == nil {
		return "coder-aibridge-unknown"
	}

	return string(i.req.Model)
}

func (i *interceptionBase) newErrorResponse(err error) map[string]any {
	return map[string]any{
		"error":   true,
		"message": err.Error(),
	}
}

func (i *interceptionBase) injectTools() {
	if i.req == nil || i.mcpProxy == nil {
		return
	}

	// Inject tools.
	for _, tool := range i.mcpProxy.ListTools() {
		fn := openai.ChatCompletionToolUnionParam{
			OfFunction: &openai.ChatCompletionFunctionToolParam{
				Function: openai.FunctionDefinitionParam{
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

func (i *interceptionBase) unmarshalArgs(in string) (args recorder.ToolArgs) {
	if len(strings.TrimSpace(in)) == 0 {
		return args // An empty string will fail JSON unmarshaling.
	}

	if err := json.Unmarshal([]byte(in), &args); err != nil {
		i.logger.Warn(context.Background(), "failed to unmarshal tool args", slog.Error(err))
	}

	return args
}

// writeUpstreamError marshals and writes a given error.
func (i *interceptionBase) writeUpstreamError(w http.ResponseWriter, oaiErr *errorResponse) {
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

func sumUsage(ref, in openai.CompletionUsage) openai.CompletionUsage {
	return openai.CompletionUsage{
		CompletionTokens: ref.CompletionTokens + in.CompletionTokens,
		PromptTokens:     ref.PromptTokens + in.PromptTokens,
		TotalTokens:      ref.TotalTokens + in.TotalTokens,
		CompletionTokensDetails: openai.CompletionUsageCompletionTokensDetails{
			AcceptedPredictionTokens: ref.CompletionTokensDetails.AcceptedPredictionTokens + in.CompletionTokensDetails.AcceptedPredictionTokens,
			AudioTokens:              ref.CompletionTokensDetails.AudioTokens + in.CompletionTokensDetails.AudioTokens,
			ReasoningTokens:          ref.CompletionTokensDetails.ReasoningTokens + in.CompletionTokensDetails.ReasoningTokens,
			RejectedPredictionTokens: ref.CompletionTokensDetails.RejectedPredictionTokens + in.CompletionTokensDetails.RejectedPredictionTokens,
		},
		PromptTokensDetails: openai.CompletionUsagePromptTokensDetails{
			AudioTokens:  ref.PromptTokensDetails.AudioTokens + in.PromptTokensDetails.AudioTokens,
			CachedTokens: ref.PromptTokensDetails.CachedTokens + in.PromptTokensDetails.CachedTokens,
		},
	}
}

// calculateActualInputTokenUsage accounts for cached tokens which are included in [openai.CompletionUsage].PromptTokens.
func calculateActualInputTokenUsage(in openai.CompletionUsage) int64 {
	// Input *includes* the cached tokens, so we subtract them here to reflect actual input token usage.
	// The original value can be reconstructed by referencing the "prompt_cached" field in metadata.
	// See https://platform.openai.com/docs/api-reference/usage/completions_object#usage/completions_object-input_tokens.
	return in.PromptTokens /* The aggregated number of text input tokens used, including cached tokens. */ -
		in.PromptTokensDetails.CachedTokens /* The aggregated number of text input tokens that has been cached from previous requests. */
}

func getErrorResponse(err error) *errorResponse {
	var apiErr *openai.Error
	if !errors.As(err, &apiErr) {
		return nil
	}

	return &errorResponse{
		ErrorObject: &shared.ErrorObject{
			Code:    apiErr.Code,
			Message: apiErr.Message,
			Type:    apiErr.Type,
		},
		StatusCode: apiErr.StatusCode,
	}
}

var _ error = &errorResponse{}

type errorResponse struct {
	ErrorObject *shared.ErrorObject `json:"error"`
	StatusCode  int                 `json:"-"`
}

func newErrorResponse(msg error) *errorResponse {
	return &errorResponse{
		ErrorObject: &shared.ErrorObject{
			Code:    "error",
			Message: msg.Error(),
			Type:    "error",
		},
	}
}

func (a *errorResponse) Error() string {
	if a.ErrorObject == nil {
		return ""
	}
	return a.ErrorObject.Message
}
