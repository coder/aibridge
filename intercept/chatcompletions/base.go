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

type InterceptionBase struct {
	id      uuid.UUID
	req     *ChatCompletionNewParamsWrapper
	baseURL string
	key     string

	logger slog.Logger
	tracer trace.Tracer

	recorder recorder.Recorder
	mcpProxy mcp.ServerProxier
}

func (i *InterceptionBase) newCompletionsService(baseURL string, key string) openai.ChatCompletionService {
	opts := []option.RequestOption{option.WithAPIKey(key), option.WithBaseURL(baseURL)}

	return openai.NewChatCompletionService(opts...)
}

func (i *InterceptionBase) ID() uuid.UUID {
	return i.id
}

func (i *InterceptionBase) Setup(logger slog.Logger, recorder recorder.Recorder, mcpProxy mcp.ServerProxier) {
	i.logger = logger
	i.recorder = recorder
	i.mcpProxy = mcpProxy
}

func (s *InterceptionBase) BaseTraceAttributes(r *http.Request, streaming bool) []attribute.KeyValue {
	return []attribute.KeyValue{
		attribute.String(tracing.RequestPath, r.URL.Path),
		attribute.String(tracing.InterceptionID, s.id.String()),
		attribute.String(tracing.InitiatorID, aibcontext.ActorFromContext(r.Context()).ID),
		attribute.String(tracing.Provider, config.ProviderOpenAI),
		attribute.String(tracing.Model, s.Model()),
		attribute.Bool(tracing.Streaming, streaming),
	}
}

func (i *InterceptionBase) Model() string {
	if i.req == nil {
		return "coder-aibridge-unknown"
	}

	return string(i.req.Model)
}

func (i *InterceptionBase) NewErrorResponse(err error) map[string]any {
	return map[string]any{
		"error":   true,
		"message": err.Error(),
	}
}

func (i *InterceptionBase) InjectTools() {
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

func (i *InterceptionBase) UnmarshalArgs(in string) (args recorder.ToolArgs) {
	if len(strings.TrimSpace(in)) == 0 {
		return args // An empty string will fail JSON unmarshaling.
	}

	if err := json.Unmarshal([]byte(in), &args); err != nil {
		i.logger.Warn(context.Background(), "failed to unmarshal tool args", slog.Error(err))
	}

	return args
}

// WriteUpstreamError marshals and writes a given error.
func (i *InterceptionBase) WriteUpstreamError(w http.ResponseWriter, oaiErr *ErrorResponse) {
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

// GetLogger returns the logger for use in subpackage types.
func (i *InterceptionBase) GetLogger() slog.Logger {
	return i.logger
}

// GetRecorder returns the recorder for use in subpackage types.
func (i *InterceptionBase) GetRecorder() recorder.Recorder {
	return i.recorder
}

// GetMCPProxy returns the MCP proxy for use in subpackage types.
func (i *InterceptionBase) GetMCPProxy() mcp.ServerProxier {
	return i.mcpProxy
}

// GetRequest returns the request wrapper for use in subpackage types.
func (i *InterceptionBase) GetRequest() *ChatCompletionNewParamsWrapper {
	return i.req
}

// GetTracer returns the tracer for use in subpackage types.
func (i *InterceptionBase) GetTracer() trace.Tracer {
	return i.tracer
}

// GetBaseURL returns the base URL for use in subpackage types.
func (i *InterceptionBase) GetBaseURL() string {
	return i.baseURL
}

// GetKey returns the API key for use in subpackage types.
func (i *InterceptionBase) GetKey() string {
	return i.key
}

func SumUsage(ref, in openai.CompletionUsage) openai.CompletionUsage {
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

// CalculateActualInputTokenUsage accounts for cached tokens which are included in [openai.CompletionUsage].PromptTokens.
func CalculateActualInputTokenUsage(in openai.CompletionUsage) int64 {
	// Input *includes* the cached tokens, so we subtract them here to reflect actual input token usage.
	// The original value can be reconstructed by referencing the "prompt_cached" field in metadata.
	// See https://platform.openai.com/docs/api-reference/usage/completions_object#usage/completions_object-input_tokens.
	return in.PromptTokens /* The aggregated number of text input tokens used, including cached tokens. */ -
		in.PromptTokensDetails.CachedTokens /* The aggregated number of text input tokens that has been cached from previous requests. */
}

func GetErrorResponse(err error) *ErrorResponse {
	var apiErr *openai.Error
	if !errors.As(err, &apiErr) {
		return nil
	}

	return &ErrorResponse{
		ErrorObject: &shared.ErrorObject{
			Code:    apiErr.Code,
			Message: apiErr.Message,
			Type:    apiErr.Type,
		},
		StatusCode: apiErr.StatusCode,
	}
}

var _ error = &ErrorResponse{}

type ErrorResponse struct {
	ErrorObject *shared.ErrorObject `json:"error"`
	StatusCode  int                 `json:"-"`
}

func NewErrorResponse(msg error) *ErrorResponse {
	return &ErrorResponse{
		ErrorObject: &shared.ErrorObject{
			Code:    "error",
			Message: msg.Error(),
			Type:    "error",
		},
	}
}

func (a *ErrorResponse) Error() string {
	if a.ErrorObject == nil {
		return ""
	}
	return a.ErrorObject.Message
}
