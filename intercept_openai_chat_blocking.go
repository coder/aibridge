package aibridge

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/coder/aibridge/mcp"
	"github.com/coder/aibridge/tracing"
	"github.com/google/uuid"
	"github.com/openai/openai-go/v2"
	"github.com/openai/openai-go/v2/option"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"

	"cdr.dev/slog"
)

var _ Interceptor = &OpenAIBlockingChatInterception{}

type OpenAIBlockingChatInterception struct {
	OpenAIChatInterceptionBase
}

func NewOpenAIBlockingChatInterception(id uuid.UUID, req *ChatCompletionNewParamsWrapper, cfg *OpenAIConfig, tracer trace.Tracer) *OpenAIBlockingChatInterception {
	return &OpenAIBlockingChatInterception{OpenAIChatInterceptionBase: OpenAIChatInterceptionBase{
		id:     id,
		cfg:    cfg,
		req:    req,
		tracer: tracer,
	}}
}

func (s *OpenAIBlockingChatInterception) Setup(logger slog.Logger, recorder Recorder, mcpProxy mcp.ServerProxier) {
	s.OpenAIChatInterceptionBase.Setup(logger.Named("blocking"), recorder, mcpProxy)
}

func (s *OpenAIBlockingChatInterception) Streaming() bool {
	return false
}

func (s *OpenAIBlockingChatInterception) TraceAttributes(r *http.Request) []attribute.KeyValue {
	return s.OpenAIChatInterceptionBase.baseTraceAttributes(r, false)
}

func (i *OpenAIBlockingChatInterception) ProcessRequest(w http.ResponseWriter, r *http.Request) (outErr error) {
	if i.req == nil {
		return fmt.Errorf("developer error: req is nil")
	}

	ctx, span := i.tracer.Start(r.Context(), "Intercept.ProcessRequest", trace.WithAttributes(tracing.InterceptionAttributesFromContext(r.Context())...))
	defer tracing.EndSpanErr(span, &outErr)

	svc := i.newChatCompletionService()
	logger := i.logger.With(slog.F("model", i.req.Model))

	var (
		cumulativeUsage openai.CompletionUsage
		completion      *openai.ChatCompletion
		err             error
	)

	i.injectTools()

	prompt, err := i.req.LastUserPrompt()
	if err != nil {
		logger.Warn(ctx, "failed to retrieve last user prompt", slog.Error(err))
	}

	for {
		// TODO add outer loop span (https://github.com/coder/aibridge/issues/67)
		var opts []option.RequestOption
		opts = append(opts, option.WithRequestTimeout(time.Second*60)) // TODO: configurable timeout

		completion, err = i.newChatCompletion(ctx, svc, opts)
		if err != nil {
			break
		}

		if prompt != nil {
			_ = i.recorder.RecordPromptUsage(ctx, &PromptUsageRecord{
				InterceptionID: i.ID().String(),
				MsgID:          completion.ID,
				Prompt:         *prompt,
			})
			prompt = nil
		}

		lastUsage := completion.Usage
		cumulativeUsage = sumUsage(cumulativeUsage, completion.Usage)

		_ = i.recorder.RecordTokenUsage(ctx, &TokenUsageRecord{
			InterceptionID: i.ID().String(),
			MsgID:          completion.ID,
			Input:          calculateActualInputTokenUsage(lastUsage),
			Output:         lastUsage.CompletionTokens,
			ExtraTokenTypes: map[string]int64{
				"prompt_audio":                   lastUsage.PromptTokensDetails.AudioTokens,
				"prompt_cached":                  lastUsage.PromptTokensDetails.CachedTokens,
				"completion_accepted_prediction": lastUsage.CompletionTokensDetails.AcceptedPredictionTokens,
				"completion_rejected_prediction": lastUsage.CompletionTokensDetails.RejectedPredictionTokens,
				"completion_audio":               lastUsage.CompletionTokensDetails.AudioTokens,
				"completion_reasoning":           lastUsage.CompletionTokensDetails.ReasoningTokens,
			},
		})

		// Check if we have tool calls to process.
		var pendingToolCalls []openai.ChatCompletionMessageToolCallUnion
		if len(completion.Choices) > 0 && completion.Choices[0].Message.ToolCalls != nil {
			for _, toolCall := range completion.Choices[0].Message.ToolCalls {
				if i.mcpProxy != nil && i.mcpProxy.GetTool(toolCall.Function.Name) != nil {
					pendingToolCalls = append(pendingToolCalls, toolCall)
				} else {
					_ = i.recorder.RecordToolUsage(ctx, &ToolUsageRecord{
						InterceptionID: i.ID().String(),
						MsgID:          completion.ID,
						Tool:           toolCall.Function.Name,
						Args:           i.unmarshalArgs(toolCall.Function.Arguments),
						Injected:       false,
					})
				}
			}
		}

		// If no injected tool calls, we're done.
		if len(pendingToolCalls) == 0 {
			break
		}

		appendedPrevMsg := false
		for _, tc := range pendingToolCalls {
			if i.mcpProxy == nil {
				continue
			}

			tool := i.mcpProxy.GetTool(tc.Function.Name)
			if tool == nil {
				// Not a known tool, don't do anything.
				logger.Warn(ctx, "pending tool call for non-managed tool, skipping", slog.F("tool", tc.Function.Name))
				continue
			}
			// Only do this once.
			if !appendedPrevMsg {
				// Append the whole message from this stream as context since we'll be sending a new request with the tool results.
				i.req.Messages = append(i.req.Messages, completion.Choices[0].Message.ToParam())
				appendedPrevMsg = true
			}

			args := i.unmarshalArgs(tc.Function.Arguments)
			res, err := tool.Call(ctx, args, i.tracer)
			_ = i.recorder.RecordToolUsage(ctx, &ToolUsageRecord{
				InterceptionID:  i.ID().String(),
				MsgID:           completion.ID,
				ServerURL:       &tool.ServerURL,
				Tool:            tool.Name,
				Args:            args,
				Injected:        true,
				InvocationError: err,
			})

			if err != nil {
				// Always provide a tool result even if the tool call failed
				errorResponse := map[string]interface{}{
					// TODO: interception ID?
					"error":   true,
					"message": err.Error(),
				}
				errorJSON, _ := json.Marshal(errorResponse)
				i.req.Messages = append(i.req.Messages, openai.ToolMessage(string(errorJSON), tc.ID))
				continue
			}

			var out strings.Builder
			if err := json.NewEncoder(&out).Encode(res); err != nil {
				logger.Warn(ctx, "failed to encode tool response", slog.Error(err))
				// Always provide a tool result even if encoding failed
				errorResponse := map[string]interface{}{
					// TODO: interception ID?
					"error":   true,
					"message": err.Error(),
				}
				errorJSON, _ := json.Marshal(errorResponse)
				i.req.Messages = append(i.req.Messages, openai.ToolMessage(string(errorJSON), tc.ID))
				continue
			}

			i.req.Messages = append(i.req.Messages, openai.ToolMessage(out.String(), tc.ID))
		}
	}

	if err != nil {
		if isConnError(err) {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return fmt.Errorf("upstream connection closed: %w", err)
		}

		if apiErr := getOpenAIErrorResponse(err); apiErr != nil {
			i.writeUpstreamError(w, apiErr)
			return fmt.Errorf("openai API error: %w", err)
		}

		http.Error(w, err.Error(), http.StatusInternalServerError)
		return fmt.Errorf("chat completion failed: %w", err)
	}

	if completion == nil {
		return nil
	}

	// Overwrite response identifier since proxy obscures injected tool call invocations.
	completion.ID = i.ID().String()

	// Update the cumulative usage in the final response.
	if completion.Usage.CompletionTokens > 0 {
		completion.Usage = cumulativeUsage
	}

	w.Header().Set("Content-Type", "application/json")
	out, err := json.Marshal(completion)
	if err != nil {
		out, _ = json.Marshal(i.newErrorResponse(fmt.Errorf("failed to marshal response: %w", err)))
		w.WriteHeader(http.StatusInternalServerError)
	} else {
		w.WriteHeader(http.StatusOK)
	}

	_, _ = w.Write(out)

	return nil
}

func (i *OpenAIBlockingChatInterception) newChatCompletion(ctx context.Context, svc openai.ChatCompletionService, opts []option.RequestOption) (_ *openai.ChatCompletion, outErr error) {
	ctx, span := i.tracer.Start(ctx, "Intercept.ProcessRequest.Upstream", trace.WithAttributes(tracing.InterceptionAttributesFromContext(ctx)...))
	defer tracing.EndSpanErr(span, &outErr)

	return svc.New(ctx, i.req.ChatCompletionNewParams, opts...)
}
