package aibridge

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/coder/aibridge/mcp"
	"github.com/google/uuid"
	"github.com/openai/openai-go/v2"
	"github.com/openai/openai-go/v2/option"

	"cdr.dev/slog"
)

var _ Interceptor = &OpenAIBlockingChatInterception{}

type OpenAIBlockingChatInterception struct {
	OpenAIChatInterceptionBase
}

func NewOpenAIBlockingChatInterception(id uuid.UUID, req *ChatCompletionNewParamsWrapper, baseURL, key string) *OpenAIBlockingChatInterception {
	return &OpenAIBlockingChatInterception{OpenAIChatInterceptionBase: OpenAIChatInterceptionBase{
		id:      id,
		req:     req,
		baseURL: baseURL,
		key:     key,
	}}
}

func (s *OpenAIBlockingChatInterception) Setup(logger slog.Logger, recorder Recorder, mcpProxy mcp.ServerProxier) {
	s.OpenAIChatInterceptionBase.Setup(logger.Named("blocking"), recorder, mcpProxy)
}

func (i *OpenAIBlockingChatInterception) ProcessRequest(w http.ResponseWriter, r *http.Request) error {
	if i.req == nil {
		return fmt.Errorf("developer error: req is nil")
	}

	ctx := r.Context()
	client := newOpenAIClient(i.baseURL, i.key)
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
		var opts []option.RequestOption
		opts = append(opts, option.WithRequestTimeout(time.Second*60)) // TODO: configurable timeout

		completion, err = client.Chat.Completions.New(ctx, i.req.ChatCompletionNewParams, opts...)
		if err != nil {
			break
		}

		if prompt != nil {
			_ = i.recorder.RecordPromptUsage(ctx, &PromptUsageRecord{
				InterceptionID: i.ID().String(),
				MsgID:          completion.ID,
				Prompt:         *prompt,
			})
		}

		lastUsage := completion.Usage
		cumulativeUsage = sumUsage(cumulativeUsage, completion.Usage)

		_ = i.recorder.RecordTokenUsage(ctx, &TokenUsageRecord{
			InterceptionID: i.ID().String(),
			MsgID:          completion.ID,
			Input:          lastUsage.PromptTokens,
			Output:         lastUsage.CompletionTokens,
			Metadata: Metadata{
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

			var (
				args map[string]string
				buf  bytes.Buffer
			)
			_ = json.NewEncoder(&buf).Encode(tc.Function.Arguments)
			_ = json.NewDecoder(&buf).Decode(&args)
			res, err := tool.Call(ctx, args)

			_ = i.recorder.RecordToolUsage(ctx, &ToolUsageRecord{
				InterceptionID:  i.ID().String(),
				MsgID:           completion.ID,
				ServerURL:       &tool.ServerURL,
				Tool:            tool.Name,
				Args:            i.unmarshalArgs(tc.Function.Arguments),
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

	// TODO: these probably have to be formatted as JSON errs?
	if err != nil {
		if isConnError(err) {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return fmt.Errorf("upstream connection closed: %w", err)
		}

		logger.Warn(ctx, "openai API error", slog.Error(err))
		var apierr *openai.Error
		if errors.As(err, &apierr) {
			http.Error(w, apierr.Message, apierr.StatusCode)
			return fmt.Errorf("api error: %w", apierr)
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
