package aibridge

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"time"

	"cdr.dev/slog"
	"github.com/coder/aibridge/mcp"
	"github.com/google/uuid"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/responses"
)

type OpenAIBlockingResponsesInterception struct {
	OpenAIResponsesInterceptionBase
}

func NewOpenAIBlockingResponsesInterception(id uuid.UUID, req *ResponsesNewParamsWrapper, baseURL, key string) *OpenAIBlockingResponsesInterception {
	return &OpenAIBlockingResponsesInterception{OpenAIResponsesInterceptionBase: OpenAIResponsesInterceptionBase{
		id:      id,
		req:     req,
		baseURL: baseURL,
		key:     key,
	}}
}

func (i *OpenAIBlockingResponsesInterception) Setup(logger slog.Logger, recorder Recorder, mcpProxy mcp.ServerProxier) {
	i.OpenAIResponsesInterceptionBase.Setup(logger.Named("responses"), recorder, mcpProxy)
}

func (i *OpenAIBlockingResponsesInterception) ProcessRequest(w http.ResponseWriter, r *http.Request) error {
	if i.req == nil {
		return fmt.Errorf("developer error: req is nil")
	}
	if i.req.Background.Value {
		return fmt.Errorf("background requests are currently not supported by aibridge")
	}

	ctx := r.Context()
	client := newOpenAIClient(i.baseURL, i.key)
	logger := i.logger.With(slog.F("model", i.req.Model))

	var (
		cumulativeUsage responses.ResponseUsage
		response        *responses.Response
		err             error
	)

	i.injectTools()

	// TODO: implement parallel tool calls.
	// TODO: safe to send if there are zero tools?
	i.req.ParallelToolCalls = openai.Bool(false)

	prompt, err := i.req.LastUserPrompt()
	if err != nil {
		logger.Warn(ctx, "failed to retrieve last user prompt", slog.Error(err))
	}

	for {
		var opts []option.RequestOption
		opts = append(opts, option.WithRequestTimeout(time.Second*60)) // TODO: configurable timeout

		// TODO: only allow a single tool call per request.
		response, err = client.Responses.New(ctx, i.req.ResponseNewParams, opts...)
		if err != nil {
			break
		}

		if prompt != nil {
			_ = i.recorder.RecordPromptUsage(ctx, &PromptUsageRecord{
				InterceptionID: i.ID().String(),
				MsgID:          response.ID,
				Prompt:         *prompt,
			})
			prompt = nil
		}

		lastUsage := response.Usage
		cumulativeUsage = sumOpenAIResponsesUsage(cumulativeUsage, response.Usage)

		_ = i.recorder.RecordTokenUsage(ctx, &TokenUsageRecord{
			InterceptionID: i.ID().String(),
			MsgID:          response.ID,
			Input:          calculateOpenAIResponsesActualInputTokenUsage(lastUsage),
			Output:         lastUsage.OutputTokens,
			Metadata: Metadata{
				"input_cached":     lastUsage.InputTokensDetails.CachedTokens,
				"output_reasoning": lastUsage.OutputTokensDetails.ReasoningTokens,
			},
		})

		// Check if we have custom tool calls to process.
		var pendingToolCalls []responses.ResponseCustomToolCall
		for _, output := range response.Output {
			if output.Type != "custom_tool_call" {
				continue
			}

			customToolCall := output.AsCustomToolCall()
			if i.mcpProxy != nil && i.mcpProxy.GetTool(customToolCall.Name) != nil {
				pendingToolCalls = append(pendingToolCalls, customToolCall)
			} else {
				_ = i.recorder.RecordToolUsage(ctx, &ToolUsageRecord{
					InterceptionID: i.ID().String(),
					MsgID:          response.ID,
					Tool:           customToolCall.Name,
					Args:           customToolCall.Input,
					Injected:       false,
					Metadata: Metadata{
						"custom_tool_call.id":      customToolCall.ID,
						"custom_tool_call.call_id": customToolCall.CallID,
					},
				})
			}
		}

		// TODO: generate tool call records for each server-side tool call that
		// was returned by the API.

		// If no injected tool calls, we're done.
		if len(pendingToolCalls) == 0 {
			break
		}

		// Append everything from the current response to the next request.
		i.appendResponseOutput(response)

		// Handle each pending tool call.
		for _, tc := range pendingToolCalls {
			if i.mcpProxy == nil {
				i.appendToolCallError(tc, fmt.Sprintf("Error: tool %s not found", tc.Name))
				continue
			}

			tool := i.mcpProxy.GetTool(tc.Name)
			if tool == nil {
				// Not a known tool, don't do anything.
				logger.Warn(ctx, "pending tool call for non-managed tool, skipping", slog.F("tool", tc.Name))
				i.appendToolCallError(tc, fmt.Sprintf("Error: tool %s not found", tc.Name))
				continue
			}

			// Parse the tool call arguments.
			args, err := parseToolCallArguments(tc.Input)
			if err != nil {
				logger.Warn(ctx, "failed to parse tool call arguments", slog.Error(err))
				i.appendToolCallError(tc, fmt.Sprintf("Error: parse tool call arguments: %v", err))
				continue
			}
			toolRes, toolErr := tool.Call(ctx, args)

			_ = i.recorder.RecordToolUsage(ctx, &ToolUsageRecord{
				InterceptionID:  i.ID().String(),
				MsgID:           response.ID,
				ServerURL:       &tool.ServerURL,
				Tool:            tool.Name,
				Args:            args,
				Injected:        true,
				InvocationError: toolErr,
			})

			if toolErr != nil {
				i.appendToolCallError(tc, fmt.Sprintf("Error: calling tool: %v", toolErr))
				continue
			}

			var out strings.Builder
			if err := json.NewEncoder(&out).Encode(toolRes); err != nil {
				logger.Warn(ctx, "failed to encode tool response", slog.Error(err))
				i.appendToolCallError(tc, fmt.Sprintf("Error: encode tool response: %v", err))
				continue
			}

			i.appendToolCallOutput(tc, out.String())
		}
	}

	// TODO: these probably have to be formatted as JSON errs?
	if err != nil {
		if isConnError(err) {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return fmt.Errorf("upstream connection closed: %w", err)
		}

		logger.Warn(ctx, "openai responses API error", slog.Error(err))
		var apierr *responses.Error
		if errors.As(err, &apierr) {
			i.writeErrorResponse(w, apierr.StatusCode, apierr.Code, apierr.Message)
			return fmt.Errorf("responses api error: %w", apierr)
		}

		i.writeErrorResponse(w, http.StatusInternalServerError, "internal_server_error", "Internal server error")
		return fmt.Errorf("responses API failed: %w", err)
	}

	if response == nil {
		return nil
	}

	// Overwrite response identifier since proxy obscures injected tool call
	// invocations.
	response.ID = i.ID().String()

	// Update the cumulative usage in the final response.
	response.Usage = cumulativeUsage

	out, err := json.Marshal(response)
	if err != nil {
		i.logger.Warn(ctx, "failed to marshal response JSON", slog.Error(err))
		i.writeErrorResponse(w, http.StatusInternalServerError, "internal_server_error", "Internal server error")
		return fmt.Errorf("marshal response JSON: %w", err)
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(out)

	return nil
}
