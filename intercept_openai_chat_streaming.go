package aibridge

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/coder/aibridge/mcp"
	"github.com/google/uuid"
	"github.com/openai/openai-go/v2"
	"github.com/openai/openai-go/v2/packages/ssestream"

	"cdr.dev/slog"
)

var _ Interceptor = &OpenAIStreamingChatInterception{}

type OpenAIStreamingChatInterception struct {
	OpenAIChatInterceptionBase
}

func NewOpenAIStreamingChatInterception(id uuid.UUID, req *ChatCompletionNewParamsWrapper, baseURL, key string) *OpenAIStreamingChatInterception {
	return &OpenAIStreamingChatInterception{OpenAIChatInterceptionBase: OpenAIChatInterceptionBase{
		id:      id,
		req:     req,
		baseURL: baseURL,
		key:     key,
	}}
}

func (i *OpenAIStreamingChatInterception) Setup(logger slog.Logger, recorder Recorder, mcpProxy mcp.ServerProxier) {
	i.OpenAIChatInterceptionBase.Setup(logger.Named("streaming"), recorder, mcpProxy)
}

// ProcessRequest handles a request to /v1/chat/completions.
// See https://platform.openai.com/docs/api-reference/chat-streaming/streaming.
//
// It will inject any tools which have been provided by the [mcp.ServerProxier].
//
// When a response from the server includes an event indicating that a tool must be invoked, a conditional
// flow takes place:
//
// a) if the tool is not injected (i.e. defined by the client), relay the event unmodified
// b) if the tool is injected, it will be invoked by the [mcp.ServerProxier] in the remote MCP server, and its
// results relayed to the SERVER. The response from the server will be handled synchronously, and this loop
// can continue until all injected tool invocations are completed and the response is relayed to the client.
func (i *OpenAIStreamingChatInterception) ProcessRequest(w http.ResponseWriter, r *http.Request) error {
	if i.req == nil {
		return fmt.Errorf("developer error: req is nil")
	}

	// Include token usage.
	i.req.StreamOptions.IncludeUsage = openai.Bool(true)

	i.injectTools()

	// Allow us to interrupt watch via cancel.
	ctx, cancel := context.WithCancel(r.Context())
	defer cancel()
	r = r.WithContext(ctx) // Rewire context for SSE cancellation.

	client := newOpenAIClient(i.baseURL, i.key)
	logger := i.logger.With(slog.F("model", i.req.Model))

	streamCtx, streamCancel := context.WithCancelCause(ctx)
	defer streamCancel(errors.New("deferred"))

	// events will either terminate when shutdown after interaction with upstream completes, or when streamCtx is done.
	events := newEventStream(streamCtx, logger.Named("sse-sender"), nil)
	go events.run(w, r)
	defer func() {
		_ = events.Shutdown(streamCtx) // Catch-all in case it doesn't get shutdown after stream completes.
	}()

	// TODO: implement parallel tool calls.
	// TODO: don't send if not supported by model (i.e. o4-mini).
	if len(i.req.Tools) > 0 { // If no tools are specified but this setting is set, it'll cause a 400 Bad Request.
		i.req.ParallelToolCalls = openai.Bool(false)
	}

	prompt, err := i.req.LastUserPrompt()
	if err != nil {
		logger.Warn(ctx, "failed to retrieve last user prompt", slog.Error(err))
	}

	var (
		stream          *ssestream.Stream[openai.ChatCompletionChunk]
		lastErr         error
		interceptionErr error
		lastUsage       openai.CompletionUsage
		cumulativeUsage openai.CompletionUsage
	)
	for {
		var pendingToolCalls []openai.FinishedChatCompletionToolCall

		stream = client.Chat.Completions.NewStreaming(streamCtx, i.req.ChatCompletionNewParams)
		var acc openai.ChatCompletionAccumulator
		for stream.Next() {
			chunk := stream.Current()
			acc.AddChunk(chunk)

			shouldRelayChunk := true
			if toolCall, ok := acc.JustFinishedToolCall(); ok {
				if i.mcpProxy != nil && i.mcpProxy.GetTool(toolCall.Name) != nil {
					pendingToolCalls = append(pendingToolCalls, toolCall)
					// Don't relay this chunk back; we'll handle it transparently.
					shouldRelayChunk = false
				} else {
					// Don't intercept and handle builtin tools.
					_ = i.recorder.RecordToolUsage(streamCtx, &ToolUsageRecord{
						InterceptionID: i.ID().String(),
						MsgID:          chunk.ID,
						Tool:           toolCall.Name,
						Args:           i.unmarshalArgs(toolCall.Arguments),
						Injected:       false,
					})
				}
			}

			if len(pendingToolCalls) > 0 {
				// Any chunks following a tool call invocation should not be relayed.
				shouldRelayChunk = false
			}

			lastUsage = chunk.Usage
			cumulativeUsage = sumUsage(cumulativeUsage, chunk.Usage)

			if shouldRelayChunk {
				// If usage information is available, relay the cumulative usage once all tool invocations have completed.
				if chunk.Usage.CompletionTokens > 0 {
					chunk.Usage = cumulativeUsage
				}

				// Overwrite response identifier since proxy obscures injected tool call invocations.
				chunk.ID = i.ID().String()
				payload, err := i.marshal(chunk)
				if err != nil {
					logger.Warn(ctx, "failed to marshal chunk", slog.Error(err), chunk.RawJSON())
					lastErr = fmt.Errorf("marshal chunk: %w", err)
					break
				}
				if err := events.Send(streamCtx, payload); err != nil {
					logger.Warn(ctx, "failed to relay chunk", slog.Error(err))
					lastErr = fmt.Errorf("relay chunk: %w", err)
					break
				}
			}
		}

		if prompt != nil {
			_ = i.recorder.RecordPromptUsage(streamCtx, &PromptUsageRecord{
				InterceptionID: i.ID().String(),
				MsgID:          acc.ID,
				Prompt:         *prompt,
			})
		}

		if lastUsage.CompletionTokens > 0 {
			// If the usage information is set, track it.
			// The API will send usage information when the response terminates, which will happen if a tool call is invoked.
			_ = i.recorder.RecordTokenUsage(streamCtx, &TokenUsageRecord{
				InterceptionID: i.ID().String(),
				MsgID:          acc.ID,
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
		}

		// Check if the stream encountered any errors.
		if streamErr := stream.Err(); streamErr != nil {
			if isUnrecoverableError(streamErr) {
				logger.Debug(ctx, "stream terminated", slog.Error(streamErr))
				// We can't reflect an error back if there's a connection error or the request context was canceled.
			} else if oaiErr := getOpenAIErrorResponse(streamErr); oaiErr != nil {
				logger.Warn(ctx, "openai stream error", slog.Error(streamErr))
				interceptionErr = fmt.Errorf("stream error: %w", oaiErr)
			} else {
				logger.Warn(ctx, "unknown error", slog.Error(streamErr))
				// Unfortunately, the OpenAI SDK does not support parsing errors received in the stream
				// into known types (i.e. [shared.OverloadedError]).
				// See https://github.com/openai/openai-go/blob/v2.7.0/packages/ssestream/ssestream.go#L171
				// All it does is wrap the payload in an error - which is all we can return, currently.
				interceptionErr = newOpenAIErr(fmt.Errorf("unknown stream error: %w", streamErr))
			}
		} else if lastErr != nil {
			// Otherwise check if any logical errors occurred during processing.
			logger.Warn(ctx, "stream failed", slog.Error(lastErr))
			interceptionErr = newOpenAIErr(fmt.Errorf("processing error: %w", lastErr))
		}

		if interceptionErr != nil {
			payload, err := i.marshal(interceptionErr)
			if err != nil {
				logger.Warn(ctx, "failed to marshal error", slog.Error(err), slog.F("error_payload", slog.F("%+v", interceptionErr)))
			} else if err := events.Send(streamCtx, payload); err != nil {
				logger.Warn(ctx, "failed to relay error", slog.Error(err), slog.F("payload", payload))
			}
		}

		if len(pendingToolCalls) == 0 {
			break
		}

		appendedPrevMsg := false
		for _, tc := range pendingToolCalls {
			if i.mcpProxy == nil {
				continue
			}

			tool := i.mcpProxy.GetTool(tc.Name)
			if tool == nil {
				// Not a known tool, don't do anything.
				logger.Warn(streamCtx, "pending tool call for non-managed tool, skipping", slog.F("tool", tc.Name))
				continue
			}

			// Only do this once.
			if !appendedPrevMsg {
				// Append the whole message from this stream as context since we'll be sending a new request with the tool results.
				i.req.Messages = append(i.req.Messages, acc.Choices[len(acc.Choices)-1].Message.ToParam())
				appendedPrevMsg = true
			}

			args := i.unmarshalArgs(tc.Arguments)
			res, err := tool.Call(streamCtx, args)

			_ = i.recorder.RecordToolUsage(streamCtx, &ToolUsageRecord{
				InterceptionID:  i.ID().String(),
				MsgID:           acc.ID,
				ServerURL:       &tool.ServerURL,
				Tool:            tool.Name,
				Args:            args,
				Injected:        true,
				InvocationError: err,
			})

			if err != nil {
				// Always provide a tool_result even if the tool call failed.
				errorJSON, _ := json.Marshal(i.newErrorResponse(err))
				i.req.Messages = append(i.req.Messages, openai.ToolMessage(string(errorJSON), tc.ID))
				continue
			}

			var out strings.Builder
			if err := json.NewEncoder(&out).Encode(res); err != nil {
				logger.Warn(ctx, "failed to encode tool response", slog.Error(err))
				// Always provide a tool_result even if encoding failed.
				errorJSON, _ := json.Marshal(i.newErrorResponse(err))
				i.req.Messages = append(i.req.Messages, openai.ToolMessage(string(errorJSON), tc.ID))
				continue
			}

			i.req.Messages = append(i.req.Messages, openai.ToolMessage(out.String(), tc.ID))
		}
	}

	// Send termination marker.
	if err := events.sendRaw(streamCtx, i.encodeForStream([]byte("[DONE]"))); err != nil {
		logger.Debug(ctx, "failed to send termination marker", slog.Error(err))
	}

	// Give the events stream 30 seconds (TODO: configurable) to gracefully shutdown.
	shutdownCtx, shutdownCancel := context.WithTimeout(ctx, time.Second*30)
	defer shutdownCancel()
	if err = events.Shutdown(shutdownCtx); err != nil {
		logger.Warn(ctx, "event stream shutdown", slog.Error(err))
	}

	if err != nil {
		streamCancel(fmt.Errorf("stream err: %w", err))
	} else {
		streamCancel(errors.New("gracefully done"))
	}

	return interceptionErr
}

func (i *OpenAIStreamingChatInterception) marshal(payload any) ([]byte, error) {
	data, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("marshal payload: %w", err)
	}

	return i.encodeForStream(data), nil
}

func (i *OpenAIStreamingChatInterception) encodeForStream(payload []byte) []byte {
	var buf bytes.Buffer
	buf.WriteString("data: ")
	buf.Write(payload)
	buf.WriteString("\n\n")
	return buf.Bytes()
}
