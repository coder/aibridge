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
	"github.com/coder/aibridge/tracing"
	"github.com/google/uuid"
	"github.com/openai/openai-go/v2"
	"github.com/openai/openai-go/v2/packages/ssestream"
	"github.com/tidwall/sjson"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"

	"cdr.dev/slog"
)

var _ Interceptor = &OpenAIStreamingChatInterception{}

type OpenAIStreamingChatInterception struct {
	OpenAIChatInterceptionBase
}

func NewOpenAIStreamingChatInterception(id uuid.UUID, req *ChatCompletionNewParamsWrapper, baseURL, key string, tracer trace.Tracer) *OpenAIStreamingChatInterception {
	return &OpenAIStreamingChatInterception{OpenAIChatInterceptionBase: OpenAIChatInterceptionBase{
		id:      id,
		req:     req,
		baseURL: baseURL,
		key:     key,
		tracer:  tracer,
	}}
}

func (i *OpenAIStreamingChatInterception) Setup(logger slog.Logger, recorder Recorder, mcpProxy mcp.ServerProxier) {
	i.OpenAIChatInterceptionBase.Setup(logger.Named("streaming"), recorder, mcpProxy)
}

func (i *OpenAIStreamingChatInterception) Streaming() bool {
	return true
}

func (s *OpenAIStreamingChatInterception) TraceAttributes(r *http.Request) []attribute.KeyValue {
	return s.OpenAIChatInterceptionBase.baseTraceAttributes(r, true)
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
func (i *OpenAIStreamingChatInterception) ProcessRequest(w http.ResponseWriter, r *http.Request) (outErr error) {
	if i.req == nil {
		return fmt.Errorf("developer error: req is nil")
	}

	ctx, span := i.tracer.Start(r.Context(), "Intercept.ProcessRequest", trace.WithAttributes(tracing.InterceptionAttributesFromContext(r.Context())...))
	defer tracing.EndSpanErr(span, &outErr)

	// Include token usage.
	i.req.StreamOptions.IncludeUsage = openai.Bool(true)

	i.injectTools()

	// Allow us to interrupt watch via cancel.
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	r = r.WithContext(ctx) // Rewire context for SSE cancellation.

	svc := i.newCompletionsService(i.baseURL, i.key)
	logger := i.logger.With(slog.F("model", i.req.Model))

	streamCtx, streamCancel := context.WithCancelCause(ctx)
	defer streamCancel(errors.New("deferred"))

	// events will either terminate when shutdown after interaction with upstream completes, or when streamCtx is done.
	events := newEventStream(streamCtx, logger.Named("sse-sender"), nil)
	go events.start(w, r)
	defer func() {
		_ = events.Shutdown(streamCtx) // Catch-all in case it doesn't get shutdown after stream completes.
	}()

	// TODO: implement parallel tool calls.
	// TODO: don't send if not supported by model (i.e. o4-mini).
	if len(i.req.Tools) > 0 { // If no tools are specified but this setting is set, it'll cause a 400 Bad Request.
		i.req.ParallelToolCalls = openai.Bool(false)
	}

	// Force responses to only have one choice.
	// It's unnecessary to generate multiple responses, and would complicate our stream processing logic if
	// multiple choices were returned.
	i.req.N = openai.Int(1)

	prompt, err := i.req.LastUserPrompt()
	if err != nil {
		logger.Warn(ctx, "failed to retrieve last user prompt", slog.Error(err))
	}

	var (
		stream          *ssestream.Stream[openai.ChatCompletionChunk]
		lastErr         error
		interceptionErr error
	)
	for {
		// TODO add outer loop span (https://github.com/coder/aibridge/issues/67)
		stream = i.newStream(streamCtx, svc)
		processor := newStreamProcessor(streamCtx, i.logger.Named("stream-processor"), i.getInjectedToolByName)

		var toolCall *openai.FinishedChatCompletionToolCall

		for stream.Next() {
			chunk := stream.Current()

			canRelay := processor.process(chunk)
			if toolCall == nil {
				toolCall = processor.getToolCall()
			}

			if !canRelay {
				// The chunk must not be sent to the client because it contains an injected tool call.
				continue
			}

			// Marshal and relay chunk to client.
			payload, err := i.marshalChunk(&chunk, i.ID(), processor)
			if err != nil {
				logger.Warn(ctx, "failed to marshal chunk", slog.Error(err), chunk.RawJSON())
				lastErr = fmt.Errorf("marshal chunk: %w", err)
				break
			}
			if err := events.Send(ctx, payload); err != nil {
				logger.Warn(ctx, "failed to relay chunk", slog.Error(err))
				lastErr = fmt.Errorf("relay chunk: %w", err)
				break
			}
		}

		// Builtin tools are not intercepted.
		if toolCall != nil && i.getInjectedToolByName(toolCall.Name) == nil {
			_ = i.recorder.RecordToolUsage(streamCtx, &ToolUsageRecord{
				InterceptionID: i.ID().String(),
				MsgID:          processor.getMsgID(),
				Tool:           toolCall.Name,
				Args:           i.unmarshalArgs(toolCall.Arguments),
				Injected:       false,
			})
			toolCall = nil
		}

		if prompt != nil {
			_ = i.recorder.RecordPromptUsage(streamCtx, &PromptUsageRecord{
				InterceptionID: i.ID().String(),
				MsgID:          processor.getMsgID(),
				Prompt:         *prompt,
			})
			prompt = nil
		}

		if lastUsage := processor.getLastUsage(); lastUsage.CompletionTokens > 0 {
			// If the usage information is set, track it.
			// The API will send usage information when the response terminates, which will happen if a tool call is invoked.
			_ = i.recorder.RecordTokenUsage(streamCtx, &TokenUsageRecord{
				InterceptionID: i.ID().String(),
				MsgID:          processor.getMsgID(),
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
		}

		if events.isStreaming() {
			// Check if the stream encountered any errors.
			if streamErr := stream.Err(); streamErr != nil {
				if isUnrecoverableError(streamErr) {
					logger.Debug(ctx, "stream terminated", slog.Error(streamErr))
					// We can't reflect an error back if there's a connection error or the request context was canceled.
				} else if oaiErr := getOpenAIErrorResponse(streamErr); oaiErr != nil {
					logger.Warn(ctx, "openai stream error", slog.Error(streamErr))
					interceptionErr = oaiErr
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
				payload, err := i.marshalErr(interceptionErr)
				if err != nil {
					logger.Warn(ctx, "failed to marshal error", slog.Error(err), slog.F("error_payload", slog.F("%+v", interceptionErr)))
				} else if err := events.Send(streamCtx, payload); err != nil {
					logger.Warn(ctx, "failed to relay error", slog.Error(err), slog.F("payload", payload))
				}
			}
		} else {
			// Stream has not started yet; write to response if present.
			i.writeUpstreamError(w, getOpenAIErrorResponse(stream.Err()))
		}

		// No tool call, nothing more to do.
		if toolCall == nil {
			break
		}

		tool := i.getInjectedToolByName(toolCall.Name)
		if tool == nil {
			// Not a known tool, don't do anything.
			logger.Warn(streamCtx, "pending tool call for non-injected tool, this is unexpected", slog.F("tool", toolCall.Name))
			break
		}

		// Invoke the injected tool, and use the tool result to make a subsequent request to the upstream.
		// Append the completion from this stream as context.
		i.req.Messages = append(i.req.Messages, processor.getLastCompletion().ToParam())

		id := toolCall.ID
		args := i.unmarshalArgs(toolCall.Arguments)
		toolRes, toolErr := tool.Call(streamCtx, args, i.tracer)
		_ = i.recorder.RecordToolUsage(streamCtx, &ToolUsageRecord{
			InterceptionID:  i.ID().String(),
			MsgID:           processor.getMsgID(),
			ServerURL:       &tool.ServerURL,
			Tool:            tool.Name,
			Args:            args,
			Injected:        true,
			InvocationError: toolErr,
		})

		// Reset.
		toolCall = nil

		if toolErr != nil {
			// Always provide a tool_result even if the tool call failed.
			errorJSON, _ := json.Marshal(i.newErrorResponse(toolErr))
			i.req.Messages = append(i.req.Messages, openai.ToolMessage(string(errorJSON), id))
			continue
		}

		var out strings.Builder
		if err := json.NewEncoder(&out).Encode(toolRes); err != nil {
			logger.Warn(ctx, "failed to encode tool response", slog.Error(err))
			// Always provide a tool_result even if encoding failed.
			errorJSON, _ := json.Marshal(i.newErrorResponse(err))
			i.req.Messages = append(i.req.Messages, openai.ToolMessage(string(errorJSON), id))
			continue
		}

		i.req.Messages = append(i.req.Messages, openai.ToolMessage(out.String(), id))
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

func (i *OpenAIStreamingChatInterception) getInjectedToolByName(name string) *mcp.Tool {
	if i.mcpProxy == nil {
		return nil
	}

	return i.mcpProxy.GetTool(name)
}

// Mashals received stream chunk.
// Overrides id (since proxy obscures injected tool call invocations).
// If usage field was set in original chunk overrides it to culminative usage.
//
// sjson is used instead of normal struct marshaling so forwarded data
// is as close to the original as possible. Structs from openai library lack
// `omitzero/omitempty` annotations which adds additional empty fields
// when marshaling structs. Those additional empty fields can break Codex client.
func (i *OpenAIStreamingChatInterception) marshalChunk(chunk *openai.ChatCompletionChunk, id uuid.UUID, prc *openAIStreamProcessor) ([]byte, error) {
	sj, err := sjson.Set(chunk.RawJSON(), "id", id.String())
	if err != nil {
		return nil, fmt.Errorf("marshal chunk id failed: %w", err)
	}

	// If usage information is available, relay the cumulative usage once all tool invocations have completed.
	if chunk.JSON.Usage.Valid() {
		u := prc.getCumulativeUsage()
		sj, err = sjson.Set(sj, "usage", u)
		if err != nil {
			return nil, fmt.Errorf("marshal chunk usage failed: %w", err)
		}
	}

	return i.encodeForStream([]byte(sj)), nil
}

func (i *OpenAIStreamingChatInterception) marshalErr(err error) ([]byte, error) {
	data, err := json.Marshal(err)
	if err != nil {
		return nil, fmt.Errorf("marshal error failed: %w", err)
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

// newStream traces svc.NewStreaming(streamCtx, i.req.ChatCompletionNewParams) call
func (i *OpenAIStreamingChatInterception) newStream(ctx context.Context, svc openai.ChatCompletionService) *ssestream.Stream[openai.ChatCompletionChunk] {
	_, span := i.tracer.Start(ctx, "Intercept.ProcessRequest.Upstream", trace.WithAttributes(tracing.InterceptionAttributesFromContext(ctx)...))
	defer span.End()

	return svc.NewStreaming(ctx, i.req.ChatCompletionNewParams)
}

type openAIStreamProcessor struct {
	ctx    context.Context
	logger slog.Logger

	acc openai.ChatCompletionAccumulator

	// Tool handling.
	pendingToolCall     bool
	getInjectedToolFunc func(string) *mcp.Tool

	// Token handling.
	lastUsage       openai.CompletionUsage
	cumulativeUsage openai.CompletionUsage
}

func newStreamProcessor(ctx context.Context, logger slog.Logger, isToolInjectedFunc func(string) *mcp.Tool) *openAIStreamProcessor {
	return &openAIStreamProcessor{
		ctx:    ctx,
		logger: logger,

		getInjectedToolFunc: isToolInjectedFunc,
	}
}

// process receives a completion chunk and returns a bool indicating whether it should be
// relayed to the client.
func (s *openAIStreamProcessor) process(chunk openai.ChatCompletionChunk) bool {
	if !s.acc.AddChunk(chunk) {
		s.logger.Debug(s.ctx, "failed to accumulate chunk", slog.F("chunk", chunk.RawJSON()))
		// Potentially not fatal, move along in best effort...
	}

	// Accumulate token usage.
	s.lastUsage = chunk.Usage
	s.cumulativeUsage = sumUsage(s.cumulativeUsage, chunk.Usage)

	// If the stream has reached a terminal state (i.e. call a tool), and this tool is injected,
	// then it must not be relayed.
	if _, ok := s.acc.JustFinishedToolCall(); ok && s.pendingToolCall {
		return false
	}

	if len(chunk.Choices) == 0 {
		// Odd, should not occur, relay it on in case.
		// Nothing more to be done.
		return true
	}

	// We explicitly set n=1, so this shouldn't happen.
	if count := len(chunk.Choices); count > 1 {
		s.logger.Warn(s.ctx, "multiple choices returned, only handling first", slog.F("count", count))
	}

	// Check if we have a tool call in progress.
	//
	// The API will send partial tool call events like this:
	//
	// data: ... delta":{"tool_calls":[{"index":0,"id":"call_0TxntkwDB66KH8z4RwNqeWrZ","type":"function","function":{"name":"bmcp_coder_coder_list_workspaces","arguments":""}}]}...
	// data: ... delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\""}}]}...
	// data: ... delta":{"tool_calls":[{"index":0,"function":{"arguments":"owner"}}]}...
	// data: ... delta":{"tool_calls":[{"index":0,"function":{"arguments":"\":\""}}]}...
	// data: ... delta":{"tool_calls":[{"index":0,"function":{"arguments":"admin"}}]}...
	// data: ... delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"}"}}]}...
	//
	// So we need to ensure that we don't relay any of the partial events to the client in the case of
	// an injected tool.
	//
	// The first partial will tell us the tool name, and we can then decide how to proceed.

	choice := chunk.Choices[0]
	if len(choice.Delta.ToolCalls) == 0 {
		// No tool calls, no special handling required.
		return true
	}

	// If we have a pending injected tool call in progress, do not relay any subsequent partial chunks.
	if s.pendingToolCall {
		return false
	}

	// This shouldn't happen since we have parallel tool calls disabled currently.
	if count := len(choice.Delta.ToolCalls); count > 1 {
		s.logger.Warn(context.Background(), "unexpected tool call count", slog.F("count", count))
		// We'll continue and just examine the first tool.
	}

	toolCall := choice.Delta.ToolCalls[0]
	if s.isInjected(toolCall) {
		// Mark tool as pending until tool call is finished.
		s.pendingToolCall = true
		return false
	}

	// There is a tool call, but it's not injected.
	return true
}

// getMsgID returns the ID given by the API for this (accumulated) message.
func (s *openAIStreamProcessor) getMsgID() string {
	return s.acc.ID
}

func (s *openAIStreamProcessor) isInjected(toolCall openai.ChatCompletionChunkChoiceDeltaToolCall) bool {
	return s.getInjectedToolFunc(strings.TrimSpace(toolCall.Function.Name)) != nil
}

func (s *openAIStreamProcessor) getToolCall() *openai.FinishedChatCompletionToolCall {
	tc, ok := s.acc.JustFinishedToolCall()
	if !ok {
		return nil
	}

	return &tc
}

func (s *openAIStreamProcessor) getLastCompletion() *openai.ChatCompletionMessage {
	if len(s.acc.Choices) == 0 {
		return nil
	}

	return &s.acc.Choices[0].Message
}

func (s *openAIStreamProcessor) getLastUsage() openai.CompletionUsage {
	return s.lastUsage
}

func (s *openAIStreamProcessor) getCumulativeUsage() openai.CompletionUsage {
	return s.cumulativeUsage
}
