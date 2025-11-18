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
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/packages/ssestream"
	"github.com/openai/openai-go/v3/responses"

	"cdr.dev/slog"
)

var _ Interceptor = &OpenAIStreamingChatInterception{}

type OpenAIStreamingResponsesInterception struct {
	OpenAIResponsesInterceptionBase
}

func NewOpenAIStreamingResponsesInterception(id uuid.UUID, req *ResponsesNewParamsWrapper, baseURL, key string) *OpenAIStreamingResponsesInterception {
	return &OpenAIStreamingResponsesInterception{OpenAIResponsesInterceptionBase: OpenAIResponsesInterceptionBase{
		id:      id,
		req:     req,
		baseURL: baseURL,
		key:     key,
	}}
}

func (i *OpenAIStreamingResponsesInterception) Setup(logger slog.Logger, recorder Recorder, mcpProxy mcp.ServerProxier) {
	i.OpenAIResponsesInterceptionBase.Setup(logger.Named("streaming"), recorder, mcpProxy)
}

// ProcessRequest handles a streaming request to /v1/responses.
//
// It will inject any tools which have been provided by the [mcp.ServerProxier].
//
// When a response from the server includes an event indicating that a tool must
// be invoked, a conditional flow takes place:
//
// a) if the tool is not injected (i.e. defined by the client), relay the event unmodified
// b) if the tool is injected, it will be invoked by the [mcp.ServerProxier] in the remote MCP server, and its
// results relayed to the SERVER. The response from the server will be handled synchronously, and this loop
// can continue until all injected tool invocations are completed and the response is relayed to the client.
func (i *OpenAIStreamingResponsesInterception) ProcessRequest(w http.ResponseWriter, r *http.Request) error {
	if i.req == nil {
		return fmt.Errorf("developer error: req is nil")
	}
	if i.req.Background.Value {
		return fmt.Errorf("background requests are currently not supported by aibridge")
	}

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
	// TODO: safe to send if there are zero tools?
	i.req.ParallelToolCalls = openai.Bool(false)

	prompt, err := i.req.LastUserPrompt()
	if err != nil {
		logger.Warn(ctx, "failed to retrieve last user prompt", slog.Error(err))
	}

	var (
		stream          *ssestream.Stream[responses.ResponseStreamEventUnion]
		lastErr         error
		interceptionErr error
	)
	for {
		stream = client.Responses.NewStreaming(streamCtx, i.req.ResponseNewParams)
		processor := newOpenAIResponsesStreamProcessor(streamCtx, i.logger.Named("stream-processor"), i.getInjectedToolByName)

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

			// If usage information is available, relay the cumulative usage once all tool invocations have completed.
			if chunk.Response.Usage.OutputTokens > 0 {
				chunk.Response.Usage = processor.getCumulativeUsage()
			}

			// Overwrite response identifier since proxy obscures injected tool call invocations.
			chunk.Response.ID = i.ID().String()

			// Marshal and relay chunk to client.
			payload, err := i.marshal(chunk)
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
				Input:          calculateOpenAICompletionsActualInputTokenUsage(lastUsage),
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
			payload, err := i.marshal(interceptionErr)
			if err != nil {
				logger.Warn(ctx, "failed to marshal error", slog.Error(err), slog.F("error_payload", slog.F("%+v", interceptionErr)))
			} else if err := events.Send(streamCtx, payload); err != nil {
				logger.Warn(ctx, "failed to relay error", slog.Error(err), slog.F("payload", payload))
			}
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
		toolRes, toolErr := tool.Call(streamCtx, args)

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

func (i *OpenAIStreamingResponsesInterception) getInjectedToolByName(name string) *mcp.Tool {
	if i.mcpProxy == nil {
		return nil
	}

	return i.mcpProxy.GetTool(name)
}

func (i *OpenAIStreamingResponsesInterception) marshal(payload any) ([]byte, error) {
	data, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("marshal payload: %w", err)
	}

	return i.encodeForStream(data), nil
}

func (i *OpenAIStreamingResponsesInterception) encodeForStream(payload []byte) []byte {
	var buf bytes.Buffer
	buf.WriteString("data: ")
	buf.Write(payload)
	buf.WriteString("\n\n")
	return buf.Bytes()
}

type openAIResponsesStreamProcessor struct {
	ctx    context.Context
	logger slog.Logger

	allEvents []responses.ResponseStreamEventUnion
	// sentCreated tracks whether we've sent a response.created event to the
	// client.
	sentCreated bool

	// pendingInjectedToolCallOutput stores a pending completed tool call output
	// incompletedInjectedToolCallOutputIndices tracks the indices of output
	// items that are still being streamed but relate to pending tool calls.
	incompletedInjectedToolCallOutputIndices map[int]struct{}

	// Tool handling.
	pendingToolCall     bool
	getInjectedToolFunc func(string) *mcp.Tool

	// Token handling.
	lastUsage       openai.CompletionUsage
	cumulativeUsage openai.CompletionUsage
}

func newOpenAIResponsesStreamProcessor(ctx context.Context, logger slog.Logger, isToolInjectedFunc func(string) *mcp.Tool) *openAIResponsesStreamProcessor {
	return &openAIResponsesStreamProcessor{
		ctx:    ctx,
		logger: logger,

		getInjectedToolFunc: isToolInjectedFunc,
	}
}

// process receives a completion chunk and returns a potentially modified
// response object and a bool representing whether the chunk should be relayed
// to the client.
func (s *openAIResponsesStreamProcessor) process(chunk responses.ResponseStreamEventUnion) (any, bool) {
	s.allEvents = append(s.allEvents, chunk)

	switch chunk.Type() {
	case "response.created":
		v := chunk.AsResponseCreated()
		if len(v.Response.Output) > 0 {
			// TODO: do we need to handle tool calls that could be in here??
			// for now we just ignore them
			logger.Warn(s.ctx, "response created with output items", slog.F("output_items", len(v.Response.Output)))
		}

		if !s.sentCreated {
			s.sentCreated = true
			return v, true
		}



	case "response.output_item.added":
		outputItemAdded := chunk.AsResponseOutputItemAdded()
		if outputItemAdded != nil {
			toolCall := outputItemAdded.Item.AsCustomToolCall()
			if toolCall != nil {
				s.incompletedInjectedToolCallOutputIndices[outputItemAdded.Index] = struct{}{}
			}
		}
	case
	}

	/*

	// Accumulate token usage.
	s.lastUsage = chunk.Response.Usage
	s.cumulativeUsage = sumOpenAICompletionsUsage(s.cumulativeUsage, chunk.Usage)

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

	*/
}

// getMsgID returns the ID given by the API for this (accumulated) message.
func (s *openAIResponsesStreamProcessor) getMsgID() string {
	return s.acc.ID
}

func (s *openAIResponsesStreamProcessor) isInjected(toolCall responses.ResponseCustomToolCall) bool {
	return s.getInjectedToolFunc(strings.TrimSpace(toolCall.Name)) != nil
}

func (s *openAIResponsesStreamProcessor) getToolCall() *openai.FinishedChatCompletionToolCall {
	tc, ok := s.acc.JustFinishedToolCall()
	if !ok {
		return nil
	}

	return &tc
}

func (s *openAIResponsesStreamProcessor) getLastCompletion() *openai.ChatCompletionMessage {
	if len(s.acc.Choices) == 0 {
		return nil
	}

	return &s.acc.Choices[0].Message
}

func (s *openAIResponsesStreamProcessor) getLastUsage() openai.CompletionUsage {
	return s.lastUsage
}

func (s *openAIResponsesStreamProcessor) getCumulativeUsage() openai.CompletionUsage {
	return s.cumulativeUsage
}
