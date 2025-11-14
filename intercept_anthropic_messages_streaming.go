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

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/shared/constant"
	"github.com/coder/aibridge/mcp"
	"github.com/google/uuid"
	mcplib "github.com/mark3labs/mcp-go/mcp"

	"cdr.dev/slog"
)

var _ Interceptor = &AnthropicMessagesStreamingInterception{}

type AnthropicMessagesStreamingInterception struct {
	AnthropicMessagesInterceptionBase
}

func NewAnthropicMessagesStreamingInterception(id uuid.UUID, req *MessageNewParamsWrapper, cfg AnthropicConfig, bedrockCfg *AWSBedrockConfig) *AnthropicMessagesStreamingInterception {
	return &AnthropicMessagesStreamingInterception{AnthropicMessagesInterceptionBase: AnthropicMessagesInterceptionBase{
		id:         id,
		req:        req,
		cfg:        cfg,
		bedrockCfg: bedrockCfg,
	}}
}

func (s *AnthropicMessagesStreamingInterception) Setup(logger slog.Logger, recorder Recorder, mcpProxy mcp.ServerProxier) {
	s.AnthropicMessagesInterceptionBase.Setup(logger.Named("streaming"), recorder, mcpProxy)
}

// ProcessRequest handles a request to /v1/messages.
// This API has a state-machine behind it, which is described in https://docs.claude.com/en/docs/build-with-claude/streaming#event-types.
//
// Each stream uses the following event flow:
// - `message_start`: contains a Message object with empty content.
// - A series of content blocks, each of which have a `content_block_start`, one or more `content_block_delta` events, and a `content_block_stop` event.
// - Each content block will have an index that corresponds to its index in the final Message content array.
// - One or more `message_delta` events, indicating top-level changes to the final Message object.
// - A final `message_stop` event.
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
func (i *AnthropicMessagesStreamingInterception) ProcessRequest(w http.ResponseWriter, r *http.Request) error {
	if i.req == nil {
		return fmt.Errorf("developer error: req is nil")
	}

	// Allow us to interrupt watch via cancel.
	ctx, cancel := context.WithCancel(r.Context())
	defer cancel()
	r = r.WithContext(ctx) // Rewire context for SSE cancellation.

	logger := i.logger.With(slog.F("model", i.req.Model))

	var (
		prompt *string
		err    error
	)

	// Claude Code uses a "small/fast model" for certain tasks.
	if !i.isSmallFastModel() {
		prompt, err = i.req.LastUserPrompt()
		if err != nil {
			logger.Warn(ctx, "failed to determine last user prompt", slog.Error(err))
		}

		// Only inject tools into "actual" request.
		i.injectTools()
	}

	streamCtx, streamCancel := context.WithCancelCause(ctx)
	defer streamCancel(errors.New("deferred"))

	client, err := i.newAnthropicClient(streamCtx)
	if err != nil {
		err = fmt.Errorf("create anthropic client: %w", err)
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return err
	}

	// events will either terminate when shutdown after interaction with upstream completes, or when streamCtx is done.
	events := newEventStream(streamCtx, logger.Named("sse-sender"), i.pingPayload())
	go events.start(w, r)
	defer func() {
		_ = events.Shutdown(streamCtx) // Catch-all in case it doesn't get shutdown after stream completes.
	}()

	messages := i.req.MessageNewParams

	// Accumulate usage across the entire streaming interaction (including tool reinvocations).
	var cumulativeUsage anthropic.Usage

	var lastErr error
	var interceptionErr error

	isFirst := true
newStream:
	for {
		if err := streamCtx.Err(); err != nil {
			lastErr = fmt.Errorf("stream exit: %w", err)
			break
		}

		stream := client.Messages.NewStreaming(streamCtx, messages)

		var message anthropic.Message
		var lastToolName string

		pendingToolCalls := make(map[string]string)

		for stream.Next() {
			event := stream.Current()
			if err := message.Accumulate(event); err != nil {
				logger.Warn(ctx, "failed to accumulate streaming events", slog.Error(err), slog.F("event", event), slog.F("msg", message.RawJSON()))
				lastErr = fmt.Errorf("accumulate event: %w", err)
				break
			}

			// Tool-related handling.
			switch event.Type {
			case string(constant.ValueOf[constant.ContentBlockStart]()):
				switch block := event.AsContentBlockStart().ContentBlock.AsAny().(type) {
				case anthropic.ToolUseBlock:
					lastToolName = block.Name

					if i.mcpProxy != nil && i.mcpProxy.GetTool(block.Name) != nil {
						pendingToolCalls[block.Name] = block.ID
						// Don't relay this event back, otherwise the client will try invoke the tool as well.
						continue
					}
				}
			case string(constant.ValueOf[constant.ContentBlockDelta]()):
				if len(pendingToolCalls) > 0 && i.mcpProxy != nil && i.mcpProxy.GetTool(lastToolName) != nil {
					// We're busy with a tool call, don't relay this event back.
					continue
				}
			case string(constant.ValueOf[constant.ContentBlockStop]()):
				// Reset the tool name
				isInjected := i.mcpProxy != nil && i.mcpProxy.GetTool(lastToolName) != nil
				lastToolName = ""

				if len(pendingToolCalls) > 0 && isInjected {
					// We're busy with a tool call, don't relay this event back.
					continue
				}
			case string(constant.ValueOf[constant.MessageStart]()):
				start := event.AsMessageStart()
				accumulateUsage(&cumulativeUsage, start.Message.Usage)

				_ = i.recorder.RecordTokenUsage(streamCtx, &TokenUsageRecord{
					InterceptionID: i.ID().String(),
					MsgID:          message.ID,
					Input:          start.Message.Usage.InputTokens,
					Output:         start.Message.Usage.OutputTokens,
					Metadata: Metadata{
						"web_search_requests":      start.Message.Usage.ServerToolUse.WebSearchRequests,
						"cache_creation_input":     start.Message.Usage.CacheCreationInputTokens,
						"cache_read_input":         start.Message.Usage.CacheReadInputTokens,
						"cache_ephemeral_1h_input": start.Message.Usage.CacheCreation.Ephemeral1hInputTokens,
						"cache_ephemeral_5m_input": start.Message.Usage.CacheCreation.Ephemeral5mInputTokens,
					},
				})

				if !isFirst {
					// Don't send message_start unless first message!
					// We're sending multiple messages back and forth with the API, but from the client's perspective
					// they're just expecting a single message.
					continue
				}
			case string(constant.ValueOf[constant.MessageDelta]()):
				delta := event.AsMessageDelta()
				accumulateUsage(&cumulativeUsage, delta.Usage)

				// Only output tokens should change in message_delta.
				_ = i.recorder.RecordTokenUsage(streamCtx, &TokenUsageRecord{
					InterceptionID: i.ID().String(),
					MsgID:          message.ID,
					Output:         delta.Usage.OutputTokens,
				})

				// Don't relay message_delta events which indicate injected tool use.
				if len(pendingToolCalls) > 0 && i.mcpProxy != nil && i.mcpProxy.GetTool(lastToolName) != nil {
					continue
				}

				// If currently calling a tool.
				if len(message.Content) > 0 && message.Content[len(message.Content)-1].Type == string(constant.ValueOf[constant.ToolUse]()) {
					toolName := message.Content[len(message.Content)-1].AsToolUse().Name
					if len(pendingToolCalls) > 0 && i.mcpProxy != nil && i.mcpProxy.GetTool(toolName) != nil {
						continue
					}
				}

				// We should be updating the event's usage to the calculated cumulative usage. However...
				// the SDK only accumulates output tokens on message_delta, since that's all that *should* change.
				//
				// Backstory: the API reports tokens during message_start AND message_delta. message_start reports the input
				// tokens and others, while the delta should only report changes to output tokens.
				// HOWEVER, when we invoke injected tools we're starting a whole new message (and subsequently receive
				// message_start and message_delta events), and the previous message_start has already been relayed, so in effect
				// we can't really modify anything other than output tokens here according to the SDK.
				// This will affect how the client reports token usage for input tokens, for example.
				// For our purposes, the server (aibridge) is authoritative anyway so it's not a big deal, but this is something to note.
				//
				// See https://github.com/anthropics/anthropic-sdk-go/blob/v1.12.0/message.go#L2619-L2622
				event.Usage.OutputTokens = cumulativeUsage.OutputTokens

			// Don't send message_stop until all tools have been called.
			case string(constant.ValueOf[constant.MessageStop]()):
				if len(pendingToolCalls) > 0 {
					// Append the whole message from this stream as context since we'll be sending a new request with the tool results.
					messages.Messages = append(messages.Messages, message.ToParam())

					for name, id := range pendingToolCalls {
						if i.mcpProxy == nil {
							continue
						}

						if i.mcpProxy.GetTool(name) == nil {
							// Not an MCP proxy call, don't do anything.
							continue
						}

						tool := i.mcpProxy.GetTool(name)
						if tool == nil {
							logger.Warn(ctx, "tool not found in manager", slog.F("tool_name", name))
							continue
						}

						var (
							input      json.RawMessage
							foundTool  bool
							foundTools int
						)
						for _, block := range message.Content {
							switch variant := block.AsAny().(type) {
							case anthropic.ToolUseBlock:
								foundTools++
								if variant.Name == name {
									input = variant.Input
									foundTool = true
								}
							}
						}

						if !foundTool {
							logger.Warn(ctx, "failed to find tool input", slog.F("tool_name", name), slog.F("found_tools", foundTools))
							continue
						}

						res, err := tool.Call(streamCtx, input)

						_ = i.recorder.RecordToolUsage(streamCtx, &ToolUsageRecord{
							InterceptionID:  i.ID().String(),
							MsgID:           message.ID,
							ServerURL:       &tool.ServerURL,
							Tool:            tool.Name,
							Args:            input,
							Injected:        true,
							InvocationError: err,
						})

						if err != nil {
							// Always provide a tool_result even if the tool call failed
							messages.Messages = append(messages.Messages,
								anthropic.NewUserMessage(anthropic.NewToolResultBlock(id, fmt.Sprintf("Error calling tool: %v", err), true)),
							)
							continue
						}

						// Process tool result
						toolResult := anthropic.ContentBlockParamUnion{
							OfToolResult: &anthropic.ToolResultBlockParam{
								ToolUseID: id,
								IsError:   anthropic.Bool(false),
							},
						}

						var hasValidResult bool
						for _, content := range res.Content {
							switch cb := content.(type) {
							case mcplib.TextContent:
								toolResult.OfToolResult.Content = append(toolResult.OfToolResult.Content, anthropic.ToolResultBlockParamContentUnion{
									OfText: &anthropic.TextBlockParam{
										Text: cb.Text,
									},
								})
								hasValidResult = true
							case mcplib.EmbeddedResource:
								switch resource := cb.Resource.(type) {
								case mcplib.TextResourceContents:
									val := fmt.Sprintf("Binary resource (MIME: %s, URI: %s): %s",
										resource.MIMEType, resource.URI, resource.Text)
									toolResult.OfToolResult.Content = append(toolResult.OfToolResult.Content, anthropic.ToolResultBlockParamContentUnion{
										OfText: &anthropic.TextBlockParam{
											Text: val,
										},
									})
									hasValidResult = true
								case mcplib.BlobResourceContents:
									val := fmt.Sprintf("Binary resource (MIME: %s, URI: %s): %s",
										resource.MIMEType, resource.URI, resource.Blob)
									toolResult.OfToolResult.Content = append(toolResult.OfToolResult.Content, anthropic.ToolResultBlockParamContentUnion{
										OfText: &anthropic.TextBlockParam{
											Text: val,
										},
									})
									hasValidResult = true
								default:
									logger.Warn(ctx, "unknown embedded resource type", slog.F("type", fmt.Sprintf("%T", resource)))
									toolResult.OfToolResult.Content = append(toolResult.OfToolResult.Content, anthropic.ToolResultBlockParamContentUnion{
										OfText: &anthropic.TextBlockParam{
											Text: "Error: unknown embedded resource type",
										},
									})
									toolResult.OfToolResult.IsError = anthropic.Bool(true)
									hasValidResult = true
								}
							default:
								logger.Warn(ctx, "not handling non-text tool result", slog.F("type", fmt.Sprintf("%T", cb)))
								toolResult.OfToolResult.Content = append(toolResult.OfToolResult.Content, anthropic.ToolResultBlockParamContentUnion{
									OfText: &anthropic.TextBlockParam{
										Text: "Error: unsupported tool result type",
									},
								})
								toolResult.OfToolResult.IsError = anthropic.Bool(true)
								hasValidResult = true
							}
						}

						// If no content was processed, still add a tool_result
						if !hasValidResult {
							logger.Warn(ctx, "no tool result added", slog.F("content_len", len(res.Content)), slog.F("is_error", res.IsError))
							toolResult.OfToolResult.Content = append(toolResult.OfToolResult.Content, anthropic.ToolResultBlockParamContentUnion{
								OfText: &anthropic.TextBlockParam{
									Text: "Error: no valid tool result content",
								},
							})
							toolResult.OfToolResult.IsError = anthropic.Bool(true)
						}

						if len(toolResult.OfToolResult.Content) > 0 {
							messages.Messages = append(messages.Messages, anthropic.NewUserMessage(toolResult))
						}
					}

					// Causes a new stream to be run with updated messages.
					isFirst = false
					continue newStream
				} else {
					// Find all the non-injected tools and track their uses.
					for _, block := range message.Content {
						switch variant := block.AsAny().(type) {
						case anthropic.ToolUseBlock:
							if i.mcpProxy != nil && i.mcpProxy.GetTool(variant.Name) != nil {
								continue
							}

							_ = i.recorder.RecordToolUsage(streamCtx, &ToolUsageRecord{
								InterceptionID: i.ID().String(),
								MsgID:          message.ID,
								Tool:           variant.Name,
								Args:           variant.Input,
								Injected:       false,
							})
						}
					}
				}
			}

			// Overwrite response identifier since proxy obscures injected tool call invocations.
			event.Message.ID = i.ID().String()
			payload, err := i.marshal(event)
			if err != nil {
				logger.Warn(ctx, "failed to marshal event", slog.Error(err), slog.F("event", event.RawJSON()))
				lastErr = fmt.Errorf("marshal event: %w", err)
				break
			}
			if err := events.Send(streamCtx, payload); err != nil {
				if isUnrecoverableError(err) {
					logger.Debug(ctx, "processing terminated", slog.Error(err))
					break // Stop processing if client disconnected or context canceled.
				} else {
					logger.Warn(ctx, "failed to relay event", slog.Error(err))
					lastErr = fmt.Errorf("relay event: %w", err)
					break
				}
			}
		}

		if prompt != nil {
			_ = i.recorder.RecordPromptUsage(ctx, &PromptUsageRecord{
				InterceptionID: i.ID().String(),
				MsgID:          message.ID,
				Prompt:         *prompt,
			})
			prompt = nil
		}

		if events.isStreaming() {
			// Check if the stream encountered any errors.
			if streamErr := stream.Err(); streamErr != nil {
				if isUnrecoverableError(streamErr) {
					logger.Debug(ctx, "stream terminated", slog.Error(streamErr))
					// We can't reflect an error back if there's a connection error or the request context was canceled.
				} else if antErr := getAnthropicErrorResponse(streamErr); antErr != nil {
					logger.Warn(ctx, "anthropic stream error", slog.Error(streamErr))
					interceptionErr = antErr
				} else {
					logger.Warn(ctx, "unknown error", slog.Error(streamErr))
					// Unfortunately, the Anthropic SDK does not support parsing errors received in the stream
					// into known types (i.e. [shared.OverloadedError]).
					// See https://github.com/anthropics/anthropic-sdk-go/blob/v1.12.0/packages/ssestream/ssestream.go#L172-L174
					// All it does is wrap the payload in an error - which is all we can return, currently.
					interceptionErr = newAnthropicErr(fmt.Errorf("unknown stream error: %w", streamErr))
				}
			} else if lastErr != nil {
				// Otherwise check if any logical errors occurred during processing.
				logger.Warn(ctx, "stream failed", slog.Error(lastErr))
				interceptionErr = newAnthropicErr(fmt.Errorf("processing error: %w", lastErr))
			}

			if interceptionErr != nil {
				payload, err := i.marshal(interceptionErr)
				if err != nil {
					logger.Warn(ctx, "failed to marshal error", slog.Error(err), slog.F("error_payload", slog.F("%+v", interceptionErr)))
				} else if err := events.Send(streamCtx, payload); err != nil {
					logger.Warn(ctx, "failed to relay error", slog.Error(err), slog.F("payload", payload))
				}
			}
		} else {
			// Stream has not started yet; write to response if present.
			i.writeUpstreamError(w, getAnthropicErrorResponse(stream.Err()))
		}

		shutdownCtx, shutdownCancel := context.WithTimeout(ctx, time.Second*30)
		defer shutdownCancel()
		// Give the events stream 30 seconds (TODO: configurable) to gracefully shutdown.
		if err := events.Shutdown(shutdownCtx); err != nil {
			logger.Warn(ctx, "event stream shutdown", slog.Error(err))
		}

		// Cancel the stream context, we're now done.
		if interceptionErr != nil {
			streamCancel(interceptionErr)
		} else {
			streamCancel(errors.New("gracefully done"))
		}

		break
	}

	return interceptionErr
}

func (s *AnthropicMessagesStreamingInterception) marshal(payload any) ([]byte, error) {
	data, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("marshal payload: %w", err)
	}

	var parsed map[string]any
	if err := json.Unmarshal(data, &parsed); err != nil {
		return nil, fmt.Errorf("unmarshal payload: %w", err)
	}

	eventType, ok := parsed["type"].(string)
	if !ok || strings.TrimSpace(eventType) == "" {
		return nil, fmt.Errorf("could not determine type from payload %q", data)
	}

	return s.encodeForStream(data, eventType), nil
}

// https://docs.anthropic.com/en/docs/build-with-claude/streaming#basic-streaming-request
func (s *AnthropicMessagesStreamingInterception) pingPayload() []byte {
	return s.encodeForStream([]byte(`{"type": "ping"}`), "ping")
}

func (s *AnthropicMessagesStreamingInterception) encodeForStream(payload []byte, typ string) []byte {
	var buf bytes.Buffer
	buf.WriteString("event: ")
	buf.WriteString(typ)
	buf.WriteString("\n")
	buf.WriteString("data: ")
	buf.Write(payload)
	buf.WriteString("\n\n")
	return buf.Bytes()
}
