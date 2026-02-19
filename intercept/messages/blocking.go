package messages

import (
	"context"
	"fmt"
	"net/http"
	"time"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/coder/aibridge/config"
	aibcontext "github.com/coder/aibridge/context"
	"github.com/coder/aibridge/intercept"
	"github.com/coder/aibridge/intercept/eventstream"
	"github.com/coder/aibridge/mcp"
	"github.com/coder/aibridge/recorder"
	"github.com/coder/aibridge/tracing"
	"github.com/google/uuid"
	mcplib "github.com/mark3labs/mcp-go/mcp"
	"github.com/tidwall/sjson"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"

	"cdr.dev/slog/v3"
)

type BlockingInterception struct {
	interceptionBase
}

func NewBlockingInterceptor(id uuid.UUID, req *MessageNewParamsWrapper, cfg config.Anthropic, bedrockCfg *config.AWSBedrock, tracer trace.Tracer) *BlockingInterception {
	return &BlockingInterception{interceptionBase: interceptionBase{
		id:         id,
		req:        req,
		cfg:        cfg,
		bedrockCfg: bedrockCfg,
		tracer:     tracer,
	}}
}

func (i *BlockingInterception) Setup(logger slog.Logger, recorder recorder.Recorder, mcpProxy mcp.ServerProxier) {
	i.interceptionBase.Setup(logger.Named("blocking"), recorder, mcpProxy)
}

func (i *BlockingInterception) TraceAttributes(r *http.Request) []attribute.KeyValue {
	return i.interceptionBase.baseTraceAttributes(r, false)
}

func (s *BlockingInterception) Streaming() bool {
	return false
}

func (i *BlockingInterception) ProcessRequest(w http.ResponseWriter, r *http.Request) (outErr error) {
	if i.req == nil {
		return fmt.Errorf("developer error: req is nil")
	}

	ctx, span := i.tracer.Start(r.Context(), "Intercept.ProcessRequest", trace.WithAttributes(tracing.InterceptionAttributesFromContext(r.Context())...))
	defer tracing.EndSpanErr(span, &outErr)

	i.injectTools()

	var (
		prompt *string
		err    error
	)
	// Track user prompt if not a small/fast model
	if !i.isSmallFastModel() {
		prompt, err = i.req.lastUserPrompt()
		if err != nil {
			i.logger.Warn(ctx, "failed to retrieve last user prompt", slog.Error(err))
		}
	}

	opts := []option.RequestOption{option.WithRequestTimeout(time.Second * 600)}
	if actor := aibcontext.ActorFromContext(r.Context()); actor != nil && i.cfg.SendActorHeaders {
		opts = append(opts, intercept.ActorHeadersAsAnthropicOpts(actor)...)
	}

	svc, err := i.newMessagesService(ctx, opts...)
	if err != nil {
		err = fmt.Errorf("create anthropic client: %w", err)
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return err
	}

	messages := i.req.MessageNewParams
	logger := i.logger.With(slog.F("model", i.req.Model))

	// Scan the request for tool results; we use these to correlate requests together.
	for _, block := range messages.Messages[len(messages.Messages)-1].Content {
		if block.OfToolResult == nil {
			continue
		}
		i.correlatingToolCallID = block.OfToolResult.ToolUseID
		break
	}

	var resp *anthropic.Message
	// Accumulate usage across the entire streaming interaction (including tool reinvocations).
	var cumulativeUsage anthropic.Usage

	for {
		// TODO add outer loop span (https://github.com/coder/aibridge/issues/67)
		resp, err = i.newMessage(ctx, svc, messages)
		if err != nil {
			if eventstream.IsConnError(err) {
				// Can't write a response, just error out.
				return fmt.Errorf("upstream connection closed: %w", err)
			}

			if antErr := getErrorResponse(err); antErr != nil {
				i.writeUpstreamError(w, antErr)
				return fmt.Errorf("anthropic API error: %w", err)
			}

			http.Error(w, "internal error", http.StatusInternalServerError)
			return fmt.Errorf("internal error: %w", err)
		}

		if prompt != nil {
			_ = i.recorder.RecordPromptUsage(ctx, &recorder.PromptUsageRecord{
				InterceptionID: i.ID().String(),
				MsgID:          resp.ID,
				Prompt:         *prompt,
			})
			prompt = nil
		}

		_ = i.recorder.RecordTokenUsage(ctx, &recorder.TokenUsageRecord{
			InterceptionID: i.ID().String(),
			MsgID:          resp.ID,
			Input:          resp.Usage.InputTokens,
			Output:         resp.Usage.OutputTokens,
			ExtraTokenTypes: map[string]int64{
				"web_search_requests":      resp.Usage.ServerToolUse.WebSearchRequests,
				"cache_creation_input":     resp.Usage.CacheCreationInputTokens,
				"cache_read_input":         resp.Usage.CacheReadInputTokens,
				"cache_ephemeral_1h_input": resp.Usage.CacheCreation.Ephemeral1hInputTokens,
				"cache_ephemeral_5m_input": resp.Usage.CacheCreation.Ephemeral5mInputTokens,
			},
		})

		accumulateUsage(&cumulativeUsage, resp.Usage)

		// Handle tool calls for non-streaming.
		var pendingToolCalls []anthropic.ToolUseBlock
		for _, c := range resp.Content {
			toolUse := c.AsToolUse()
			if toolUse.ID == "" {
				continue
			}

			if i.mcpProxy != nil && i.mcpProxy.GetTool(toolUse.Name) != nil {
				pendingToolCalls = append(pendingToolCalls, toolUse)
				continue
			}

			// If tool is not injected, track it since the client will be handling it.
			_ = i.recorder.RecordToolUsage(ctx, &recorder.ToolUsageRecord{
				InterceptionID: i.ID().String(),
				MsgID:          resp.ID,
				ToolCallID:     toolUse.ID,
				Tool:           toolUse.Name,
				Args:           toolUse.Input,
				Injected:       false,
			})
		}

		// If no injected tool calls, we're done.
		if len(pendingToolCalls) == 0 {
			break
		}

		// Append the assistant's message (which contains the tool_use block)
		// to the messages for the next API call.
		messages.Messages = append(messages.Messages, resp.ToParam())

		// Process each pending tool call.
		for _, tc := range pendingToolCalls {
			if i.mcpProxy == nil {
				continue
			}

			tool := i.mcpProxy.GetTool(tc.Name)
			if tool == nil {
				logger.Warn(ctx, "tool not found in manager", slog.F("tool", tc.Name))
				// Continue to next tool call, but still append an error tool_result
				messages.Messages = append(messages.Messages,
					anthropic.NewUserMessage(anthropic.NewToolResultBlock(tc.ID, fmt.Sprintf("Error: tool %s not found", tc.Name), true)),
				)
				continue
			}

			res, err := tool.Call(ctx, tc.Input, i.tracer)

			_ = i.recorder.RecordToolUsage(ctx, &recorder.ToolUsageRecord{
				InterceptionID:  i.ID().String(),
				MsgID:           resp.ID,
				ToolCallID:      tc.ID,
				ServerURL:       &tool.ServerURL,
				Tool:            tool.Name,
				Args:            tc.Input,
				Injected:        true,
				InvocationError: err,
			})

			if err != nil {
				// Always provide a tool_result even if the tool call failed
				messages.Messages = append(messages.Messages,
					anthropic.NewUserMessage(anthropic.NewToolResultBlock(tc.ID, fmt.Sprintf("Error: calling tool: %v", err), true)),
				)
				continue
			}

			// Process tool result
			toolResult := anthropic.ContentBlockParamUnion{
				OfToolResult: &anthropic.ToolResultBlockParam{
					ToolUseID: tc.ID,
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
				// TODO: is there a more correct way of handling these non-text content responses?
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
						i.logger.Warn(ctx, "unknown embedded resource type", slog.F("type", fmt.Sprintf("%T", resource)))
						toolResult.OfToolResult.Content = append(toolResult.OfToolResult.Content, anthropic.ToolResultBlockParamContentUnion{
							OfText: &anthropic.TextBlockParam{
								Text: "Error: unknown embedded resource type",
							},
						})
						toolResult.OfToolResult.IsError = anthropic.Bool(true)
						hasValidResult = true
					}
				default:
					i.logger.Warn(ctx, "not handling non-text tool result", slog.F("type", fmt.Sprintf("%T", cb)))
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
				i.logger.Warn(ctx, "no tool result added", slog.F("content_len", len(res.Content)), slog.F("is_error", res.IsError))
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
	}

	if resp == nil {
		return nil
	}

	// Overwrite response identifier since proxy obscures injected tool call invocations.
	sj, err := sjson.Set(resp.RawJSON(), "id", i.ID().String())
	if err != nil {
		return fmt.Errorf("marshal response id failed: %w", err)
	}

	// Overwrite the response's usage with the cumulative usage across any inner loops which invokes injected MCP tools.
	sj, err = sjson.Set(sj, "usage", cumulativeUsage)
	if err != nil {
		return fmt.Errorf("marshal response usage failed: %w", err)
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write([]byte(sj))

	return nil
}

func (i *BlockingInterception) newMessage(ctx context.Context, svc anthropic.MessageService, msgParams anthropic.MessageNewParams) (_ *anthropic.Message, outErr error) {
	ctx, span := i.tracer.Start(ctx, "Intercept.ProcessRequest.Upstream", trace.WithAttributes(tracing.InterceptionAttributesFromContext(ctx)...))
	defer tracing.EndSpanErr(span, &outErr)

	return svc.New(ctx, msgParams)
}
