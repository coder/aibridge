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

func NewBlockingInterceptor(
	id uuid.UUID,
	reqPayload MessagesRequestPayload,
	cfg config.Anthropic,
	bedrockCfg *config.AWSBedrock,
	clientHeaders http.Header,
	authHeaderName string,
	tracer trace.Tracer,
) *BlockingInterception {
	return &BlockingInterception{interceptionBase: interceptionBase{
		id:             id,
		reqPayload:     reqPayload,
		cfg:            cfg,
		bedrockCfg:     bedrockCfg,
		clientHeaders:  clientHeaders,
		authHeaderName: authHeaderName,
		tracer:         tracer,
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
	if len(i.reqPayload) == 0 {
		return fmt.Errorf("developer error: request payload is empty")
	}

	ctx, span := i.tracer.Start(r.Context(), "Intercept.ProcessRequest", trace.WithAttributes(tracing.InterceptionAttributesFromContext(r.Context())...))
	defer tracing.EndSpanErr(span, &outErr)

	i.injectTools()

	var prompt *string
	if !i.isSmallFastModel() {
		promptText, promptFound, promptErr := i.reqPayload.lastUserPrompt()
		if promptErr != nil {
			i.logger.Warn(ctx, "failed to retrieve last user prompt", slog.Error(promptErr))
		} else if promptFound {
			prompt = &promptText
		}
	}

	// TODO(ssncferreira): inject actor headers directly in the client-header
	//   middleware instead of using SDK options.
	requestOptions := []option.RequestOption{option.WithRequestTimeout(time.Second * 600)}
	if actor := aibcontext.ActorFromContext(r.Context()); actor != nil && i.cfg.SendActorHeaders {
		requestOptions = append(requestOptions, intercept.ActorHeadersAsAnthropicOpts(actor)...)
	}

	svc, err := i.newMessagesService(ctx, requestOptions...)
	if err != nil {
		err = fmt.Errorf("create anthropic client: %w", err)
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return err
	}

	logger := i.logger.With(slog.F("model", i.Model()))

	var resp *anthropic.Message
	var cumulativeUsage anthropic.Usage

	for {
		resp, err = i.newMessage(ctx, svc)
		if err != nil {
			if eventstream.IsConnError(err) {
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

		// Capture any thinking blocks that were returned.
		for _, t := range i.extractModelThoughts(resp) {
			_ = i.recorder.RecordModelThought(ctx, &recorder.ModelThoughtRecord{
				InterceptionID: i.ID().String(),
				Content:        t.Content,
				Metadata:       t.Metadata,
			})
		}

		// Handle tool calls.
		var pendingToolCalls []anthropic.ToolUseBlock
		for _, contentBlock := range resp.Content {
			toolUse := contentBlock.AsToolUse()
			if toolUse.ID == "" {
				continue
			}

			if i.mcpProxy != nil && i.mcpProxy.GetTool(toolUse.Name) != nil {
				pendingToolCalls = append(pendingToolCalls, toolUse)
				continue
			}

			_ = i.recorder.RecordToolUsage(ctx, &recorder.ToolUsageRecord{
				InterceptionID: i.ID().String(),
				MsgID:          resp.ID,
				ToolCallID:     toolUse.ID,
				Tool:           toolUse.Name,
				Args:           toolUse.Input,
				Injected:       false,
			})
		}

		if len(pendingToolCalls) == 0 {
			break
		}

		var loopMessages []anthropic.MessageParam
		loopMessages = append(loopMessages, resp.ToParam())

		for _, toolCall := range pendingToolCalls {
			if i.mcpProxy == nil {
				continue
			}

			tool := i.mcpProxy.GetTool(toolCall.Name)
			if tool == nil {
				logger.Warn(ctx, "tool not found in manager", slog.F("tool", toolCall.Name))
				loopMessages = append(loopMessages,
					anthropic.NewUserMessage(anthropic.NewToolResultBlock(toolCall.ID, fmt.Sprintf("Error: tool %s not found", toolCall.Name), true)),
				)
				continue
			}

			toolResultResponse, toolCallErr := tool.Call(ctx, toolCall.Input, i.tracer)
			_ = i.recorder.RecordToolUsage(ctx, &recorder.ToolUsageRecord{
				InterceptionID:  i.ID().String(),
				MsgID:           resp.ID,
				ToolCallID:      toolCall.ID,
				ServerURL:       &tool.ServerURL,
				Tool:            tool.Name,
				Args:            toolCall.Input,
				Injected:        true,
				InvocationError: toolCallErr,
			})

			if toolCallErr != nil {
				loopMessages = append(loopMessages,
					anthropic.NewUserMessage(anthropic.NewToolResultBlock(toolCall.ID, fmt.Sprintf("Error: calling tool: %v", toolCallErr), true)),
				)
				continue
			}

			toolResult := anthropic.ContentBlockParamUnion{
				OfToolResult: &anthropic.ToolResultBlockParam{
					ToolUseID: toolCall.ID,
					IsError:   anthropic.Bool(false),
				},
			}

			var hasValidResult bool
			for _, toolContent := range toolResultResponse.Content {
				switch contentBlock := toolContent.(type) {
				case mcplib.TextContent:
					toolResult.OfToolResult.Content = append(toolResult.OfToolResult.Content, anthropic.ToolResultBlockParamContentUnion{
						OfText: &anthropic.TextBlockParam{
							Text: contentBlock.Text,
						},
					})
					hasValidResult = true
				case mcplib.EmbeddedResource:
					switch resource := contentBlock.Resource.(type) {
					case mcplib.TextResourceContents:
						value := fmt.Sprintf("Binary resource (MIME: %s, URI: %s): %s", resource.MIMEType, resource.URI, resource.Text)
						toolResult.OfToolResult.Content = append(toolResult.OfToolResult.Content, anthropic.ToolResultBlockParamContentUnion{
							OfText: &anthropic.TextBlockParam{
								Text: value,
							},
						})
						hasValidResult = true
					case mcplib.BlobResourceContents:
						value := fmt.Sprintf("Binary resource (MIME: %s, URI: %s): %s", resource.MIMEType, resource.URI, resource.Blob)
						toolResult.OfToolResult.Content = append(toolResult.OfToolResult.Content, anthropic.ToolResultBlockParamContentUnion{
							OfText: &anthropic.TextBlockParam{
								Text: value,
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
					i.logger.Warn(ctx, "not handling non-text tool result", slog.F("type", fmt.Sprintf("%T", contentBlock)))
					toolResult.OfToolResult.Content = append(toolResult.OfToolResult.Content, anthropic.ToolResultBlockParamContentUnion{
						OfText: &anthropic.TextBlockParam{
							Text: "Error: unsupported tool result type",
						},
					})
					toolResult.OfToolResult.IsError = anthropic.Bool(true)
					hasValidResult = true
				}
			}

			if !hasValidResult {
				i.logger.Warn(ctx, "no tool result added", slog.F("content_len", len(toolResultResponse.Content)), slog.F("is_error", toolResultResponse.IsError))
				toolResult.OfToolResult.Content = append(toolResult.OfToolResult.Content, anthropic.ToolResultBlockParamContentUnion{
					OfText: &anthropic.TextBlockParam{
						Text: "Error: no valid tool result content",
					},
				})
				toolResult.OfToolResult.IsError = anthropic.Bool(true)
			}

			if len(toolResult.OfToolResult.Content) > 0 {
				loopMessages = append(loopMessages, anthropic.NewUserMessage(toolResult))
			}
		}

		updatedPayload, rewriteErr := i.reqPayload.appendedMessages(loopMessages)
		if rewriteErr != nil {
			http.Error(w, rewriteErr.Error(), http.StatusInternalServerError)
			return fmt.Errorf("rewrite payload for agentic loop: %w", rewriteErr)
		}
		i.reqPayload = updatedPayload
	}

	if resp == nil {
		return nil
	}

	responseJSON, err := sjson.Set(resp.RawJSON(), "id", i.ID().String())
	if err != nil {
		return fmt.Errorf("marshal response id failed: %w", err)
	}

	responseJSON, err = sjson.Set(responseJSON, "usage", cumulativeUsage)
	if err != nil {
		return fmt.Errorf("marshal response usage failed: %w", err)
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write([]byte(responseJSON))

	return nil
}

func (i *BlockingInterception) newMessage(ctx context.Context, svc anthropic.MessageService) (_ *anthropic.Message, outErr error) {
	ctx, span := i.tracer.Start(ctx, "Intercept.ProcessRequest.Upstream", trace.WithAttributes(tracing.InterceptionAttributesFromContext(ctx)...))
	defer tracing.EndSpanErr(span, &outErr)

	return svc.New(ctx, anthropic.MessageNewParams{}, i.withBody())
}
