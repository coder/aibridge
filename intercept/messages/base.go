package messages

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/bedrock"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/anthropics/anthropic-sdk-go/shared"
	"github.com/anthropics/anthropic-sdk-go/shared/constant"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/credentials"
	aibconfig "github.com/coder/aibridge/config"
	aibcontext "github.com/coder/aibridge/context"
	"github.com/coder/aibridge/intercept/apidump"
	"github.com/coder/aibridge/mcp"
	"github.com/coder/aibridge/recorder"
	"github.com/coder/aibridge/tracing"
	"github.com/coder/quartz"

	"github.com/google/uuid"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"

	"cdr.dev/slog/v3"
)

type interceptionBase struct {
	id  uuid.UUID
	req *MessageNewParamsWrapper

	cfg        aibconfig.Anthropic
	bedrockCfg *aibconfig.AWSBedrock

	tracer trace.Tracer
	logger slog.Logger

	recorder recorder.Recorder
	mcpProxy mcp.ServerProxier
}

func (i *interceptionBase) ID() uuid.UUID {
	return i.id
}

func (i *interceptionBase) Setup(logger slog.Logger, recorder recorder.Recorder, mcpProxy mcp.ServerProxier) {
	i.logger = logger
	i.recorder = recorder
	i.mcpProxy = mcpProxy
}

func (i *interceptionBase) Model() string {
	if i.req == nil {
		return "coder-aibridge-unknown"
	}

	if i.bedrockCfg != nil {
		model := i.bedrockCfg.Model
		if i.isSmallFastModel() {
			model = i.bedrockCfg.SmallFastModel
		}
		return model
	}

	return string(i.req.Model)
}

func (s *interceptionBase) baseTraceAttributes(r *http.Request, streaming bool) []attribute.KeyValue {
	return []attribute.KeyValue{
		attribute.String(tracing.RequestPath, r.URL.Path),
		attribute.String(tracing.InterceptionID, s.id.String()),
		attribute.String(tracing.InitiatorID, aibcontext.ActorFromContext(r.Context()).ID),
		attribute.String(tracing.Provider, aibconfig.ProviderAnthropic),
		attribute.String(tracing.Model, s.Model()),
		attribute.Bool(tracing.Streaming, streaming),
		attribute.Bool(tracing.IsBedrock, s.bedrockCfg != nil),
	}
}

func (i *interceptionBase) injectTools() {
	if i.req == nil || i.mcpProxy == nil {
		return
	}

	tools := i.mcpProxy.ListTools()
	if len(tools) == 0 {
		// No injected tools: no need to influence parallel tool calling.
		return
	}

	// Inject tools.
	var injectedTools []anthropic.ToolUnionParam
	for _, tool := range tools {
		injectedTools = append(injectedTools, anthropic.ToolUnionParam{
			OfTool: &anthropic.ToolParam{
				InputSchema: anthropic.ToolInputSchemaParam{
					Properties: tool.Params,
					Required:   tool.Required,
				},
				Name:        tool.ID,
				Description: anthropic.String(tool.Description),
				Type:        anthropic.ToolTypeCustom,
			},
		})
	}

	// Prepend the injected tools in order to maintain any configured cache breakpoints.
	// The order of injected tools is expected to be stable, and therefore will not cause
	// any cache invalidation when prepended.
	i.req.Tools = append(injectedTools, i.req.Tools...)

	// Note: Parallel tool calls are disabled to avoid tool_use/tool_result block mismatches.
	// https://github.com/coder/aibridge/issues/2
	toolChoiceType := i.req.ToolChoice.GetType()
	var toolChoiceTypeStr string
	if toolChoiceType != nil {
		toolChoiceTypeStr = *toolChoiceType
	}

	switch toolChoiceTypeStr {
	// If no tool_choice was defined, assume auto.
	// See https://platform.claude.com/docs/en/agents-and-tools/tool-use/implement-tool-use#parallel-tool-use.
	case "", string(constant.ValueOf[constant.Auto]()):
		// We only set OfAuto if no tool_choice was provided (the default).
		// "auto" is the default when a zero value is provided, so we can safely disable parallel checks on it.
		if i.req.ToolChoice.OfAuto == nil {
			i.req.ToolChoice.OfAuto = &anthropic.ToolChoiceAutoParam{}
		}
		i.req.ToolChoice.OfAuto.DisableParallelToolUse = anthropic.Bool(true)
	case string(constant.ValueOf[constant.Any]()):
		i.req.ToolChoice.OfAny.DisableParallelToolUse = anthropic.Bool(true)
	case string(constant.ValueOf[constant.Tool]()):
		i.req.ToolChoice.OfTool.DisableParallelToolUse = anthropic.Bool(true)
	case string(constant.ValueOf[constant.None]()):
		// No-op; if tool_choice=none then tools are not used at all.
	}
}

// IsSmallFastModel checks if the model is a small/fast model (Haiku 3.5).
// These models are optimized for tasks like code autocomplete and other small, quick operations.
// See `ANTHROPIC_SMALL_FAST_MODEL`: https://docs.anthropic.com/en/docs/claude-code/settings#environment-variables
// https://docs.claude.com/en/docs/claude-code/costs#background-token-usage
func (i *interceptionBase) isSmallFastModel() bool {
	return strings.Contains(string(i.req.Model), "haiku")
}

func (i *interceptionBase) newMessagesService(ctx context.Context, opts ...option.RequestOption) (anthropic.MessageService, error) {
	opts = append(opts, option.WithAPIKey(i.cfg.Key))
	opts = append(opts, option.WithBaseURL(i.cfg.BaseURL))

	// Add API dump middleware if configured
	if mw := apidump.NewMiddleware(i.cfg.APIDumpDir, aibconfig.ProviderAnthropic, i.Model(), i.id, i.logger, quartz.NewReal()); mw != nil {
		opts = append(opts, option.WithMiddleware(mw))
	}

	if i.bedrockCfg != nil {
		ctx, cancel := context.WithTimeout(ctx, time.Second*30)
		defer cancel()
		bedrockOpt, err := i.withAWSBedrock(ctx, i.bedrockCfg)
		if err != nil {
			return anthropic.MessageService{}, err
		}
		opts = append(opts, bedrockOpt)
		i.augmentRequestForBedrock()

		// If a custom base URL is set, add it AFTER the bedrock config to override
		// the default endpoint constructed by the bedrock middleware.
		if i.bedrockCfg.BaseURL != "" {
			opts = append(opts, option.WithBaseURL(i.bedrockCfg.BaseURL))
		}
	}

	return anthropic.NewMessageService(opts...), nil
}

func (i *interceptionBase) withAWSBedrock(ctx context.Context, cfg *aibconfig.AWSBedrock) (option.RequestOption, error) {
	if cfg == nil {
		return nil, fmt.Errorf("nil config given")
	}
	if cfg.Region == "" && cfg.BaseURL == "" {
		return nil, fmt.Errorf("region required")
	}
	if cfg.AccessKey == "" {
		return nil, fmt.Errorf("access key required")
	}
	if cfg.AccessKeySecret == "" {
		return nil, fmt.Errorf("access key secret required")
	}
	if cfg.Model == "" {
		return nil, fmt.Errorf("model required")
	}
	if cfg.SmallFastModel == "" {
		return nil, fmt.Errorf("small fast model required")
	}

	opts := []func(*config.LoadOptions) error{
		config.WithRegion(cfg.Region),
		config.WithCredentialsProvider(
			credentials.NewStaticCredentialsProvider(
				cfg.AccessKey,
				cfg.AccessKeySecret,
				"",
			),
		),
	}

	awsCfg, err := config.LoadDefaultConfig(ctx, opts...)
	if err != nil {
		return nil, fmt.Errorf("failed to load AWS Bedrock config: %w", err)
	}

	return bedrock.WithConfig(awsCfg), nil
}

// augmentRequestForBedrock will change the model used for the request since AWS Bedrock doesn't support
// Anthropics' model names.
func (i *interceptionBase) augmentRequestForBedrock() {
	if i.bedrockCfg == nil {
		return
	}

	i.req.MessageNewParams.Model = anthropic.Model(i.Model())
}

// writeUpstreamError marshals and writes a given error.
func (i *interceptionBase) writeUpstreamError(w http.ResponseWriter, antErr *ErrorResponse) {
	if antErr == nil {
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(antErr.StatusCode)

	out, err := json.Marshal(antErr)
	if err != nil {
		i.logger.Warn(context.Background(), "failed to marshal upstream error", slog.Error(err), slog.F("error_payload", slog.F("%+v", antErr)))
		// Response has to match expected format.
		// See https://docs.claude.com/en/api/errors#error-shapes.
		_, _ = w.Write([]byte(fmt.Sprintf(`{
	"type":"error",
	"error": {
		"type": "error",
		"message":"error marshaling upstream error"
	},
	"request_id": "%s"
}`, i.ID().String())))
	} else {
		_, _ = w.Write(out)
	}
}

// accumulateUsage accumulates usage statistics from source into dest.
// It handles both [anthropic.Usage] and [anthropic.MessageDeltaUsage] types through [any].
// The function uses reflection to handle the differences between the types:
// - [anthropic.Usage] has CacheCreation field with ephemeral tokens
// - [anthropic.MessageDeltaUsage] doesn't have CacheCreation field
func accumulateUsage(dest, src any) {
	switch d := dest.(type) {
	case *anthropic.Usage:
		if d == nil {
			return
		}
		switch s := src.(type) {
		case anthropic.Usage:
			// Usage -> Usage
			d.CacheCreation.Ephemeral1hInputTokens += s.CacheCreation.Ephemeral1hInputTokens
			d.CacheCreation.Ephemeral5mInputTokens += s.CacheCreation.Ephemeral5mInputTokens
			d.CacheCreationInputTokens += s.CacheCreationInputTokens
			d.CacheReadInputTokens += s.CacheReadInputTokens
			d.InputTokens += s.InputTokens
			d.OutputTokens += s.OutputTokens
			d.ServerToolUse.WebSearchRequests += s.ServerToolUse.WebSearchRequests
		case anthropic.MessageDeltaUsage:
			// MessageDeltaUsage -> Usage
			d.CacheCreationInputTokens += s.CacheCreationInputTokens
			d.CacheReadInputTokens += s.CacheReadInputTokens
			d.InputTokens += s.InputTokens
			d.OutputTokens += s.OutputTokens
			d.ServerToolUse.WebSearchRequests += s.ServerToolUse.WebSearchRequests
		}
	case *anthropic.MessageDeltaUsage:
		if d == nil {
			return
		}
		switch s := src.(type) {
		case anthropic.Usage:
			// Usage -> MessageDeltaUsage (only common fields)
			d.CacheCreationInputTokens += s.CacheCreationInputTokens
			d.CacheReadInputTokens += s.CacheReadInputTokens
			d.InputTokens += s.InputTokens
			d.OutputTokens += s.OutputTokens
			d.ServerToolUse.WebSearchRequests += s.ServerToolUse.WebSearchRequests
		case anthropic.MessageDeltaUsage:
			// MessageDeltaUsage -> MessageDeltaUsage
			d.CacheCreationInputTokens += s.CacheCreationInputTokens
			d.CacheReadInputTokens += s.CacheReadInputTokens
			d.InputTokens += s.InputTokens
			d.OutputTokens += s.OutputTokens
			d.ServerToolUse.WebSearchRequests += s.ServerToolUse.WebSearchRequests
		}
	}
}

func getErrorResponse(err error) *ErrorResponse {
	var apierr *anthropic.Error
	if !errors.As(err, &apierr) {
		return nil
	}

	msg := apierr.Error()
	typ := string(constant.ValueOf[constant.APIError]())

	var detail *anthropic.APIErrorObject
	if field, ok := apierr.JSON.ExtraFields["error"]; ok {
		_ = json.Unmarshal([]byte(field.Raw()), &detail)
	}
	if detail != nil {
		msg = detail.Message
		typ = string(detail.Type)
	}

	return &ErrorResponse{
		ErrorResponse: &anthropic.ErrorResponse{
			Error: anthropic.ErrorObjectUnion{
				Message: msg,
				Type:    typ,
			},
			Type: constant.ValueOf[constant.Error](),
		},
		StatusCode: apierr.StatusCode,
	}
}

var _ error = &ErrorResponse{}

type ErrorResponse struct {
	*anthropic.ErrorResponse

	StatusCode int `json:"-"`
}

func newErrorResponse(msg error) *ErrorResponse {
	return &ErrorResponse{
		ErrorResponse: &shared.ErrorResponse{
			Error: shared.ErrorObjectUnion{
				Message: msg.Error(),
				Type:    "error",
			},
		},
	}
}

func (a *ErrorResponse) Error() string {
	if a.ErrorResponse == nil {
		return ""
	}
	return a.ErrorResponse.Error.Message
}
