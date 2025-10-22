package aibridge

import (
	"strings"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/coder/aibridge/mcp"
	"github.com/google/uuid"

	"cdr.dev/slog"
)

type AnthropicMessagesInterceptionBase struct {
	id  uuid.UUID
	req *MessageNewParamsWrapper

	cfg    *ProviderConfig
	logger slog.Logger

	recorder Recorder
	mcpProxy mcp.ServerProxier
}

func (i *AnthropicMessagesInterceptionBase) ID() uuid.UUID {
	return i.id
}

func (i *AnthropicMessagesInterceptionBase) Setup(logger slog.Logger, recorder Recorder, mcpProxy mcp.ServerProxier) {
	i.logger = logger
	i.recorder = recorder
	i.mcpProxy = mcpProxy
}

func (i *AnthropicMessagesInterceptionBase) Model() string {
	if i.req == nil {
		return "coder-aibridge-unknown"
	}

	return string(i.req.Model)
}

func (i *AnthropicMessagesInterceptionBase) injectTools() {
	if i.req == nil || i.mcpProxy == nil {
		return
	}

	// Inject tools.
	for _, tool := range i.mcpProxy.ListTools() {
		i.req.Tools = append(i.req.Tools, anthropic.ToolUnionParam{
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

	// Note: Parallel tool calls are disabled to avoid tool_use/tool_result block mismatches.
	i.req.ToolChoice = anthropic.ToolChoiceUnionParam{
		OfAny: &anthropic.ToolChoiceAnyParam{
			Type:                   "auto",
			DisableParallelToolUse: anthropic.Bool(true),
		},
	}
}

// isSmallFastModel checks if the model is a small/fast model (Haiku 3.5).
// These models are optimized for tasks like code autocomplete and other small, quick operations.
// See `ANTHROPIC_SMALL_FAST_MODEL`: https://docs.anthropic.com/en/docs/claude-code/settings#environment-variables
// https://docs.claude.com/en/docs/claude-code/costs#background-token-usage
func (i *AnthropicMessagesInterceptionBase) isSmallFastModel() bool {
	return strings.Contains(string(i.req.Model), "haiku")
}
