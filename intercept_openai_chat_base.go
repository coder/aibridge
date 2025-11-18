package aibridge

import (
	"context"
	"encoding/json"
	"strings"

	"github.com/coder/aibridge/mcp"
	"github.com/google/uuid"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/shared"

	"cdr.dev/slog"
)

type OpenAIChatInterceptionBase struct {
	id  uuid.UUID
	req *ChatCompletionNewParamsWrapper

	baseURL, key string
	logger       slog.Logger

	recorder Recorder
	mcpProxy mcp.ServerProxier
}

func (i *OpenAIChatInterceptionBase) ID() uuid.UUID {
	return i.id
}

func (i *OpenAIChatInterceptionBase) Setup(logger slog.Logger, recorder Recorder, mcpProxy mcp.ServerProxier) {
	i.logger = logger
	i.recorder = recorder
	i.mcpProxy = mcpProxy
}

func (i *OpenAIChatInterceptionBase) Model() string {
	if i.req == nil {
		return "coder-aibridge-unknown"
	}

	return string(i.req.Model)
}

func (i *OpenAIChatInterceptionBase) newErrorResponse(err error) map[string]any {
	return map[string]any{
		"error":   true,
		"message": err.Error(),
	}
}

func (i *OpenAIChatInterceptionBase) injectTools() {
	if i.req == nil || i.mcpProxy == nil {
		return
	}

	// Inject tools.
	for _, tool := range i.mcpProxy.ListTools() {
		fn := openai.ChatCompletionToolUnionParam{
			OfFunction: &openai.ChatCompletionFunctionToolParam{
				Function: shared.FunctionDefinitionParam{
					Name:        tool.ID,
					Strict:      openai.Bool(false), // TODO: configurable.
					Description: openai.String(tool.Description),
					Parameters: openai.FunctionParameters{
						"type":       "object",
						"properties": tool.Params,
						// "additionalProperties": false, // Only relevant when strict=true.
					},
				},
			},
		}

		// Otherwise the request fails with "None is not of type 'array'" if a nil slice is given.
		if len(tool.Required) > 0 {
			// Must list ALL properties when strict=true.
			fn.OfFunction.Function.Parameters["required"] = tool.Required
		}

		i.req.Tools = append(i.req.Tools, fn)
	}
}

func (i *OpenAIChatInterceptionBase) unmarshalArgs(in string) (args ToolArgs) {
	if len(strings.TrimSpace(in)) == 0 {
		return args // An empty string will fail JSON unmarshaling.
	}

	if err := json.Unmarshal([]byte(in), &args); err != nil {
		i.logger.Warn(context.Background(), "failed to unmarshal tool args", slog.Error(err))
	}

	return args
}
