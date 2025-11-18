package aibridge

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"

	"github.com/coder/aibridge/mcp"
	"github.com/coder/aibridge/utils"
	"github.com/google/uuid"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"

	"cdr.dev/slog"
)

type OpenAIResponsesInterceptionBase struct {
	id  uuid.UUID
	req *ResponsesNewParamsWrapper

	baseURL, key string
	logger       slog.Logger

	recorder Recorder
	mcpProxy mcp.ServerProxier
}

func (i *OpenAIResponsesInterceptionBase) ID() uuid.UUID {
	return i.id
}

func (i *OpenAIResponsesInterceptionBase) Setup(logger slog.Logger, recorder Recorder, mcpProxy mcp.ServerProxier) {
	i.logger = logger
	i.recorder = recorder
	i.mcpProxy = mcpProxy
}

func (i *OpenAIResponsesInterceptionBase) Model() string {
	if i.req == nil {
		return "coder-aibridge-unknown"
	}

	return string(i.req.Model)
}

func (i *OpenAIResponsesInterceptionBase) injectTools() {
	if i.req == nil || i.mcpProxy == nil {
		return
	}

	// Inject tools.
	for _, tool := range i.mcpProxy.ListTools() {
		fn := responses.ToolUnionParam{
			OfFunction: &responses.FunctionToolParam{
				Name:        tool.ID,
				Strict:      openai.Bool(false), // TODO: configurable.
				Description: openai.String(tool.Description),
				Parameters: openai.FunctionParameters{
					"type":       "object",
					"properties": tool.Params,
					// "additionalProperties": false, // Only relevant when strict=true.
				},
			},
		}

		// TODO: does this fail on responses? is this needed?
		// Otherwise the request fails with "None is not of type 'array'" if a nil slice is given.
		if len(tool.Required) > 0 {
			// Must list ALL properties when strict=true.
			fn.OfFunction.Parameters["required"] = tool.Required
		}

		i.req.Tools = append(i.req.Tools, fn)
	}
}

// appendResponseOutput appends the output of the response to the input of the
// request.
func (i *OpenAIResponsesInterceptionBase) appendResponseOutput(response *responses.Response) {
	if i.req == nil || response == nil {
		return
	}

	// If the input is a string, convert it to an array of objects.
	if i.req.Input.OfString.Valid() {
		input := i.req.Input.OfString.String()
		i.req.Input.OfString = param.Opt[string]{}
		i.req.Input.OfInputItemList = responses.ResponseInputParam{
			{
				OfMessage: &responses.EasyInputMessageParam{
					Role: responses.EasyInputMessageRoleUser,
					Content: responses.EasyInputMessageContentUnionParam{
						OfString: param.NewOpt(input),
					},
				},
			},
		}
	}

	for _, output := range response.Output {
		input, err := openAIResponsesOutputToInput(output)
		if err != nil {
			i.logger.Warn(context.Background(), "failed to convert response output to input", slog.Error(err))
			continue
		}
		i.req.Input.OfInputItemList = append(i.req.Input.OfInputItemList, input)
	}
}

func openAIResponsesOutputToInput(output responses.ResponseOutputItemUnion) (responses.ResponseInputItemUnionParam, error) {
	inputParam := responses.ResponseInputItemUnionParam{}

	switch v := output.AsAny().(type) {
	case responses.ResponseOutputMessage:
		// inputParam.OfMessage = utils.PtrTo(v.ToParam())
	case responses.ResponseFileSearchToolCall:
		inputParam.OfFileSearchCall = utils.PtrTo(v.ToParam())
	case responses.ResponseFunctionToolCall:
		inputParam.OfFunctionCall = utils.PtrTo(v.ToParam())
	case responses.ResponseFunctionWebSearch:
		inputParam.OfWebSearchCall = utils.PtrTo(v.ToParam())
	case responses.ResponseComputerToolCall:
		inputParam.OfComputerCall = utils.PtrTo(v.ToParam())
	case responses.ResponseReasoningItem:
		inputParam.OfReasoning = utils.PtrTo(v.ToParam())
	case responses.ResponseOutputItemImageGenerationCall:
		// No ToParam() method.
		inputParam.OfImageGenerationCall = &responses.ResponseInputItemImageGenerationCallParam{
			Result: param.NewOpt(v.Result),
			ID:     v.ID,
			Status: v.Status,
		}
	case responses.ResponseCodeInterpreterToolCall:
		inputParam.OfCodeInterpreterCall = utils.PtrTo(v.ToParam())
	case responses.ResponseOutputItemLocalShellCall:
		// No ToParam() method.
		inputParam.OfLocalShellCall = &responses.ResponseInputItemLocalShellCallParam{
			ID: v.ID,
			Action: responses.ResponseInputItemLocalShellCallActionParam{
				Command:          v.Action.Command,
				Env:              v.Action.Env,
				TimeoutMs:        param.NewOpt(v.Action.TimeoutMs),
				User:             param.NewOpt(v.Action.User),
				WorkingDirectory: param.NewOpt(v.Action.WorkingDirectory),
				Type:             v.Action.Type,
			},
			CallID: v.CallID,
			Status: v.Status,
		}
	default:
		return inputParam, fmt.Errorf("unexpected response.Output item type: %T (%q)", v, output.Type)
	}

	return inputParam, nil
}

func (i *OpenAIResponsesInterceptionBase) appendToolCallOutput(toolCall responses.ResponseCustomToolCall, output string) {
	i.req.Input.OfInputItemList = append(i.req.Input.OfInputItemList, responses.ResponseInputItemUnionParam{
		OfCustomToolCallOutput: &responses.ResponseCustomToolCallOutputParam{
			CallID: toolCall.CallID,
			Output: responses.ResponseCustomToolCallOutputOutputUnionParam{
				OfString: param.NewOpt(output),
			},
		},
	})
}

func (i *OpenAIResponsesInterceptionBase) appendToolCallError(toolCall responses.ResponseCustomToolCall, message string) {
	response := map[string]any{
		"error":   true,
		"message": message,
	}

	// Pretty much infallible since the map is always valid JSON. Not much we
	// can do if it fails anyway.
	responseBytes, err := json.Marshal(response)
	if err != nil {
		responseBytes = []byte("{}")
	}

	i.appendToolCallOutput(toolCall, string(responseBytes))
}

func (i *OpenAIResponsesInterceptionBase) writeErrorResponse(w http.ResponseWriter, status int, code, message string) {
	response := map[string]any{
		"status": "failed",
		"error": map[string]any{
			"code":    code,
			"message": message,
		},
	}

	responseBytes, err := json.Marshal(response)
	if err != nil {
		status = http.StatusInternalServerError
		responseBytes = []byte(`{"status": "failed", "error": {"code": "internal_server_error", "message": "Internal server error"}}`)
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	w.Write(responseBytes)
}

// ResponsesNewParamsWrapper exists because the "stream" param is not included
// in responses.ResponseNewParams.
type ResponsesNewParamsWrapper struct {
	responses.ResponseNewParams `json:""`
	Stream                      bool `json:"stream,omitempty"`
}

func (c ResponsesNewParamsWrapper) MarshalJSON() ([]byte, error) {
	type shadow ResponsesNewParamsWrapper
	return param.MarshalWithExtras(c, (*shadow)(&c), map[string]any{
		"stream": c.Stream,
	})
}

func (c *ResponsesNewParamsWrapper) UnmarshalJSON(raw []byte) error {
	err := c.ResponseNewParams.UnmarshalJSON(raw)
	if err != nil {
		return err
	}

	if stream := utils.ExtractJSONField[bool](raw, "stream"); stream {
		c.Stream = stream
		if c.Stream {
			// TODO: always include usage when streaming.
		} else {
			// TODO: this?
		}
	} else {
		// TODO: this?
	}

	return nil
}

func (c *ResponsesNewParamsWrapper) LastUserPrompt() (*string, error) {
	if c == nil {
		return nil, errors.New("nil struct")
	}

	// Plain text input, assume this is a prompt directly from the user.
	if inputStr := c.ResponseNewParams.Input.OfString; inputStr.Valid() {
		return utils.PtrTo(inputStr.String()), nil
	}

	// Walk backwards on "user"-initiated message content. Clients often inject
	// content ahead of the actual prompt to provide context to the model,
	// so the last item in the slice is most likely the user's prompt.
	for i := len(c.ResponseNewParams.Input.OfInputItemList) - 1; i >= 0; i-- {
		item := c.ResponseNewParams.Input.OfInputItemList[i]
		// Must be a Message from the User.
		if item.OfMessage.Type == responses.EasyInputMessageTypeMessage && item.OfMessage.Role == responses.EasyInputMessageRoleUser {
			// If the content is a plain string, return it.
			if item.OfMessage.Content.OfString.Valid() {
				return utils.PtrTo(item.OfMessage.Content.OfString.String()), nil
			}

			// Otherwise if it's a list, take the last input_text part.
			for i := len(item.OfMessage.Content.OfInputItemContentList) - 1; i >= 0; i-- {
				if textContent := item.OfMessage.Content.OfInputItemContentList[i].OfInputText; textContent != nil {
					return &textContent.Text, nil
				}
			}
		}
	}

	return nil, nil
}

func sumOpenAIResponsesUsage(ref, in responses.ResponseUsage) responses.ResponseUsage {
	return responses.ResponseUsage{
		InputTokens: ref.InputTokens + in.InputTokens,
		InputTokensDetails: responses.ResponseUsageInputTokensDetails{
			CachedTokens: ref.InputTokensDetails.CachedTokens + in.InputTokensDetails.CachedTokens,
		},
		OutputTokens: ref.OutputTokens + in.OutputTokens,
		OutputTokensDetails: responses.ResponseUsageOutputTokensDetails{
			ReasoningTokens: ref.OutputTokensDetails.ReasoningTokens + in.OutputTokensDetails.ReasoningTokens,
		},
		TotalTokens: ref.TotalTokens + in.TotalTokens,
	}
}

// calculateOpenAICompletionsActualInputTokenUsage accounts for cached tokens
// which are included in [responses.ResponseUsage].InputTokens.
// TODO: verify that this is the same for completions or no longer the case for
// responses.
func calculateOpenAIResponsesActualInputTokenUsage(in responses.ResponseUsage) int64 {
	// Input *includes* the cached tokens, so we subtract them here to reflect actual input token usage.
	// The original value can be reconstructed by referencing the "input_cached" field in metadata.
	return in.InputTokens /* The aggregated number of text input tokens used, including cached tokens. */ -
		in.InputTokensDetails.CachedTokens /* The aggregated number of text input tokens that has been cached from previous requests. */
}
