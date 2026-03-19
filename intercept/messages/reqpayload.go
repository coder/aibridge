package messages

import (
	"bytes"
	"encoding/json"
	"fmt"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/shared/constant"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

const (
	// Absolute JSON paths from the request root.
	messagesReqPathMessages                  = "messages"
	messagesReqPathMaxTokens                 = "max_tokens"
	messagesReqPathModel                     = "model"
	messagesReqPathOutputConfig              = "output_config"
	messagesReqPathOutputConfigEffort        = "output_config.effort"
	messagesReqPathMetadata                  = "metadata"
	messagesReqPathServiceTier               = "service_tier"
	messagesReqPathContainer                 = "container"
	messagesReqPathInferenceGeo              = "inference_geo"
	messagesReqPathContextManagement         = "context_management"
	messagesReqPathStream                    = "stream"
	messagesReqPathThinking                  = "thinking"
	messagesReqPathThinkingBudgetTokens      = "thinking.budget_tokens"
	messagesReqPathThinkingType              = "thinking.type"
	messagesReqPathToolChoice                = "tool_choice"
	messagesReqPathToolChoiceDisableParallel = "tool_choice.disable_parallel_tool_use"
	messagesReqPathToolChoiceType            = "tool_choice.type"
	messagesReqPathTools                     = "tools"

	// Relative field names used within sub-objects.
	messagesReqFieldContent   = "content"
	messagesReqFieldRole      = "role"
	messagesReqFieldText      = "text"
	messagesReqFieldToolUseID = "tool_use_id"
	messagesReqFieldType      = "type"
)

const (
	constAdaptive = "adaptive"
	constDisabled = "disabled"
	constEnabled  = "enabled"
)

var (
	constAny        = string(constant.ValueOf[constant.Any]())
	constAuto       = string(constant.ValueOf[constant.Auto]())
	constNone       = string(constant.ValueOf[constant.None]())
	constText       = string(constant.ValueOf[constant.Text]())
	constTool       = string(constant.ValueOf[constant.Tool]())
	constToolResult = string(constant.ValueOf[constant.ToolResult]())
	constUser       = string(anthropic.MessageParamRoleUser)

	// bedrockUnsupportedFields are top-level fields present in the Anthropic Messages
	// API that are absent from the Bedrock request body schema. Sending them results
	// in a 400 "Extra inputs are not permitted" error.
	//
	// Anthropic API fields: https://platform.claude.com/docs/en/api/messages/create
	// Bedrock request body: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages-request-response.html
	bedrockUnsupportedFields = []string{
		messagesReqPathOutputConfig, // requires beta header 'effort-2025-11-24'
		messagesReqPathMetadata,
		messagesReqPathServiceTier,
		messagesReqPathContainer,
		messagesReqPathInferenceGeo,
		messagesReqPathContextManagement,
	}
)

// MessagesRequestPayload is raw JSON bytes of an Anthropic Messages API request.
// Methods provide package-specific reads and rewrites while preserving the
// original body for upstream pass-through.
type MessagesRequestPayload []byte

func NewMessagesRequestPayload(raw []byte) (MessagesRequestPayload, error) {
	if len(bytes.TrimSpace(raw)) == 0 {
		return nil, fmt.Errorf("messages empty request body")
	}
	if !json.Valid(raw) {
		return nil, fmt.Errorf("messages invalid JSON request body")
	}

	return MessagesRequestPayload(raw), nil
}

func (p MessagesRequestPayload) Stream() bool {
	v := gjson.GetBytes(p, messagesReqPathStream)
	if !v.IsBool() {
		return false
	}
	return v.Bool()
}

func (p MessagesRequestPayload) model() string {
	return gjson.GetBytes(p, messagesReqPathModel).Str
}

func (p MessagesRequestPayload) correlatingToolCallID() *string {
	messages := gjson.GetBytes(p, messagesReqPathMessages)
	if !messages.IsArray() {
		return nil
	}

	messageItems := messages.Array()
	if len(messageItems) == 0 {
		return nil
	}

	content := messageItems[len(messageItems)-1].Get(messagesReqFieldContent)
	if !content.IsArray() {
		return nil
	}

	contentItems := content.Array()
	for idx := len(contentItems) - 1; idx >= 0; idx-- {
		contentItem := contentItems[idx]
		if contentItem.Get(messagesReqFieldType).String() != constToolResult {
			continue
		}

		toolUseID := contentItem.Get(messagesReqFieldToolUseID).String()
		if toolUseID == "" {
			continue
		}

		return &toolUseID
	}

	return nil
}

// lastUserPrompt returns the prompt text from the last user message. If no prompt
// is found, it returns empty string, false, nil. Unexpected shapes are treated as
// unsupported and do not fail the request path.
func (p MessagesRequestPayload) lastUserPrompt() (string, bool, error) {
	messages := gjson.GetBytes(p, messagesReqPathMessages)
	if !messages.Exists() || messages.Type == gjson.Null {
		return "", false, nil
	}
	if !messages.IsArray() {
		return "", false, fmt.Errorf("unexpected messages type: %s", messages.Type)
	}

	messageItems := messages.Array()
	if len(messageItems) == 0 {
		return "", false, nil
	}

	lastMessage := messageItems[len(messageItems)-1]
	if lastMessage.Get(messagesReqFieldRole).String() != constUser {
		return "", false, nil
	}

	content := lastMessage.Get(messagesReqFieldContent)
	if !content.Exists() || content.Type == gjson.Null {
		return "", false, nil
	}
	if content.Type == gjson.String {
		return content.String(), true, nil
	}
	if !content.IsArray() {
		return "", false, fmt.Errorf("unexpected message content type: %s", content.Type)
	}

	contentItems := content.Array()
	for idx := len(contentItems) - 1; idx >= 0; idx-- {
		contentItem := contentItems[idx]
		if contentItem.Get(messagesReqFieldType).String() != constText {
			continue
		}

		text := contentItem.Get(messagesReqFieldText)
		if text.Type != gjson.String {
			continue
		}

		return text.String(), true, nil
	}

	return "", false, nil
}

func (p MessagesRequestPayload) injectTools(injected []anthropic.ToolUnionParam) (MessagesRequestPayload, error) {
	if len(injected) == 0 {
		return p, nil
	}

	existing, err := p.tools()
	if err != nil {
		return p, fmt.Errorf("get existing tools: %w", err)
	}

	// Using []any to merge differently-typed slices ([]anthropic.ToolUnionParam
	// and []any containing json.RawMessage) keeps JSON re-marshalings to a minimum:
	// sjson.SetBytes marshals each element exactly once, and json.RawMessage
	// elements are passed through without re-serialization.
	allTools := make([]any, 0, len(injected)+len(existing))
	for _, tool := range injected {
		allTools = append(allTools, tool)
	}

	for _, e := range existing {
		allTools = append(allTools, e)
	}

	return p.set(messagesReqPathTools, allTools)
}

func (p MessagesRequestPayload) disableParallelToolCalls() (MessagesRequestPayload, error) {
	toolChoice := gjson.GetBytes(p, messagesReqPathToolChoice)

	// If no tool_choice was defined, assume auto.
	// See https://platform.claude.com/docs/en/agents-and-tools/tool-use/implement-tool-use#parallel-tool-use.
	if !toolChoice.Exists() || toolChoice.Type == gjson.Null {
		updated, err := p.set(messagesReqPathToolChoiceType, constAuto)
		if err != nil {
			return p, fmt.Errorf("set tool choice type: %w", err)
		}
		return updated.set(messagesReqPathToolChoiceDisableParallel, true)
	}
	if !toolChoice.IsObject() {
		return p, fmt.Errorf("unsupported tool_choice type: %s", toolChoice.Type)
	}

	toolChoiceType := gjson.GetBytes(p, messagesReqPathToolChoiceType)
	if toolChoiceType.Exists() && toolChoiceType.Type != gjson.String {
		return p, fmt.Errorf("unsupported tool_choice.type type: %s", toolChoiceType.Type)
	}

	switch toolChoiceType.String() {
	case "":
		updated, err := p.set(messagesReqPathToolChoiceType, constAuto)
		if err != nil {
			return p, fmt.Errorf("set tool_choice.type: %w", err)
		}
		return updated.set(messagesReqPathToolChoiceDisableParallel, true)
	case constAuto, constAny, constTool:
		return p.set(messagesReqPathToolChoiceDisableParallel, true)
	case constNone:
		return p, nil
	default:
		return p, fmt.Errorf("unsupported tool_choice.type value: %q", toolChoiceType.String())
	}
}

func (p MessagesRequestPayload) appendedMessages(newMessages []anthropic.MessageParam) (MessagesRequestPayload, error) {
	if len(newMessages) == 0 {
		return p, nil
	}

	existing, err := p.messages()
	if err != nil {
		return p, fmt.Errorf("get existing messages: %w", err)
	}

	// Using []any to merge differently-typed slices ([]any containing
	// json.RawMessage and []anthropic.MessageParam) keeps JSON re-marshalings
	// to a minimum: sjson.SetBytes marshals each element exactly once, and
	// json.RawMessage elements are passed through without re-serialization.
	allMessages := make([]any, 0, len(existing)+len(newMessages))

	for _, e := range existing {
		allMessages = append(allMessages, e)
	}

	for _, new := range newMessages {
		allMessages = append(allMessages, new)
	}

	return p.set(messagesReqPathMessages, allMessages)
}

func (p MessagesRequestPayload) withModel(model string) (MessagesRequestPayload, error) {
	return p.set(messagesReqPathModel, model)
}

func (p MessagesRequestPayload) messages() ([]json.RawMessage, error) {
	messages := gjson.GetBytes(p, messagesReqPathMessages)
	if !messages.Exists() || messages.Type == gjson.Null {
		return nil, nil
	}
	if !messages.IsArray() {
		return nil, fmt.Errorf("unsupported messages type: %s", messages.Type)
	}

	return p.resultToRawMessage(messages.Array()), nil
}

func (p MessagesRequestPayload) tools() ([]json.RawMessage, error) {
	tools := gjson.GetBytes(p, messagesReqPathTools)
	if !tools.Exists() || tools.Type == gjson.Null {
		return nil, nil
	}
	if !tools.IsArray() {
		return nil, fmt.Errorf("unsupported tools type: %s", tools.Type)
	}

	return p.resultToRawMessage(tools.Array()), nil
}

func (p MessagesRequestPayload) resultToRawMessage(items []gjson.Result) []json.RawMessage {
	// gjson.Result conversion to json.RawMessage is needed because
	// gjson.Result does not implement json.Marshaler — placing it in []any
	// would serialize its struct fields instead of the raw JSON it represents.
	rawMessages := make([]json.RawMessage, 0, len(items))
	for _, item := range items {
		rawMessages = append(rawMessages, json.RawMessage(item.Raw))
	}
	return rawMessages
}

// convertAdaptiveThinkingForBedrock converts thinking.type "adaptive" to "enabled" with a calculated budget_tokens
func (p MessagesRequestPayload) convertAdaptiveThinkingForBedrock() (MessagesRequestPayload, error) {
	thinkingType := gjson.GetBytes(p, messagesReqPathThinkingType)
	if thinkingType.String() != constAdaptive {
		return p, nil
	}

	maxTokens := gjson.GetBytes(p, messagesReqPathMaxTokens).Int()
	if maxTokens <= 0 {
		// max_tokens is required by messages API
		return p, fmt.Errorf("max_tokens: field required")
	}

	effort := gjson.GetBytes(p, messagesReqPathOutputConfigEffort).String()

	// Effort-to-ratio mapping adapted from OpenRouter:
	// https://openrouter.ai/docs/guides/best-practices/reasoning-tokens#reasoning-effort-level
	var ratio float64
	switch effort {
	case "low":
		ratio = 0.2
	case "medium":
		ratio = 0.5
	case "max":
		ratio = 0.95
	default: // "high" or absent (high is the default effort)
		ratio = 0.8
	}

	// budget_tokens must be ≥ 1024 && < max_tokens. If the calculated budget
	// doesn't meet the minimum, disable thinking entirely rather than forcing
	// an artificially high budget that would starve the output.
	// https://platform.claude.com/docs/en/api/messages/create#create.thinking
	// https://platform.claude.com/docs/en/build-with-claude/extended-thinking#how-to-use-extended-thinking
	budgetTokens := int64(float64(maxTokens) * ratio)
	if budgetTokens < 1024 {
		return p.set(messagesReqPathThinking, map[string]string{"type": constDisabled})
	}

	return p.set(messagesReqPathThinking, map[string]any{
		"type":          constEnabled,
		"budget_tokens": budgetTokens,
	})
}

// removeUnsupportedBedrockFields strips all top-level fields that Bedrock does
// not support from the payload.
func (p MessagesRequestPayload) removeUnsupportedBedrockFields() (MessagesRequestPayload, error) {
	result := p
	for _, field := range bedrockUnsupportedFields {
		var err error
		result, err = result.delete(field)
		if err != nil {
			return p, fmt.Errorf("removing %q: %w", field, err)
		}
	}
	return result, nil
}

func (p MessagesRequestPayload) set(path string, value any) (MessagesRequestPayload, error) {
	out, err := sjson.SetBytes(p, path, value)
	if err != nil {
		return p, fmt.Errorf("set %s: %w", path, err)
	}
	return MessagesRequestPayload(out), nil
}

func (p MessagesRequestPayload) delete(path string) (MessagesRequestPayload, error) {
	out, err := sjson.DeleteBytes(p, path)
	return MessagesRequestPayload(out), err
}
