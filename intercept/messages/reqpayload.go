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
	messagesReqPathModel                     = "model"
	messagesReqPathStream                    = "stream"
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

var (
	constAny        = string(constant.ValueOf[constant.Any]())
	constAuto       = string(constant.ValueOf[constant.Auto]())
	constNone       = string(constant.ValueOf[constant.None]())
	constText       = string(constant.ValueOf[constant.Text]())
	constTool       = string(constant.ValueOf[constant.Tool]())
	constToolResult = string(constant.ValueOf[constant.ToolResult]())
	constUser       = string(anthropic.MessageParamRoleUser)
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
	return gjson.GetBytes(p, messagesReqPathStream).Bool()
}

func (p MessagesRequestPayload) model() string {
	return gjson.GetBytes(p, messagesReqPathModel).String()
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
		return p, err
	}

	allTools := make([]any, 0, len(injected)+len(existing))
	for _, tool := range injected {
		allTools = append(allTools, tool)
	}
	for _, tool := range existing {
		allTools = append(allTools, tool)
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
			return p, err
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
			return p, err
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

func (p MessagesRequestPayload) appendedMessages(messages []anthropic.MessageParam) (MessagesRequestPayload, error) {
	if len(messages) == 0 {
		return p, nil
	}

	existing, err := p.messages()
	if err != nil {
		return p, err
	}

	allMessages := make([]any, 0, len(existing)+len(messages))
	allMessages = append(allMessages, existing...)
	for _, message := range messages {
		allMessages = append(allMessages, message)
	}

	return p.set(messagesReqPathMessages, allMessages)
}

func (p MessagesRequestPayload) withModel(model string) (MessagesRequestPayload, error) {
	return p.set(messagesReqPathModel, model)
}

func (p MessagesRequestPayload) messages() ([]any, error) {
	messages := gjson.GetBytes(p, messagesReqPathMessages)
	if !messages.Exists() || messages.Type == gjson.Null {
		return []any{}, nil
	}
	if !messages.IsArray() {
		return nil, fmt.Errorf("unsupported messages type: %s", messages.Type)
	}

	messageItems := messages.Array()
	existing := make([]any, 0, len(messageItems))
	for _, item := range messageItems {
		existing = append(existing, json.RawMessage(item.Raw))
	}

	return existing, nil
}

func (p MessagesRequestPayload) tools() ([]json.RawMessage, error) {
	tools := gjson.GetBytes(p, messagesReqPathTools)
	if !tools.Exists() || tools.Type == gjson.Null {
		return nil, nil
	}
	if !tools.IsArray() {
		return nil, fmt.Errorf("unsupported tools type: %s", tools.Type)
	}

	toolItems := tools.Array()
	existing := make([]json.RawMessage, 0, len(toolItems))
	for _, item := range toolItems {
		existing = append(existing, json.RawMessage(item.Raw))
	}

	return existing, nil
}

func (p MessagesRequestPayload) set(path string, value any) (MessagesRequestPayload, error) {
	out, err := sjson.SetBytes(p, path, value)
	return MessagesRequestPayload(out), err
}
