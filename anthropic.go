package aibridge

import (
	"encoding/json"
	"errors"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/packages/param"
	"github.com/coder/aibridge/utils"
)

// MessageNewParamsWrapper exists because the "stream" param is not included in anthropic.MessageNewParams.
type MessageNewParamsWrapper struct {
	anthropic.MessageNewParams `json:""`
	Stream                     bool `json:"stream,omitempty"`
}

func (b MessageNewParamsWrapper) MarshalJSON() ([]byte, error) {
	type shadow MessageNewParamsWrapper
	return param.MarshalWithExtras(b, (*shadow)(&b), map[string]any{
		"stream": b.Stream,
	})
}

func (b *MessageNewParamsWrapper) UnmarshalJSON(raw []byte) error {
	convertedRaw, err := convertStringContentToArray(raw)
	if err != nil {
		return err
	}

	err = b.MessageNewParams.UnmarshalJSON(convertedRaw)
	if err != nil {
		return err
	}

	b.Stream = utils.ExtractJSONField[bool](raw, "stream")
	return nil
}

func (b *MessageNewParamsWrapper) UseStreaming() bool {
	return b.Stream
}

func (b *MessageNewParamsWrapper) LastUserPrompt() (*string, error) {
	if b == nil {
		return nil, errors.New("nil struct")
	}

	if len(b.Messages) == 0 {
		return nil, errors.New("no messages")
	}

	// We only care if the last message was issued by a user.
	msg := b.Messages[len(b.Messages)-1]
	if msg.Role != anthropic.MessageParamRoleUser {
		return nil, nil
	}

	if len(msg.Content) == 0 {
		return nil, nil
	}

	// Walk backwards on "user"-initiated message content. Clients often inject
	// content ahead of the actual prompt to provide context to the model,
	// so the last item in the slice is most likely the user's prompt.
	for i := len(msg.Content) - 1; i >= 0; i-- {
		// Only text content is supported currently.
		if textContent := msg.Content[i].GetText(); textContent != nil {
			return textContent, nil
		}
	}

	return nil, nil
}

// convertStringContentToArray converts string content to array format for Anthropic messages.
// https://docs.anthropic.com/en/api/messages#body-messages
//
// Each input message content may be either a single string or an array of content blocks, where each block has a
// specific type. Using a string for content is shorthand for an array of one content block of type "text".
func convertStringContentToArray(raw []byte) ([]byte, error) {
	var modifiedJSON map[string]any
	if err := json.Unmarshal(raw, &modifiedJSON); err != nil {
		return raw, err
	}

	// Check if messages exist and need content conversion
	if _, hasMessages := modifiedJSON["messages"]; hasMessages {
		convertStringContentRecursive(modifiedJSON)

		// Marshal back to JSON
		return json.Marshal(modifiedJSON)
	}

	return raw, nil
}

// convertStringContentRecursive recursively scans JSON data and converts string "content" fields
// to proper text block arrays where needed for Anthropic SDK compatibility
func convertStringContentRecursive(data any) {
	switch v := data.(type) {
	case map[string]any:
		// Check if this object has a "content" field with string value
		if content, hasContent := v["content"]; hasContent {
			if contentStr, isString := content.(string); isString {
				// Check if this needs conversion based on context
				if shouldConvertContentField(v) {
					v["content"] = []map[string]any{
						{
							"type": "text",
							"text": contentStr,
						},
					}
				}
			}
		}

		// Recursively process all values in the map
		for _, value := range v {
			convertStringContentRecursive(value)
		}

	case []any:
		// Recursively process all items in the array
		for _, item := range v {
			convertStringContentRecursive(item)
		}
	}
}

// shouldConvertContentField determines if a "content" string field should be converted to text block array
func shouldConvertContentField(obj map[string]any) bool {
	// Check if this is a message-level content (has "role" field)
	if _, hasRole := obj["role"]; hasRole {
		return true
	}

	// Check if this is a tool_result block (but not mcp_tool_result which supports strings)
	if objType, hasType := obj["type"].(string); hasType {
		switch objType {
		case "tool_result":
			return true // Regular tool_result needs array format
		case "mcp_tool_result":
			return false // MCP tool_result supports strings
		}
	}

	return false
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
