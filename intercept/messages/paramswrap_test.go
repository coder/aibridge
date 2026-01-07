package messages

import (
	"encoding/json"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/stretchr/testify/require"
)

func TestConvertStringContentToArray(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "empty json",
			input:    `{}`,
			expected: `{}`,
		},
		{
			name: "message with string content",
			input: `{
				"messages": [
					{
						"role": "user",
						"content": "Hello world"
					}
				]
			}`,
			expected: `{"messages":[{"content":[{"text":"Hello world","type":"text"}],"role":"user"}]}`,
		},
		{
			name: "message with array content unchanged",
			input: `{
				"messages": [
					{
						"role": "user",
						"content": [{"type": "text", "text": "Hello"}]
					}
				]
			}`,
			expected: `{"messages":[{"content":[{"text":"Hello","type":"text"}],"role":"user"}]}`,
		},
		{
			name: "multiple messages with mixed content",
			input: `{
				"messages": [
					{
						"role": "user",
						"content": "First message"
					},
					{
						"role": "assistant",
						"content": [{"type": "text", "text": "Response"}]
					},
					{
						"role": "user",
						"content": "Second message"
					}
				]
			}`,
			expected: `{"messages":[{"content":[{"text":"First message","type":"text"}],"role":"user"},{"content":[{"text":"Response","type":"text"}],"role":"assistant"},{"content":[{"text":"Second message","type":"text"}],"role":"user"}]}`,
		},
		{
			name: "tool_result with string content",
			input: `{
				"messages": [
					{
						"role": "user",
						"content": [
							{
								"type": "tool_result",
								"tool_use_id": "123",
								"content": "Tool output"
							}
						]
					}
				]
			}`,
			expected: `{"messages":[{"content":[{"content":[{"text":"Tool output","type":"text"}],"tool_use_id":"123","type":"tool_result"}],"role":"user"}]}`,
		},
		{
			name: "mcp_tool_result with string content unchanged",
			input: `{
				"messages": [
					{
						"role": "user",
						"content": [
							{
								"type": "mcp_tool_result",
								"tool_use_id": "456",
								"content": "MCP output"
							}
						]
					}
				]
			}`,
			expected: `{"messages":[{"content":[{"content":"MCP output","tool_use_id":"456","type":"mcp_tool_result"}],"role":"user"}]}`,
		},
		{
			name: "no messages field",
			input: `{
				"model": "claude-3",
				"max_tokens": 1000
			}`,
			expected: `{"max_tokens":1000,"model":"claude-3"}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := convertStringContentToArray([]byte(tt.input))
			require.NoError(t, err)

			var resultJSON, expectedJSON any
			err = json.Unmarshal(result, &resultJSON)
			require.NoError(t, err)
			err = json.Unmarshal([]byte(tt.expected), &expectedJSON)
			require.NoError(t, err)

			require.Equal(t, expectedJSON, resultJSON)
		})
	}
}

func TestShouldConvertContentField(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		obj      map[string]any
		expected bool
	}{
		{
			name: "message with role",
			obj: map[string]any{
				"role":    "user",
				"content": "test",
			},
			expected: true,
		},
		{
			name: "tool_result type",
			obj: map[string]any{
				"type":    "tool_result",
				"content": "result",
			},
			expected: true,
		},
		{
			name: "mcp_tool_result type",
			obj: map[string]any{
				"type":    "mcp_tool_result",
				"content": "result",
			},
			expected: false,
		},
		{
			name: "other type",
			obj: map[string]any{
				"type":    "text",
				"content": "text",
			},
			expected: false,
		},
		{
			name: "no role or type",
			obj: map[string]any{
				"content": "test",
			},
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := shouldConvertContentField(tt.obj)
			require.Equal(t, tt.expected, result)
		})
	}
}

func TestAnthropicLastUserPrompt(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name        string
		wrapper     *MessageNewParamsWrapper
		expected    string
		expectError bool
		errorMsg    string
	}{
		{
			name:        "nil struct",
			expectError: true,
			errorMsg:    "nil struct",
		},
		{
			name: "no messages",
			wrapper: &MessageNewParamsWrapper{
				MessageNewParams: anthropic.MessageNewParams{
					Messages: []anthropic.MessageParam{},
				},
			},
			expectError: true,
			errorMsg:    "no messages",
		},
		{
			name: "last message not from user",
			wrapper: &MessageNewParamsWrapper{
				MessageNewParams: anthropic.MessageNewParams{
					Messages: []anthropic.MessageParam{
						{
							Role: anthropic.MessageParamRoleUser,
							Content: []anthropic.ContentBlockParamUnion{
								anthropic.NewTextBlock("user message"),
							},
						},
						{
							Role: anthropic.MessageParamRoleAssistant,
							Content: []anthropic.ContentBlockParamUnion{
								anthropic.NewTextBlock("assistant message"),
							},
						},
					},
				},
			},
		},
		{
			name: "last user message with empty content",
			wrapper: &MessageNewParamsWrapper{
				MessageNewParams: anthropic.MessageNewParams{
					Messages: []anthropic.MessageParam{
						{
							Role:    anthropic.MessageParamRoleUser,
							Content: []anthropic.ContentBlockParamUnion{},
						},
					},
				},
			},
		},
		{
			name: "last user message with single text content",
			wrapper: &MessageNewParamsWrapper{
				MessageNewParams: anthropic.MessageNewParams{
					Messages: []anthropic.MessageParam{
						{
							Role: anthropic.MessageParamRoleUser,
							Content: []anthropic.ContentBlockParamUnion{
								anthropic.NewTextBlock("Hello, world!"),
							},
						},
					},
				},
			},
			expected: "Hello, world!",
		},
		{
			name: "last user message with multiple content blocks - text at end",
			wrapper: &MessageNewParamsWrapper{
				MessageNewParams: anthropic.MessageNewParams{
					Messages: []anthropic.MessageParam{
						{
							Role: anthropic.MessageParamRoleUser,
							Content: []anthropic.ContentBlockParamUnion{
								anthropic.NewImageBlockBase64("image/png", "base64data"),
								anthropic.NewTextBlock("First text"),
								anthropic.NewImageBlockBase64("image/jpeg", "moredata"),
								anthropic.NewTextBlock("Last text"),
							},
						},
					},
				},
			},
			expected: "Last text",
		},
		{
			name: "last user message with only non-text content",
			wrapper: &MessageNewParamsWrapper{
				MessageNewParams: anthropic.MessageNewParams{
					Messages: []anthropic.MessageParam{
						{
							Role: anthropic.MessageParamRoleUser,
							Content: []anthropic.ContentBlockParamUnion{
								anthropic.NewImageBlockBase64("image/png", "base64data"),
								anthropic.NewImageBlockBase64("image/jpeg", "moredata"),
							},
						},
					},
				},
			},
		},
		{
			name: "multiple messages with last being user",
			wrapper: &MessageNewParamsWrapper{
				MessageNewParams: anthropic.MessageNewParams{
					Messages: []anthropic.MessageParam{
						{
							Role: anthropic.MessageParamRoleUser,
							Content: []anthropic.ContentBlockParamUnion{
								anthropic.NewTextBlock("First user message"),
							},
						},
						{
							Role: anthropic.MessageParamRoleAssistant,
							Content: []anthropic.ContentBlockParamUnion{
								anthropic.NewTextBlock("Assistant response"),
							},
						},
						{
							Role: anthropic.MessageParamRoleUser,
							Content: []anthropic.ContentBlockParamUnion{
								anthropic.NewTextBlock("Second user message"),
							},
						},
					},
				},
			},
			expected: "Second user message",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := tt.wrapper.lastUserPrompt()

			if tt.expectError {
				require.Error(t, err)
				require.Contains(t, err.Error(), tt.errorMsg)
				require.Nil(t, result)
			} else {
				require.NoError(t, err)
				// Check pointer equality - both nil or both non-nil
				if tt.expected == "" {
					require.Nil(t, result)
				} else {
					require.NotNil(t, result)
					// The result should point to the same string from the content block
					require.Equal(t, tt.expected, *result)
				}
			}
		})
	}
}
