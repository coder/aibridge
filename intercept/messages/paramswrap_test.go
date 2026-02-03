package messages

import (
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/stretchr/testify/require"
)

func TestMessageNewParamsWrapperUnmarshalJSON(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name           string
		input          string
		expectedStream bool
		checkContent   func(t *testing.T, w *MessageNewParamsWrapper)
	}{
		{
			name:           "message with string content converts to array",
			input:          `{"model":"claude-3","max_tokens":1000,"messages":[{"role":"user","content":"Hello world"}]}`,
			expectedStream: false,
			checkContent: func(t *testing.T, w *MessageNewParamsWrapper) {
				require.Len(t, w.Messages, 1)
				require.Equal(t, anthropic.MessageParamRoleUser, w.Messages[0].Role)
				text := w.Messages[0].Content[0].GetText()
				require.NotNil(t, text)
				require.Equal(t, "Hello world", *text)
			},
		},
		{
			name:           "stream field extracted",
			input:          `{"model":"claude-3","max_tokens":1000,"stream":true,"messages":[{"role":"user","content":"Hi"}]}`,
			expectedStream: true,
			checkContent: func(t *testing.T, w *MessageNewParamsWrapper) {
				require.Len(t, w.Messages, 1)
			},
		},
		{
			name:           "stream false",
			input:          `{"model":"claude-3","max_tokens":1000,"stream":false,"messages":[{"role":"user","content":"Hi"}]}`,
			expectedStream: false,
			checkContent:   nil,
		},
		{
			name:           "array content unchanged",
			input:          `{"model":"claude-3","max_tokens":1000,"messages":[{"role":"user","content":[{"type":"text","text":"Hello"}]}]}`,
			expectedStream: false,
			checkContent: func(t *testing.T, w *MessageNewParamsWrapper) {
				require.Len(t, w.Messages, 1)
				text := w.Messages[0].Content[0].GetText()
				require.NotNil(t, text)
				require.Equal(t, "Hello", *text)
			},
		},
		{
			name:           "multiple messages with mixed content",
			input:          `{"model":"claude-3","max_tokens":1000,"messages":[{"role":"user","content":"First"},{"role":"assistant","content":[{"type":"text","text":"Response"}]},{"role":"user","content":"Second"}]}`,
			expectedStream: false,
			checkContent: func(t *testing.T, w *MessageNewParamsWrapper) {
				require.Len(t, w.Messages, 3)
				text0 := w.Messages[0].Content[0].GetText()
				require.NotNil(t, text0)
				require.Equal(t, "First", *text0)
				text2 := w.Messages[2].Content[0].GetText()
				require.NotNil(t, text2)
				require.Equal(t, "Second", *text2)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var wrapper MessageNewParamsWrapper
			err := wrapper.UnmarshalJSON([]byte(tt.input))
			require.NoError(t, err)
			require.Equal(t, tt.expectedStream, wrapper.Stream)
			if tt.checkContent != nil {
				tt.checkContent(t, &wrapper)
			}
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
