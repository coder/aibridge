package aibridge_test

import (
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/coder/aibridge"
	"github.com/stretchr/testify/require"
)

func TestAnthropicLastUserPrompt(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name        string
		wrapper     *aibridge.MessageNewParamsWrapper
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
			wrapper: &aibridge.MessageNewParamsWrapper{
				MessageNewParams: anthropic.MessageNewParams{
					Messages: []anthropic.MessageParam{},
				},
			},
			expectError: true,
			errorMsg:    "no messages",
		},
		{
			name: "last message not from user",
			wrapper: &aibridge.MessageNewParamsWrapper{
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
			wrapper: &aibridge.MessageNewParamsWrapper{
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
			wrapper: &aibridge.MessageNewParamsWrapper{
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
			wrapper: &aibridge.MessageNewParamsWrapper{
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
			wrapper: &aibridge.MessageNewParamsWrapper{
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
			wrapper: &aibridge.MessageNewParamsWrapper{
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
			result, err := tt.wrapper.LastUserPrompt()

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
