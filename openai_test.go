package aibridge_test

import (
	"testing"

	"github.com/coder/aibridge"
	"github.com/openai/openai-go/v2"
	"github.com/stretchr/testify/require"
)

func TestOpenAILastUserPrompt(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name        string
		wrapper     *aibridge.ChatCompletionNewParamsWrapper
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
			wrapper: &aibridge.ChatCompletionNewParamsWrapper{
				ChatCompletionNewParams: openai.ChatCompletionNewParams{
					Messages: []openai.ChatCompletionMessageParamUnion{},
				},
			},
			expectError: true,
			errorMsg:    "no messages",
		},
		{
			name: "last message not from user",
			wrapper: &aibridge.ChatCompletionNewParamsWrapper{
				ChatCompletionNewParams: openai.ChatCompletionNewParams{
					Messages: []openai.ChatCompletionMessageParamUnion{
						openai.UserMessage("user message"),
						openai.AssistantMessage("assistant message"),
					},
				},
			},
		},
		{
			name: "user message with string content",
			wrapper: &aibridge.ChatCompletionNewParamsWrapper{
				ChatCompletionNewParams: openai.ChatCompletionNewParams{
					Messages: []openai.ChatCompletionMessageParamUnion{
						openai.UserMessage("Hello, world!"),
					},
				},
			},
			expected: "Hello, world!",
		},
		{
			name: "user message with empty string",
			wrapper: &aibridge.ChatCompletionNewParamsWrapper{
				ChatCompletionNewParams: openai.ChatCompletionNewParams{
					Messages: []openai.ChatCompletionMessageParamUnion{
						openai.UserMessage(""),
					},
				},
			},
		},
		{
			name: "user message with array content - text at end",
			wrapper: &aibridge.ChatCompletionNewParamsWrapper{
				ChatCompletionNewParams: openai.ChatCompletionNewParams{
					Messages: []openai.ChatCompletionMessageParamUnion{
						openai.UserMessage([]openai.ChatCompletionContentPartUnionParam{
							openai.ImageContentPart(openai.ChatCompletionContentPartImageImageURLParam{
								URL: "https://example.com/image.png",
							}),
							openai.TextContentPart("First text"),
							openai.ImageContentPart(openai.ChatCompletionContentPartImageImageURLParam{
								URL: "https://example.com/image2.png",
							}),
							openai.TextContentPart("Last text"),
						}),
					},
				},
			},
			expected: "Last text",
		},
		{
			name: "user message with array content - no text",
			wrapper: &aibridge.ChatCompletionNewParamsWrapper{
				ChatCompletionNewParams: openai.ChatCompletionNewParams{
					Messages: []openai.ChatCompletionMessageParamUnion{
						openai.UserMessage([]openai.ChatCompletionContentPartUnionParam{
							openai.ImageContentPart(openai.ChatCompletionContentPartImageImageURLParam{
								URL: "https://example.com/image.png",
							}),
						}),
					},
				},
			},
		},
		{
			name: "user message with empty array",
			wrapper: &aibridge.ChatCompletionNewParamsWrapper{
				ChatCompletionNewParams: openai.ChatCompletionNewParams{
					Messages: []openai.ChatCompletionMessageParamUnion{
						openai.UserMessage([]openai.ChatCompletionContentPartUnionParam{}),
					},
				},
			},
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
				if tt.expected == "" {
					require.Nil(t, result)
				} else {
					require.NotNil(t, result)
					require.Equal(t, tt.expected, *result)
				}
			}
		})
	}
}
