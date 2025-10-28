package aibridge_test

import (
	"encoding/json"
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

func TestMaxCompletionTokens(t *testing.T) {
	t.Parallel()

	t.Run("unmarshal max_completion_tokens from JSON", func(t *testing.T) {
		jsonStr := `{
			"model": "gpt-4o",
			"messages": [{"role": "user", "content": "Hello"}],
			"max_completion_tokens": 1024
		}`

		var wrapper aibridge.ChatCompletionNewParamsWrapper
		err := json.Unmarshal([]byte(jsonStr), &wrapper)
		require.NoError(t, err)
		require.NotNil(t, wrapper.MaxCompletionTokens)
		require.Equal(t, 1024, *wrapper.MaxCompletionTokens)
	})

	t.Run("unmarshal max_completion_tokens with zero value ignored", func(t *testing.T) {
		jsonStr := `{
			"model": "gpt-4o",
			"messages": [{"role": "user", "content": "Hello"}],
			"max_completion_tokens": 0
		}`

		var wrapper aibridge.ChatCompletionNewParamsWrapper
		err := json.Unmarshal([]byte(jsonStr), &wrapper)
		require.NoError(t, err)
		require.Nil(t, wrapper.MaxCompletionTokens, "max_completion_tokens should not be set when 0 (invalid value)")
	})

	t.Run("unmarshal max_completion_tokens with negative value ignored", func(t *testing.T) {
		jsonStr := `{
			"model": "gpt-4o",
			"messages": [{"role": "user", "content": "Hello"}],
			"max_completion_tokens": -100
		}`

		var wrapper aibridge.ChatCompletionNewParamsWrapper
		err := json.Unmarshal([]byte(jsonStr), &wrapper)
		require.NoError(t, err)
		require.Nil(t, wrapper.MaxCompletionTokens, "max_completion_tokens should not be set when negative (invalid value)")
	})

	t.Run("marshal max_completion_tokens to JSON", func(t *testing.T) {
		maxTokens := 2048
		wrapper := aibridge.ChatCompletionNewParamsWrapper{
			ChatCompletionNewParams: openai.ChatCompletionNewParams{
				Model: openai.ChatModelGPT4o,
				Messages: []openai.ChatCompletionMessageParamUnion{
					openai.UserMessage("Hello"),
				},
			},
			MaxCompletionTokens: &maxTokens,
		}

		jsonBytes, err := json.Marshal(wrapper)
		require.NoError(t, err)

		var result map[string]interface{}
		err = json.Unmarshal(jsonBytes, &result)
		require.NoError(t, err)
		require.Equal(t, float64(2048), result["max_completion_tokens"])
	})

	t.Run("max_completion_tokens not set when nil", func(t *testing.T) {
		wrapper := aibridge.ChatCompletionNewParamsWrapper{
			ChatCompletionNewParams: openai.ChatCompletionNewParams{
				Model: openai.ChatModelGPT4o,
				Messages: []openai.ChatCompletionMessageParamUnion{
					openai.UserMessage("Hello"),
				},
			},
		}

		jsonBytes, err := json.Marshal(wrapper)
		require.NoError(t, err)

		var result map[string]interface{}
		err = json.Unmarshal(jsonBytes, &result)
		require.NoError(t, err)
		_, exists := result["max_completion_tokens"]
		require.False(t, exists, "max_completion_tokens should not be present when nil")
	})
}
