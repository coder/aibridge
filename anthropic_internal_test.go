package aibridge

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

func TestAccumulateUsage(t *testing.T) {
	t.Run("Usage to Usage", func(t *testing.T) {
		dest := &anthropic.Usage{
			InputTokens:              10,
			OutputTokens:             20,
			CacheCreationInputTokens: 5,
			CacheReadInputTokens:     3,
			CacheCreation: anthropic.CacheCreation{
				Ephemeral1hInputTokens: 2,
				Ephemeral5mInputTokens: 1,
			},
			ServerToolUse: anthropic.ServerToolUsage{
				WebSearchRequests: 1,
			},
		}

		source := anthropic.Usage{
			InputTokens:              15,
			OutputTokens:             25,
			CacheCreationInputTokens: 8,
			CacheReadInputTokens:     4,
			CacheCreation: anthropic.CacheCreation{
				Ephemeral1hInputTokens: 3,
				Ephemeral5mInputTokens: 2,
			},
			ServerToolUse: anthropic.ServerToolUsage{
				WebSearchRequests: 2,
			},
		}

		accumulateUsage(dest, source)

		require.EqualValues(t, 25, dest.InputTokens)
		require.EqualValues(t, 45, dest.OutputTokens)
		require.EqualValues(t, 13, dest.CacheCreationInputTokens)
		require.EqualValues(t, 7, dest.CacheReadInputTokens)
		require.EqualValues(t, 5, dest.CacheCreation.Ephemeral1hInputTokens)
		require.EqualValues(t, 3, dest.CacheCreation.Ephemeral5mInputTokens)
		require.EqualValues(t, 3, dest.ServerToolUse.WebSearchRequests)
	})

	t.Run("MessageDeltaUsage to MessageDeltaUsage", func(t *testing.T) {
		dest := &anthropic.MessageDeltaUsage{
			InputTokens:              10,
			OutputTokens:             20,
			CacheCreationInputTokens: 5,
			CacheReadInputTokens:     3,
			ServerToolUse: anthropic.ServerToolUsage{
				WebSearchRequests: 1,
			},
		}

		source := anthropic.MessageDeltaUsage{
			InputTokens:              15,
			OutputTokens:             25,
			CacheCreationInputTokens: 8,
			CacheReadInputTokens:     4,
			ServerToolUse: anthropic.ServerToolUsage{
				WebSearchRequests: 2,
			},
		}

		accumulateUsage(dest, source)

		require.EqualValues(t, 25, dest.InputTokens)
		require.EqualValues(t, 45, dest.OutputTokens)
		require.EqualValues(t, 13, dest.CacheCreationInputTokens)
		require.EqualValues(t, 7, dest.CacheReadInputTokens)
		require.EqualValues(t, 3, dest.ServerToolUse.WebSearchRequests)
	})

	t.Run("Usage to MessageDeltaUsage", func(t *testing.T) {
		dest := &anthropic.MessageDeltaUsage{
			InputTokens:              10,
			OutputTokens:             20,
			CacheCreationInputTokens: 5,
			CacheReadInputTokens:     3,
			ServerToolUse: anthropic.ServerToolUsage{
				WebSearchRequests: 1,
			},
		}

		source := anthropic.Usage{
			InputTokens:              15,
			OutputTokens:             25,
			CacheCreationInputTokens: 8,
			CacheReadInputTokens:     4,
			CacheCreation: anthropic.CacheCreation{
				Ephemeral1hInputTokens: 3, // These won't be accumulated to MessageDeltaUsage
				Ephemeral5mInputTokens: 2,
			},
			ServerToolUse: anthropic.ServerToolUsage{
				WebSearchRequests: 2,
			},
		}

		accumulateUsage(dest, source)

		require.EqualValues(t, 25, dest.InputTokens)
		require.EqualValues(t, 45, dest.OutputTokens)
		require.EqualValues(t, 13, dest.CacheCreationInputTokens)
		require.EqualValues(t, 7, dest.CacheReadInputTokens)
		require.EqualValues(t, 3, dest.ServerToolUse.WebSearchRequests)
	})

	t.Run("MessageDeltaUsage to Usage", func(t *testing.T) {
		dest := &anthropic.Usage{
			InputTokens:              10,
			OutputTokens:             20,
			CacheCreationInputTokens: 5,
			CacheReadInputTokens:     3,
			CacheCreation: anthropic.CacheCreation{
				Ephemeral1hInputTokens: 2,
				Ephemeral5mInputTokens: 1,
			},
			ServerToolUse: anthropic.ServerToolUsage{
				WebSearchRequests: 1,
			},
		}

		source := anthropic.MessageDeltaUsage{
			InputTokens:              15,
			OutputTokens:             25,
			CacheCreationInputTokens: 8,
			CacheReadInputTokens:     4,
			ServerToolUse: anthropic.ServerToolUsage{
				WebSearchRequests: 2,
			},
		}

		accumulateUsage(dest, source)

		require.EqualValues(t, 25, dest.InputTokens)
		require.EqualValues(t, 45, dest.OutputTokens)
		require.EqualValues(t, 13, dest.CacheCreationInputTokens)
		require.EqualValues(t, 7, dest.CacheReadInputTokens)
		// Ephemeral tokens remain unchanged since MessageDeltaUsage doesn't have them
		require.EqualValues(t, 2, dest.CacheCreation.Ephemeral1hInputTokens)
		require.EqualValues(t, 1, dest.CacheCreation.Ephemeral5mInputTokens)
		require.EqualValues(t, 3, dest.ServerToolUse.WebSearchRequests)
	})

	t.Run("Nil or unsupported types", func(t *testing.T) {
		// Test with nil dest
		var nilUsage *anthropic.Usage
		source := anthropic.Usage{InputTokens: 10}
		accumulateUsage(nilUsage, source) // Should not panic

		// Test with unsupported types
		var unsupported string
		accumulateUsage(&unsupported, source) // Should not panic, just do nothing
	})
}
