package aibridge

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"cdr.dev/slog"
	"github.com/anthropics/anthropic-sdk-go"
	"github.com/google/uuid"
	"github.com/stretchr/testify/require"
)

// noopRecorder is a no-op implementation of Recorder for testing
type noopRecorder struct{}

func (n *noopRecorder) RecordInterception(ctx context.Context, req *InterceptionRecord) error {
	return nil
}

func (n *noopRecorder) RecordInterceptionEnded(ctx context.Context, req *InterceptionRecordEnded) error {
	return nil
}

func (n *noopRecorder) RecordTokenUsage(ctx context.Context, req *TokenUsageRecord) error {
	return nil
}

func (n *noopRecorder) RecordPromptUsage(ctx context.Context, req *PromptUsageRecord) error {
	return nil
}

func (n *noopRecorder) RecordToolUsage(ctx context.Context, req *ToolUsageRecord) error {
	return nil
}

func TestCustomHeadersIntegration(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name          string
		customHeaders map[string]string
		expectedInReq map[string]string
	}{
		{
			name: "single custom header",
			customHeaders: map[string]string{
				"X-Custom-Header": "test-value",
			},
			expectedInReq: map[string]string{
				"X-Custom-Header": "test-value",
			},
		},
		{
			name: "multiple custom headers",
			customHeaders: map[string]string{
				"X-Custom-Header-1": "value1",
				"X-Custom-Header-2": "value2",
			},
			expectedInReq: map[string]string{
				"X-Custom-Header-1": "value1",
				"X-Custom-Header-2": "value2",
			},
		},
		{
			name:          "no custom headers",
			customHeaders: nil,
			expectedInReq: map[string]string{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			// Track which headers were received
			receivedHeaders := make(map[string]string)
			var headerMu sync.Mutex

			// Create a mock server that captures headers
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				headerMu.Lock()
				defer headerMu.Unlock()

				// Capture custom headers
				for key := range tt.expectedInReq {
					if val := r.Header.Get(key); val != "" {
						receivedHeaders[key] = val
					}
				}

				// Return a minimal valid response
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusOK)
				// Return a simple JSON response that matches Anthropic's Message structure
				_, _ = w.Write([]byte(`{
					"id": "msg_123",
					"type": "message",
					"role": "assistant",
					"content": [{"type": "text", "text": "test"}],
					"model": "claude-3-5-sonnet-20241022",
					"usage": {"input_tokens": 10, "output_tokens": 20}
				}`))
			}))
			defer srv.Close()

			// Create config with custom headers
			cfg := AnthropicConfig{
				ProviderConfig: ProviderConfig{
					BaseURL: srv.URL,
					Key:     "test-key",
				},
				CustomHeaders: tt.customHeaders,
			}

			// Create a simple message request
			req := &MessageNewParamsWrapper{
				MessageNewParams: anthropic.MessageNewParams{
					Model:     "claude-3-5-sonnet-20241022",
					MaxTokens: 100,
					Messages: []anthropic.MessageParam{
						{
							Role: anthropic.MessageParamRoleUser,
							Content: []anthropic.ContentBlockParamUnion{
								anthropic.NewTextBlock("test"),
							},
						},
					},
				},
			}

			interception := NewAnthropicMessagesBlockingInterception(uuid.New(), req, cfg, nil)

			// Make request
			w := httptest.NewRecorder()
			r := httptest.NewRequest("POST", "/anthropic/v1/messages", nil)

			logger := slog.Make()
			// Use a no-op recorder to avoid nil pointer issues
			recorder := &noopRecorder{}
			interception.Setup(logger, recorder, nil)

			err := interception.ProcessRequest(w, r)
			require.NoError(t, err)

			// Verify headers were sent
			headerMu.Lock()
			defer headerMu.Unlock()
			require.Equal(t, tt.expectedInReq, receivedHeaders)
		})
	}
}

func TestParseCustomHeaders(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		input    string
		expected map[string]string
	}{
		{
			name:     "empty string",
			input:    "",
			expected: nil,
		},
		{
			name:     "single header",
			input:    "X-Custom-Header: value123",
			expected: map[string]string{"X-Custom-Header": "value123"},
		},
		{
			name:     "multiple headers",
			input:    "X-Custom-Header: value1\nX-Another-Header: value2",
			expected: map[string]string{"X-Custom-Header": "value1", "X-Another-Header": "value2"},
		},
		{
			name:     "header with URL value",
			input:    "X-Callback-URL: https://example.com:8080/path",
			expected: map[string]string{"X-Callback-URL": "https://example.com:8080/path"},
		},
		{
			name:     "header with leading/trailing spaces",
			input:    "  X-Custom-Header  :  value with spaces  ",
			expected: map[string]string{"X-Custom-Header": "value with spaces"},
		},
		{
			name:     "empty lines ignored",
			input:    "X-Header-1: value1\n\n\nX-Header-2: value2\n",
			expected: map[string]string{"X-Header-1": "value1", "X-Header-2": "value2"},
		},
		{
			name:     "malformed header without colon",
			input:    "InvalidHeader\nX-Valid-Header: value",
			expected: map[string]string{"X-Valid-Header": "value"},
		},
		{
			name:     "header with empty key ignored",
			input:    ": value\nX-Valid-Header: value",
			expected: map[string]string{"X-Valid-Header": "value"},
		},
		{
			name:     "header with empty value allowed",
			input:    "X-Empty-Header:",
			expected: map[string]string{"X-Empty-Header": ""},
		},
		{
			name:     "multiple colons in value",
			input:    "X-JSON: {\"key\":\"value\"}",
			expected: map[string]string{"X-JSON": "{\"key\":\"value\"}"},
		},
		{
			name:     "all malformed headers returns nil",
			input:    "NoColonHere\nAnotherInvalid",
			expected: nil,
		},
		{
			name:     "mixed valid and invalid headers",
			input:    "X-Valid: value1\nInvalidLine\nX-Another: value2",
			expected: map[string]string{"X-Valid": "value1", "X-Another": "value2"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := parseCustomHeaders(tt.input)
			require.Equal(t, tt.expected, result)
		})
	}
}

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

func TestAWSBedrockValidation(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name        string
		cfg         *AWSBedrockConfig
		expectError bool
		errorMsg    string
	}{
		{
			name: "valid",
			cfg: &AWSBedrockConfig{
				Region:          "us-east-1",
				AccessKey:       "test-key",
				AccessKeySecret: "test-secret",
				Model:           "test-model",
				SmallFastModel:  "test-small-model",
			},
		},
		{
			name: "missing region",
			cfg: &AWSBedrockConfig{
				Region:          "",
				AccessKey:       "test-key",
				AccessKeySecret: "test-secret",
				Model:           "test-model",
				SmallFastModel:  "test-small-model",
			},
			expectError: true,
			errorMsg:    "region required",
		},
		{
			name: "missing access key",
			cfg: &AWSBedrockConfig{
				Region:          "us-east-1",
				AccessKey:       "",
				AccessKeySecret: "test-secret",
				Model:           "test-model",
				SmallFastModel:  "test-small-model",
			},
			expectError: true,
			errorMsg:    "access key required",
		},
		{
			name: "missing access key secret",
			cfg: &AWSBedrockConfig{
				Region:          "us-east-1",
				AccessKey:       "test-key",
				AccessKeySecret: "",
				Model:           "test-model",
				SmallFastModel:  "test-small-model",
			},
			expectError: true,
			errorMsg:    "access key secret required",
		},
		{
			name: "missing model",
			cfg: &AWSBedrockConfig{
				Region:          "us-east-1",
				AccessKey:       "test-key",
				AccessKeySecret: "test-secret",
				Model:           "",
				SmallFastModel:  "test-small-model",
			},
			expectError: true,
			errorMsg:    "model required",
		},
		{
			name: "missing small fast model",
			cfg: &AWSBedrockConfig{
				Region:          "us-east-1",
				AccessKey:       "test-key",
				AccessKeySecret: "test-secret",
				Model:           "test-model",
				SmallFastModel:  "",
			},
			expectError: true,
			errorMsg:    "small fast model required",
		},
		{
			name:        "all fields empty",
			cfg:         &AWSBedrockConfig{},
			expectError: true,
			errorMsg:    "region required",
		},
		{
			name:        "nil config",
			cfg:         nil,
			expectError: true,
			errorMsg:    "nil config given",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			base := &AnthropicMessagesInterceptionBase{}
			_, err := base.withAWSBedrock(context.Background(), tt.cfg)

			if tt.expectError {
				require.Error(t, err)
				require.Contains(t, err.Error(), tt.errorMsg)
			} else {
				require.NoError(t, err)
			}
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
