package messages

import (
	"context"
	"testing"

	"cdr.dev/slog/v3"
	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/shared/constant"
	"github.com/coder/aibridge/config"
	"github.com/coder/aibridge/mcp"
	"github.com/coder/aibridge/utils"
	mcpgo "github.com/mark3labs/mcp-go/mcp"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
)

func TestScanForCorrelatingToolCallID(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name        string
		requestBody string
		expected    *string
	}{
		{
			name:        "no messages field",
			requestBody: `{}`,
			expected:    nil,
		},
		{
			name:        "messages string",
			requestBody: `{"messages":"test"}`,
			expected:    nil,
		},
		{
			name:        "empty messages array",
			requestBody: `{"messages":[]}`,
			expected:    nil,
		},
		{
			name:        "last message has no tool result blocks",
			requestBody: `{"messages":[{"role":"user","content":"hello"}]}`,
			expected:    nil,
		},
		{
			name:        "single tool result block",
			requestBody: `{"messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_abc","content":"result"}]}]}`,
			expected:    utils.PtrTo("toolu_abc"),
		},
		{
			name:        "multiple tool result blocks returns last",
			requestBody: `{"messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_first","content":"first"},{"type":"text","text":"ignored"},{"type":"tool_result","tool_use_id":"toolu_second","content":"second"}]}]}`,
			expected:    utils.PtrTo("toolu_second"),
		},
		{
			name:        "last message is not a tool result",
			requestBody: `{"messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_first","content":"first"}]},{"role":"user","content":"some text"}]}`,
			expected:    nil,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			base := &interceptionBase{
				reqPayload: mustMessagesPayload(t, tc.requestBody),
			}

			require.Equal(t, tc.expected, base.CorrelatingToolCallID())
		})
	}
}

func TestAWSBedrockValidation(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name        string
		cfg         *config.AWSBedrock
		expectError bool
		errorMsg    string
	}{
		// Valid cases.
		{
			name: "valid with region",
			cfg: &config.AWSBedrock{
				Region:          "us-east-1",
				AccessKey:       "test-key",
				AccessKeySecret: "test-secret",
				Model:           "test-model",
				SmallFastModel:  "test-small-model",
			},
		},
		{
			name: "valid with base url",
			cfg: &config.AWSBedrock{
				BaseURL:         "http://bedrock.internal",
				AccessKey:       "test-key",
				AccessKeySecret: "test-secret",
				Model:           "test-model",
				SmallFastModel:  "test-small-model",
			},
		},
		{
			// There unfortunately isn't a way for us to determine precedence in a unit test,
			// since the produced options take a `requestconfig.RequestConfig` input value
			// which is internal to the anthropic SDK.
			//
			// See TestAWSBedrockIntegration which validates this.
			name: "valid with base url & region",
			cfg: &config.AWSBedrock{
				Region:          "us-east-1",
				AccessKey:       "test-key",
				AccessKeySecret: "test-secret",
				Model:           "test-model",
				SmallFastModel:  "test-small-model",
			},
		},
		// Invalid cases.
		{
			name: "missing region & base url",
			cfg: &config.AWSBedrock{
				Region:          "",
				AccessKey:       "test-key",
				AccessKeySecret: "test-secret",
				Model:           "test-model",
				SmallFastModel:  "test-small-model",
			},
			expectError: true,
			errorMsg:    "region or base url required",
		},
		{
			name: "missing access key",
			cfg: &config.AWSBedrock{
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
			cfg: &config.AWSBedrock{
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
			cfg: &config.AWSBedrock{
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
			cfg: &config.AWSBedrock{
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
			cfg:         &config.AWSBedrock{},
			expectError: true,
			errorMsg:    "region or base url required",
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
			base := &interceptionBase{}
			opts, err := base.withAWSBedrockOptions(context.Background(), tt.cfg)

			if tt.expectError {
				require.Error(t, err)
				require.Contains(t, err.Error(), tt.errorMsg)
			} else {
				require.NotEmpty(t, opts)
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

func TestInjectTools_CacheBreakpoints(t *testing.T) {
	t.Parallel()

	t.Run("cache control preserved when no tools to inject", func(t *testing.T) {
		t.Parallel()

		// Request has existing tool with cache control, but no tools to inject.
		i := &interceptionBase{
			reqPayload: mustMessagesPayload(t, `{"tools":[`+
				`{"name":"existing_tool","type":"custom","input_schema":{"type":"object","properties":{}},"cache_control":{"type":"ephemeral"}}]}`),
			mcpProxy: &mockServerProxier{tools: nil},
			logger:   slog.Make(),
		}

		i.injectTools()

		// Cache control should remain untouched since no tools were injected.
		toolItems := gjson.GetBytes(i.reqPayload, "tools").Array()
		require.Len(t, toolItems, 1)
		require.Equal(t, "existing_tool", toolItems[0].Get("name").String())
		require.Equal(t, string(constant.ValueOf[constant.Ephemeral]()), toolItems[0].Get("cache_control.type").String())
	})

	t.Run("cache control breakpoint is preserved by prepending injected tools", func(t *testing.T) {
		t.Parallel()

		// Request has existing tool with cache control.
		i := &interceptionBase{
			reqPayload: mustMessagesPayload(t, `{"tools":[`+
				`{"name":"existing_tool","type":"custom","input_schema":{"type":"object","properties":{}},"cache_control":{"type":"ephemeral"}}]}`),
			mcpProxy: &mockServerProxier{
				tools: []*mcp.Tool{
					{ID: "injected_tool", Name: "injected", Description: "Injected tool"},
				},
			},
			logger: slog.Make(),
		}

		i.injectTools()

		toolItems := gjson.GetBytes(i.reqPayload, "tools").Array()
		require.Len(t, toolItems, 2)
		// Injected tools are prepended.
		require.Equal(t, "injected_tool", toolItems[0].Get("name").String())
		require.Empty(t, toolItems[0].Get("cache_control.type").String())
		// Original tool's cache control should be preserved at the end.
		require.Equal(t, "existing_tool", toolItems[1].Get("name").String())
		require.Equal(t, string(constant.ValueOf[constant.Ephemeral]()), toolItems[1].Get("cache_control.type").String())
	})

	// The cache breakpoint SHOULD be on the final tool, but may not be; we must preserve that intention.
	t.Run("cache control breakpoint in non-standard location is preserved", func(t *testing.T) {
		t.Parallel()

		// Request has multiple tools with cache control breakpoints.
		i := &interceptionBase{
			reqPayload: mustMessagesPayload(t, `{"tools":[`+
				`{"name":"tool_with_cache_1","type":"custom","input_schema":{"type":"object","properties":{}},"cache_control":{"type":"ephemeral"}},`+
				`{"name":"tool_with_cache_2","type":"custom","input_schema":{"type":"object","properties":{}}}]}`),
			mcpProxy: &mockServerProxier{
				tools: []*mcp.Tool{
					{ID: "injected_tool", Name: "injected", Description: "Injected tool"},
				},
			},
			logger: slog.Make(),
		}

		i.injectTools()

		toolItems := gjson.GetBytes(i.reqPayload, "tools").Array()
		require.Len(t, toolItems, 3)
		// Injected tool is prepended without cache control.
		require.Equal(t, "injected_tool", toolItems[0].Get("name").String())
		require.Empty(t, toolItems[0].Get("cache_control.type").String())
		// Both original tools' cache controls should remain.
		require.Equal(t, "tool_with_cache_1", toolItems[1].Get("name").String())
		require.Equal(t, string(constant.ValueOf[constant.Ephemeral]()), toolItems[1].Get("cache_control.type").String())
		require.Equal(t, "tool_with_cache_2", toolItems[2].Get("name").String())
		require.Empty(t, toolItems[2].Get("cache_control.type").String())
	})

	t.Run("no cache control added when none originally set", func(t *testing.T) {
		t.Parallel()

		// Request has tools but none with cache control.
		i := &interceptionBase{
			reqPayload: mustMessagesPayload(t, `{"tools":[`+
				`{"name":"existing_tool_no_cache","type":"custom","input_schema":{"type":"object","properties":{}}}]}`),
			mcpProxy: &mockServerProxier{
				tools: []*mcp.Tool{
					{ID: "injected_tool", Name: "injected", Description: "Injected tool"},
				},
			},
			logger: slog.Make(),
		}

		i.injectTools()

		toolItems := gjson.GetBytes(i.reqPayload, "tools").Array()
		require.Len(t, toolItems, 2)
		// Injected tool is prepended without cache control.
		require.Equal(t, "injected_tool", toolItems[0].Get("name").String())
		require.Empty(t, toolItems[0].Get("cache_control.type").String())
		// Original tool remains at the end without cache control.
		require.Equal(t, "existing_tool_no_cache", toolItems[1].Get("name").String())
		require.Empty(t, toolItems[1].Get("cache_control.type").String())
	})
}

func TestInjectTools_ParallelToolCalls(t *testing.T) {
	t.Parallel()

	t.Run("does not modify tool choice when no tools to inject", func(t *testing.T) {
		t.Parallel()

		i := &interceptionBase{
			reqPayload: mustMessagesPayload(t, `{"tool_choice":{"type":"auto"}}`),
			mcpProxy:   &mockServerProxier{tools: nil}, // No tools to inject.
			logger:     slog.Make(),
		}

		i.injectTools()

		// Tool choice should remain unchanged - DisableParallelToolUse should not be set.
		toolChoice := gjson.GetBytes(i.reqPayload, "tool_choice")
		require.Equal(t, string(constant.ValueOf[constant.Auto]()), toolChoice.Get("type").String())
		require.False(t, toolChoice.Get("disable_parallel_tool_use").Exists())
	})

	t.Run("disables parallel tool use for empty tool choice (default)", func(t *testing.T) {
		t.Parallel()

		i := &interceptionBase{
			reqPayload: mustMessagesPayload(t, `{}`),
			mcpProxy: &mockServerProxier{
				tools: []*mcp.Tool{{ID: "test_tool", Name: "test", Description: "Test"}},
			},
			logger: slog.Make(),
		}

		i.injectTools()

		toolChoice := gjson.GetBytes(i.reqPayload, "tool_choice")
		require.Equal(t, string(constant.ValueOf[constant.Auto]()), toolChoice.Get("type").String())
		require.True(t, toolChoice.Get("disable_parallel_tool_use").Exists())
		require.True(t, toolChoice.Get("disable_parallel_tool_use").Bool())
	})

	t.Run("disables parallel tool use for explicit auto tool choice", func(t *testing.T) {
		t.Parallel()

		i := &interceptionBase{
			reqPayload: mustMessagesPayload(t, `{"tool_choice":{"type":"auto"}}`),
			mcpProxy: &mockServerProxier{
				tools: []*mcp.Tool{{ID: "test_tool", Name: "test", Description: "Test"}},
			},
			logger: slog.Make(),
		}

		i.injectTools()

		toolChoice := gjson.GetBytes(i.reqPayload, "tool_choice")
		require.Equal(t, string(constant.ValueOf[constant.Auto]()), toolChoice.Get("type").String())
		require.True(t, toolChoice.Get("disable_parallel_tool_use").Exists())
		require.True(t, toolChoice.Get("disable_parallel_tool_use").Bool())
	})

	t.Run("disables parallel tool use for any tool choice", func(t *testing.T) {
		t.Parallel()

		i := &interceptionBase{
			reqPayload: mustMessagesPayload(t, `{"tool_choice":{"type":"any"}}`),
			mcpProxy: &mockServerProxier{
				tools: []*mcp.Tool{{ID: "test_tool", Name: "test", Description: "Test"}},
			},
			logger: slog.Make(),
		}

		i.injectTools()

		toolChoice := gjson.GetBytes(i.reqPayload, "tool_choice")
		require.Equal(t, string(constant.ValueOf[constant.Any]()), toolChoice.Get("type").String())
		require.True(t, toolChoice.Get("disable_parallel_tool_use").Exists())
		require.True(t, toolChoice.Get("disable_parallel_tool_use").Bool())
	})

	t.Run("disables parallel tool use for tool choice type", func(t *testing.T) {
		t.Parallel()

		i := &interceptionBase{
			reqPayload: mustMessagesPayload(t, `{"tool_choice":{"type":"tool","name":"specific_tool"}}`),
			mcpProxy: &mockServerProxier{
				tools: []*mcp.Tool{{ID: "test_tool", Name: "test", Description: "Test"}},
			},
			logger: slog.Make(),
		}

		i.injectTools()

		toolChoice := gjson.GetBytes(i.reqPayload, "tool_choice")
		require.Equal(t, string(constant.ValueOf[constant.Tool]()), toolChoice.Get("type").String())
		require.True(t, toolChoice.Get("disable_parallel_tool_use").Exists())
		require.True(t, toolChoice.Get("disable_parallel_tool_use").Bool())
	})

	t.Run("no-op for none tool choice type", func(t *testing.T) {
		t.Parallel()

		i := &interceptionBase{
			reqPayload: mustMessagesPayload(t, `{"tool_choice":{"type":"none"}}`),
			mcpProxy: &mockServerProxier{
				tools: []*mcp.Tool{{ID: "test_tool", Name: "test", Description: "Test"}},
			},
			logger: slog.Make(),
		}

		i.injectTools()

		// Tools are still injected.
		require.Len(t, gjson.GetBytes(i.reqPayload, "tools").Array(), 1)
		// But no parallel tool use modification for "none" type.
		toolChoice := gjson.GetBytes(i.reqPayload, "tool_choice")
		require.Equal(t, string(constant.ValueOf[constant.None]()), toolChoice.Get("type").String())
		require.False(t, toolChoice.Get("disable_parallel_tool_use").Exists())
	})
}

func mustMessagesPayload(t *testing.T, requestBody string) MessagesRequestPayload {
	t.Helper()

	payload, err := NewMessagesRequestPayload([]byte(requestBody))
	require.NoError(t, err)

	return payload
}

// mockServerProxier is a test implementation of mcp.ServerProxier.
type mockServerProxier struct {
	tools []*mcp.Tool
}

func (m *mockServerProxier) Init(context.Context) error {
	return nil
}

func (m *mockServerProxier) Shutdown(context.Context) error {
	return nil
}

func (m *mockServerProxier) ListTools() []*mcp.Tool {
	return m.tools
}

func (m *mockServerProxier) GetTool(id string) *mcp.Tool {
	for _, t := range m.tools {
		if t.ID == id {
			return t
		}
	}
	return nil
}

func (m *mockServerProxier) CallTool(context.Context, string, any) (*mcpgo.CallToolResult, error) {
	return nil, nil
}
