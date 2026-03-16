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

	testCases := []struct {
		name           string
		requestBody    string
		expectedToolID *string
	}{
		{
			name:           "no messages field",
			requestBody:    `{"model":"claude-opus-4-5","max_tokens":1024}`,
			expectedToolID: nil,
		},
		{
			name:           "messages string",
			requestBody:    `{"model":"claude-opus-4-5","max_tokens":1024,"messages":"test"}`,
			expectedToolID: nil,
		},
		{
			name:           "empty messages array",
			requestBody:    `{"model":"claude-opus-4-5","max_tokens":1024,"messages":[]}`,
			expectedToolID: nil,
		},
		{
			name:           "last message has no tool result blocks",
			requestBody:    `{"model":"claude-opus-4-5","max_tokens":1024,"messages":[{"role":"user","content":"hello"}]}`,
			expectedToolID: nil,
		},
		{
			name:           "single tool result block",
			requestBody:    `{"model":"claude-opus-4-5","max_tokens":1024,"messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_abc","content":"result"}]}]}`,
			expectedToolID: utils.PtrTo("toolu_abc"),
		},
		{
			name:           "multiple tool result blocks returns last",
			requestBody:    `{"model":"claude-opus-4-5","max_tokens":1024,"messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_first","content":"first"},{"type":"text","text":"ignored"},{"type":"tool_result","tool_use_id":"toolu_second","content":"second"}]}]}`,
			expectedToolID: utils.PtrTo("toolu_second"),
		},
		{
			name:           "last message is not a tool result",
			requestBody:    `{"model":"claude-opus-4-5","max_tokens":1024,"messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_first","content":"first"}]},{"role":"user","content":"some text"}]}`,
			expectedToolID: nil,
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			t.Parallel()

			base := &interceptionBase{
				reqPayload: mustMessagesPayload(t, testCase.requestBody),
			}

			require.Equal(t, testCase.expectedToolID, base.CorrelatingToolCallID())
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

	testCases := []struct {
		name                      string
		requestBody               string
		injectedTools             []*mcp.Tool
		expectedToolNames         []string
		expectedCacheControlTypes []string
	}{
		{
			name: "cache control preserved when no tools to inject",
			requestBody: `{"model":"claude-opus-4-5","max_tokens":1024,"messages":[{"role":"user","content":"hello"}],"tools":[` +
				`{"name":"existing_tool","type":"custom","input_schema":{"type":"object","properties":{}},"cache_control":{"type":"ephemeral"}}]}`,

			injectedTools:             nil,
			expectedToolNames:         []string{"existing_tool"},
			expectedCacheControlTypes: []string{string(constant.ValueOf[constant.Ephemeral]())},
		},
		{
			name: "cache control breakpoint is preserved by prepending injected tools",
			requestBody: `{"model":"claude-opus-4-5","max_tokens":1024,"messages":[{"role":"user","content":"hello"}],"tools":[` +
				`{"name":"existing_tool","type":"custom","input_schema":{"type":"object","properties":{}},"cache_control":{"type":"ephemeral"}}]}`,

			injectedTools:             []*mcp.Tool{{ID: "injected_tool", Name: "injected", Description: "Injected tool"}},
			expectedToolNames:         []string{"injected_tool", "existing_tool"},
			expectedCacheControlTypes: []string{"", string(constant.ValueOf[constant.Ephemeral]())},
		},
		{
			name: "cache control breakpoint in non standard location is preserved",
			requestBody: `{"model":"claude-opus-4-5","max_tokens":1024,"messages":[{"role":"user","content":"hello"}],"tools":[` +
				`{"name":"tool_with_cache_1","type":"custom","input_schema":{"type":"object","properties":{}},"cache_control":{"type":"ephemeral"}},` +
				`{"name":"tool_with_cache_2","type":"custom","input_schema":{"type":"object","properties":{}}}]}`,

			injectedTools:             []*mcp.Tool{{ID: "injected_tool", Name: "injected", Description: "Injected tool"}},
			expectedToolNames:         []string{"injected_tool", "tool_with_cache_1", "tool_with_cache_2"},
			expectedCacheControlTypes: []string{"", string(constant.ValueOf[constant.Ephemeral]()), ""},
		},
		{
			name: "no cache control added when none originally set",
			requestBody: `{"model":"claude-opus-4-5","max_tokens":1024,"messages":[{"role":"user","content":"hello"}],"tools":[` +
				`{"name":"existing_tool_no_cache","type":"custom","input_schema":{"type":"object","properties":{}}}]}`,

			injectedTools:             []*mcp.Tool{{ID: "injected_tool", Name: "injected", Description: "Injected tool"}},
			expectedToolNames:         []string{"injected_tool", "existing_tool_no_cache"},
			expectedCacheControlTypes: []string{"", ""},
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			t.Parallel()

			base := &interceptionBase{
				reqPayload: mustMessagesPayload(t, testCase.requestBody),
				mcpProxy:   &mockServerProxier{tools: testCase.injectedTools},
				logger:     slog.Make(),
			}

			base.injectTools()

			toolItems := gjson.GetBytes(base.reqPayload, "tools").Array()
			require.Len(t, toolItems, len(testCase.expectedToolNames))
			for idx := range toolItems {
				require.Equal(t, testCase.expectedToolNames[idx], toolItems[idx].Get("name").String())
				require.Equal(t, testCase.expectedCacheControlTypes[idx], toolItems[idx].Get("cache_control.type").String())
			}
		})
	}
}

func TestInjectTools_ParallelToolCalls(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name                    string
		requestBody             string
		injectedTools           []*mcp.Tool
		expectedToolChoiceType  string
		expectedDisableParallel *bool
		expectedToolCount       int
	}{
		{
			name:                    "does not modify tool choice when no tools to inject",
			requestBody:             `{"model":"claude-opus-4-5","max_tokens":1024,"messages":[{"role":"user","content":"hello"}],"tool_choice":{"type":"auto"}}`,
			injectedTools:           nil,
			expectedToolChoiceType:  string(constant.ValueOf[constant.Auto]()),
			expectedDisableParallel: nil,
			expectedToolCount:       0,
		},
		{
			name:                    "disables parallel tool use for auto tool choice default",
			requestBody:             `{"model":"claude-opus-4-5","max_tokens":1024,"messages":[{"role":"user","content":"hello"}]}`,
			injectedTools:           []*mcp.Tool{{ID: "test_tool", Name: "test", Description: "Test"}},
			expectedToolChoiceType:  string(constant.ValueOf[constant.Auto]()),
			expectedDisableParallel: utils.PtrTo(true),
			expectedToolCount:       1,
		},
		{
			name:                    "disables parallel tool use for explicit auto tool choice",
			requestBody:             `{"model":"claude-opus-4-5","max_tokens":1024,"messages":[{"role":"user","content":"hello"}],"tool_choice":{"type":"auto"}}`,
			injectedTools:           []*mcp.Tool{{ID: "test_tool", Name: "test", Description: "Test"}},
			expectedToolChoiceType:  string(constant.ValueOf[constant.Auto]()),
			expectedDisableParallel: utils.PtrTo(true),
			expectedToolCount:       1,
		},
		{
			name:                    "disables parallel tool use for any tool choice",
			requestBody:             `{"model":"claude-opus-4-5","max_tokens":1024,"messages":[{"role":"user","content":"hello"}],"tool_choice":{"type":"any"}}`,
			injectedTools:           []*mcp.Tool{{ID: "test_tool", Name: "test", Description: "Test"}},
			expectedToolChoiceType:  string(constant.ValueOf[constant.Any]()),
			expectedDisableParallel: utils.PtrTo(true),
			expectedToolCount:       1,
		},
		{
			name:                    "disables parallel tool use for tool choice type",
			requestBody:             `{"model":"claude-opus-4-5","max_tokens":1024,"messages":[{"role":"user","content":"hello"}],"tool_choice":{"type":"tool","name":"specific_tool"}}`,
			injectedTools:           []*mcp.Tool{{ID: "test_tool", Name: "test", Description: "Test"}},
			expectedToolChoiceType:  string(constant.ValueOf[constant.Tool]()),
			expectedDisableParallel: utils.PtrTo(true),
			expectedToolCount:       1,
		},
		{
			name:                    "no op for none tool choice type",
			requestBody:             `{"model":"claude-opus-4-5","max_tokens":1024,"messages":[{"role":"user","content":"hello"}],"tool_choice":{"type":"none"}}`,
			injectedTools:           []*mcp.Tool{{ID: "test_tool", Name: "test", Description: "Test"}},
			expectedToolChoiceType:  string(constant.ValueOf[constant.None]()),
			expectedDisableParallel: nil,
			expectedToolCount:       1,
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			t.Parallel()

			base := &interceptionBase{
				reqPayload: mustMessagesPayload(t, testCase.requestBody),
				mcpProxy:   &mockServerProxier{tools: testCase.injectedTools},
				logger:     slog.Make(),
			}

			base.injectTools()

			require.Len(t, gjson.GetBytes(base.reqPayload, "tools").Array(), testCase.expectedToolCount)

			toolChoice := gjson.GetBytes(base.reqPayload, "tool_choice")
			require.Equal(t, testCase.expectedToolChoiceType, toolChoice.Get("type").String())

			disableParallelResult := toolChoice.Get("disable_parallel_tool_use")
			if testCase.expectedDisableParallel == nil {
				require.False(t, disableParallelResult.Exists())
				return
			}

			require.True(t, disableParallelResult.Exists())
			require.Equal(t, *testCase.expectedDisableParallel, disableParallelResult.Bool())
		})
	}
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
