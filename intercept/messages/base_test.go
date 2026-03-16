package messages

import (
	"context"
	"encoding/json"
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
		name     string
		messages []anthropic.MessageParam
		expected *string
	}{
		{
			name:     "no messages",
			messages: nil,
			expected: nil,
		},
		{
			name: "last message has no tool_result blocks",
			messages: []anthropic.MessageParam{
				anthropic.NewUserMessage(anthropic.NewTextBlock("hello")),
			},
			expected: nil,
		},
		{
			name: "single tool_result block",
			messages: []anthropic.MessageParam{
				anthropic.NewUserMessage(
					anthropic.ContentBlockParamUnion{
						OfToolResult: &anthropic.ToolResultBlockParam{
							ToolUseID: "toolu_abc",
							Content: []anthropic.ToolResultBlockParamContentUnion{
								{OfText: &anthropic.TextBlockParam{Text: "result"}},
							},
						},
					},
				),
			},
			expected: utils.PtrTo("toolu_abc"),
		},
		{
			name: "multiple tool_result blocks returns last",
			messages: []anthropic.MessageParam{
				anthropic.NewUserMessage(
					anthropic.ContentBlockParamUnion{
						OfToolResult: &anthropic.ToolResultBlockParam{
							ToolUseID: "toolu_first",
							Content: []anthropic.ToolResultBlockParamContentUnion{
								{OfText: &anthropic.TextBlockParam{Text: "first"}},
							},
						},
					},
					anthropic.NewTextBlock("some text"),
					anthropic.ContentBlockParamUnion{
						OfToolResult: &anthropic.ToolResultBlockParam{
							ToolUseID: "toolu_second",
							Content: []anthropic.ToolResultBlockParamContentUnion{
								{OfText: &anthropic.TextBlockParam{Text: "second"}},
							},
						},
					},
				),
			},
			expected: utils.PtrTo("toolu_second"),
		},
		{
			name: "last message is not a tool result",
			messages: []anthropic.MessageParam{
				anthropic.NewUserMessage(
					anthropic.ContentBlockParamUnion{
						OfToolResult: &anthropic.ToolResultBlockParam{
							ToolUseID: "toolu_first",
							Content: []anthropic.ToolResultBlockParamContentUnion{
								{OfText: &anthropic.TextBlockParam{Text: "first"}},
							},
						},
					}),
				anthropic.NewUserMessage(anthropic.NewTextBlock("some text")),
			},
			expected: nil,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			base := &interceptionBase{
				req: &MessageNewParamsWrapper{
					MessageNewParams: anthropic.MessageNewParams{
						Messages: tc.messages,
					},
				},
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
			req: &MessageNewParamsWrapper{
				MessageNewParams: anthropic.MessageNewParams{
					Tools: []anthropic.ToolUnionParam{
						{
							OfTool: &anthropic.ToolParam{
								Name: "existing_tool",
								CacheControl: anthropic.CacheControlEphemeralParam{
									Type: constant.ValueOf[constant.Ephemeral](),
								},
							},
						},
					},
				},
			},
			mcpProxy: &mockServerProxier{tools: nil},
		}

		i.injectTools()

		// Cache control should remain untouched since no tools were injected.
		require.Len(t, i.req.Tools, 1)
		require.Equal(t, constant.ValueOf[constant.Ephemeral](), i.req.Tools[0].OfTool.CacheControl.Type)
	})

	t.Run("cache control breakpoint is preserved by prepending injected tools", func(t *testing.T) {
		t.Parallel()

		// Request has existing tool with cache control.
		i := &interceptionBase{
			req: &MessageNewParamsWrapper{
				MessageNewParams: anthropic.MessageNewParams{
					Tools: []anthropic.ToolUnionParam{
						{
							OfTool: &anthropic.ToolParam{
								Name: "existing_tool",
								CacheControl: anthropic.CacheControlEphemeralParam{
									Type: constant.ValueOf[constant.Ephemeral](),
								},
							},
						},
					},
				},
			},
			mcpProxy: &mockServerProxier{
				tools: []*mcp.Tool{
					{ID: "injected_tool", Name: "injected", Description: "Injected tool"},
				},
			},
		}

		i.injectTools()

		require.Len(t, i.req.Tools, 2)
		// Injected tools are prepended.
		require.Equal(t, "injected_tool", i.req.Tools[0].OfTool.Name)
		require.Zero(t, i.req.Tools[0].OfTool.CacheControl)
		// Original tool's cache control should be preserved at the end.
		require.Equal(t, "existing_tool", i.req.Tools[1].OfTool.Name)
		require.Equal(t, constant.ValueOf[constant.Ephemeral](), i.req.Tools[1].OfTool.CacheControl.Type)
	})

	// The cache breakpoint SHOULD be on the final tool, but may not be; we must preserve that intention.
	t.Run("cache control breakpoint in non-standard location is preserved", func(t *testing.T) {
		t.Parallel()

		// Request has multiple tools with cache control breakpoints.
		i := &interceptionBase{
			req: &MessageNewParamsWrapper{
				MessageNewParams: anthropic.MessageNewParams{
					Tools: []anthropic.ToolUnionParam{
						{
							OfTool: &anthropic.ToolParam{
								Name: "tool_with_cache_1",
								CacheControl: anthropic.CacheControlEphemeralParam{
									Type: constant.ValueOf[constant.Ephemeral](),
								},
							},
						},
						{
							OfTool: &anthropic.ToolParam{
								Name: "tool_with_cache_2",
							},
						},
					},
				},
			},
			mcpProxy: &mockServerProxier{
				tools: []*mcp.Tool{
					{ID: "injected_tool", Name: "injected", Description: "Injected tool"},
				},
			},
		}

		i.injectTools()

		require.Len(t, i.req.Tools, 3)
		// Injected tool is prepended without cache control.
		require.Equal(t, "injected_tool", i.req.Tools[0].OfTool.Name)
		require.Zero(t, i.req.Tools[0].OfTool.CacheControl)
		// Both original tools' cache controls should remain.
		require.Equal(t, "tool_with_cache_1", i.req.Tools[1].OfTool.Name)
		require.Equal(t, constant.ValueOf[constant.Ephemeral](), i.req.Tools[1].OfTool.CacheControl.Type)
		require.Equal(t, "tool_with_cache_2", i.req.Tools[2].OfTool.Name)
		require.Zero(t, i.req.Tools[2].OfTool.CacheControl)
	})

	t.Run("no cache control added when none originally set", func(t *testing.T) {
		t.Parallel()

		// Request has tools but none with cache control.
		i := &interceptionBase{
			req: &MessageNewParamsWrapper{
				MessageNewParams: anthropic.MessageNewParams{
					Tools: []anthropic.ToolUnionParam{
						{
							OfTool: &anthropic.ToolParam{
								Name: "existing_tool_no_cache",
							},
						},
					},
				},
			},
			mcpProxy: &mockServerProxier{
				tools: []*mcp.Tool{
					{ID: "injected_tool", Name: "injected", Description: "Injected tool"},
				},
			},
		}

		i.injectTools()

		require.Len(t, i.req.Tools, 2)
		// Injected tool is prepended without cache control.
		require.Equal(t, "injected_tool", i.req.Tools[0].OfTool.Name)
		require.Zero(t, i.req.Tools[0].OfTool.CacheControl)
		// Original tool remains at the end without cache control.
		require.Equal(t, "existing_tool_no_cache", i.req.Tools[1].OfTool.Name)
		require.Zero(t, i.req.Tools[1].OfTool.CacheControl)
	})
}

func TestInjectTools_ParallelToolCalls(t *testing.T) {
	t.Parallel()

	t.Run("does not modify tool choice when no tools to inject", func(t *testing.T) {
		t.Parallel()

		i := &interceptionBase{
			req: &MessageNewParamsWrapper{
				MessageNewParams: anthropic.MessageNewParams{
					ToolChoice: anthropic.ToolChoiceUnionParam{
						OfAuto: &anthropic.ToolChoiceAutoParam{
							Type: constant.ValueOf[constant.Auto](),
						},
					},
				},
			},
			mcpProxy: &mockServerProxier{tools: nil}, // No tools to inject.
		}

		i.injectTools()

		// Tool choice should remain unchanged - DisableParallelToolUse should not be set.
		require.NotNil(t, i.req.ToolChoice.OfAuto)
		require.False(t, i.req.ToolChoice.OfAuto.DisableParallelToolUse.Valid())
	})

	t.Run("disables parallel tool use for auto tool choice (default)", func(t *testing.T) {
		t.Parallel()

		i := &interceptionBase{
			req: &MessageNewParamsWrapper{
				MessageNewParams: anthropic.MessageNewParams{
					// No tool choice set (default).
				},
			},
			mcpProxy: &mockServerProxier{
				tools: []*mcp.Tool{{ID: "test_tool", Name: "test", Description: "Test"}},
			},
		}

		i.injectTools()

		require.NotNil(t, i.req.ToolChoice.OfAuto)
		require.True(t, i.req.ToolChoice.OfAuto.DisableParallelToolUse.Valid())
		require.True(t, i.req.ToolChoice.OfAuto.DisableParallelToolUse.Value)
	})

	t.Run("disables parallel tool use for explicit auto tool choice", func(t *testing.T) {
		t.Parallel()

		i := &interceptionBase{
			req: &MessageNewParamsWrapper{
				MessageNewParams: anthropic.MessageNewParams{
					ToolChoice: anthropic.ToolChoiceUnionParam{
						OfAuto: &anthropic.ToolChoiceAutoParam{
							Type: constant.ValueOf[constant.Auto](),
						},
					},
				},
			},
			mcpProxy: &mockServerProxier{
				tools: []*mcp.Tool{{ID: "test_tool", Name: "test", Description: "Test"}},
			},
		}

		i.injectTools()

		require.NotNil(t, i.req.ToolChoice.OfAuto)
		require.True(t, i.req.ToolChoice.OfAuto.DisableParallelToolUse.Valid())
		require.True(t, i.req.ToolChoice.OfAuto.DisableParallelToolUse.Value)
	})

	t.Run("disables parallel tool use for any tool choice", func(t *testing.T) {
		t.Parallel()

		i := &interceptionBase{
			req: &MessageNewParamsWrapper{
				MessageNewParams: anthropic.MessageNewParams{
					ToolChoice: anthropic.ToolChoiceUnionParam{
						OfAny: &anthropic.ToolChoiceAnyParam{
							Type: constant.ValueOf[constant.Any](),
						},
					},
				},
			},
			mcpProxy: &mockServerProxier{
				tools: []*mcp.Tool{{ID: "test_tool", Name: "test", Description: "Test"}},
			},
		}

		i.injectTools()

		require.NotNil(t, i.req.ToolChoice.OfAny)
		require.True(t, i.req.ToolChoice.OfAny.DisableParallelToolUse.Valid())
		require.True(t, i.req.ToolChoice.OfAny.DisableParallelToolUse.Value)
	})

	t.Run("disables parallel tool use for tool choice type", func(t *testing.T) {
		t.Parallel()

		i := &interceptionBase{
			req: &MessageNewParamsWrapper{
				MessageNewParams: anthropic.MessageNewParams{
					ToolChoice: anthropic.ToolChoiceUnionParam{
						OfTool: &anthropic.ToolChoiceToolParam{
							Type: constant.ValueOf[constant.Tool](),
							Name: "specific_tool",
						},
					},
				},
			},
			mcpProxy: &mockServerProxier{
				tools: []*mcp.Tool{{ID: "test_tool", Name: "test", Description: "Test"}},
			},
		}

		i.injectTools()

		require.NotNil(t, i.req.ToolChoice.OfTool)
		require.True(t, i.req.ToolChoice.OfTool.DisableParallelToolUse.Valid())
		require.True(t, i.req.ToolChoice.OfTool.DisableParallelToolUse.Value)
	})

	t.Run("no-op for none tool choice type", func(t *testing.T) {
		t.Parallel()

		i := &interceptionBase{
			req: &MessageNewParamsWrapper{
				MessageNewParams: anthropic.MessageNewParams{
					ToolChoice: anthropic.ToolChoiceUnionParam{
						OfNone: &anthropic.ToolChoiceNoneParam{
							Type: constant.ValueOf[constant.None](),
						},
					},
				},
			},
			mcpProxy: &mockServerProxier{
				tools: []*mcp.Tool{{ID: "test_tool", Name: "test", Description: "Test"}},
			},
		}

		i.injectTools()

		// Tools are still injected.
		require.Len(t, i.req.Tools, 1)
		// But no parallel tool use modification for "none" type.
		require.Nil(t, i.req.ToolChoice.OfAuto)
		require.Nil(t, i.req.ToolChoice.OfAny)
		require.Nil(t, i.req.ToolChoice.OfTool)
		require.NotNil(t, i.req.ToolChoice.OfNone)
	})
}

func TestBedrockModelSupportsAdaptiveThinking(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		model    string
		expected bool
	}{
		{"opus 4.6 with version", "anthropic.claude-opus-4-6-v1", true},
		{"sonnet 4.6", "anthropic.claude-sonnet-4-6", true},
		{"us prefix opus 4.6", "us.anthropic.claude-opus-4-6-v1", true},
		{"opus 4.6 dot notation", "claude-opus-4.6", true},
		{"sonnet 4.6 dot notation", "claude-sonnet-4.6", true},
		{"sonnet 4.5", "anthropic.claude-sonnet-4-5-20250929-v1:0", false},
		{"opus 4.5", "anthropic.claude-opus-4-5-20251101-v1:0", false},
		{"haiku 4.5", "anthropic.claude-haiku-4-5-20251001-v1:0", false},
		{"sonnet 3.7", "anthropic.claude-3-7-sonnet-20250219-v1:0", false},
		{"custom model name", "my-custom-model", false},
		{"empty", "", false},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			require.Equal(t, tc.expected, bedrockModelSupportsAdaptiveThinking(tc.model))
		})
	}
}

func TestConvertAdaptiveThinkingForBedrock(t *testing.T) {
	t.Parallel()

	newBaseWithBedrock := func(model string, payload map[string]any) *interceptionBase {
		raw, err := json.Marshal(payload)
		require.NoError(t, err)
		return &interceptionBase{
			req: &MessageNewParamsWrapper{
				MessageNewParams: anthropic.MessageNewParams{
					Model: anthropic.Model(model),
				},
			},
			payload:    raw,
			bedrockCfg: &config.AWSBedrock{Model: model, SmallFastModel: "haiku"},
			logger:     slogtest(t),
		}
	}

	t.Run("converts adaptive to enabled for non-4.6 model", func(t *testing.T) {
		t.Parallel()

		base := newBaseWithBedrock("anthropic.claude-sonnet-4-5-20250929-v1:0", map[string]any{
			"model":      "claude-sonnet-4-5",
			"max_tokens": 16000,
			"thinking":   map[string]string{"type": "adaptive"},
			"messages":   []any{},
		})

		base.convertAdaptiveThinkingForBedrock()

		require.Equal(t, "enabled", gjson.GetBytes(base.payload, "thinking.type").Str)
		// 80% of 16000 = 12800
		require.Equal(t, int64(12800), gjson.GetBytes(base.payload, "thinking.budget_tokens").Int())
	})

	t.Run("uses default when max_tokens is absent", func(t *testing.T) {
		t.Parallel()

		base := newBaseWithBedrock("anthropic.claude-sonnet-4-5-20250929-v1:0", map[string]any{
			"model":    "claude-sonnet-4-5",
			"thinking": map[string]string{"type": "adaptive"},
			"messages": []any{},
		})

		base.convertAdaptiveThinkingForBedrock()

		require.Equal(t, "enabled", gjson.GetBytes(base.payload, "thinking.type").Str)
		require.Equal(t, int64(defaultAdaptiveFallbackBudgetTokens), gjson.GetBytes(base.payload, "thinking.budget_tokens").Int())
	})

	t.Run("clamps to minimum when max_tokens is small", func(t *testing.T) {
		t.Parallel()

		// max_tokens=1200: 80% = 960, below min 1024, so clamped to 1024.
		// 1024 < 1200, so budget_tokens = 1024.
		base := newBaseWithBedrock("anthropic.claude-sonnet-4-5-20250929-v1:0", map[string]any{
			"model":      "claude-sonnet-4-5",
			"max_tokens": 1200,
			"thinking":   map[string]string{"type": "adaptive"},
			"messages":   []any{},
		})

		base.convertAdaptiveThinkingForBedrock()

		require.Equal(t, "enabled", gjson.GetBytes(base.payload, "thinking.type").Str)
		require.Equal(t, int64(1024), gjson.GetBytes(base.payload, "thinking.budget_tokens").Int())
	})

	t.Run("falls back to half when max_tokens is very small", func(t *testing.T) {
		t.Parallel()

		base := newBaseWithBedrock("anthropic.claude-sonnet-4-5-20250929-v1:0", map[string]any{
			"model":      "claude-sonnet-4-5",
			"max_tokens": 1024,
			"thinking":   map[string]string{"type": "adaptive"},
			"messages":   []any{},
		})

		base.convertAdaptiveThinkingForBedrock()

		require.Equal(t, "enabled", gjson.GetBytes(base.payload, "thinking.type").Str)
		// 80% of 1024 = 819, clamped to min 1024, but 1024 >= 1024 (max_tokens), so half: 512.
		require.Equal(t, int64(512), gjson.GetBytes(base.payload, "thinking.budget_tokens").Int())
	})

	t.Run("preserves adaptive for opus 4.6", func(t *testing.T) {
		t.Parallel()

		base := newBaseWithBedrock("anthropic.claude-opus-4-6-v1", map[string]any{
			"model":    "claude-opus-4-6",
			"thinking": map[string]string{"type": "adaptive"},
			"messages": []any{},
		})

		base.convertAdaptiveThinkingForBedrock()

		require.Equal(t, "adaptive", gjson.GetBytes(base.payload, "thinking.type").Str)
		require.False(t, gjson.GetBytes(base.payload, "thinking.budget_tokens").Exists())
	})

	t.Run("preserves adaptive for sonnet 4.6", func(t *testing.T) {
		t.Parallel()

		base := newBaseWithBedrock("anthropic.claude-sonnet-4-6", map[string]any{
			"model":    "claude-sonnet-4-6",
			"thinking": map[string]string{"type": "adaptive"},
			"messages": []any{},
		})

		base.convertAdaptiveThinkingForBedrock()

		require.Equal(t, "adaptive", gjson.GetBytes(base.payload, "thinking.type").Str)
	})

	t.Run("no-op when thinking type is enabled", func(t *testing.T) {
		t.Parallel()

		base := newBaseWithBedrock("anthropic.claude-sonnet-4-5-20250929-v1:0", map[string]any{
			"model":    "claude-sonnet-4-5",
			"thinking": map[string]any{"type": "enabled", "budget_tokens": 5000},
			"messages": []any{},
		})

		base.convertAdaptiveThinkingForBedrock()

		require.Equal(t, "enabled", gjson.GetBytes(base.payload, "thinking.type").Str)
		require.Equal(t, int64(5000), gjson.GetBytes(base.payload, "thinking.budget_tokens").Int())
	})

	t.Run("no-op when thinking is absent", func(t *testing.T) {
		t.Parallel()

		base := newBaseWithBedrock("anthropic.claude-sonnet-4-5-20250929-v1:0", map[string]any{
			"model":    "claude-sonnet-4-5",
			"messages": []any{},
		})

		base.convertAdaptiveThinkingForBedrock()

		require.False(t, gjson.GetBytes(base.payload, "thinking").Exists())
	})

	t.Run("no-op when thinking type is disabled", func(t *testing.T) {
		t.Parallel()

		base := newBaseWithBedrock("anthropic.claude-sonnet-4-5-20250929-v1:0", map[string]any{
			"model":    "claude-sonnet-4-5",
			"thinking": map[string]string{"type": "disabled"},
			"messages": []any{},
		})

		base.convertAdaptiveThinkingForBedrock()

		require.Equal(t, "disabled", gjson.GetBytes(base.payload, "thinking.type").Str)
	})
}

// slogtest returns a no-op logger for tests.
func slogtest(t *testing.T) slog.Logger {
	t.Helper()
	return slog.Logger{}
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
