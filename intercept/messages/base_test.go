package messages

import (
	"context"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/shared/constant"
	"github.com/coder/aibridge/config"
	"github.com/coder/aibridge/mcp"
	mcpgo "github.com/mark3labs/mcp-go/mcp"
	"github.com/stretchr/testify/require"
)

func TestAWSBedrockValidation(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name        string
		cfg         *config.AWSBedrockConfig
		expectError bool
		errorMsg    string
	}{
		{
			name: "valid",
			cfg: &config.AWSBedrockConfig{
				Region:          "us-east-1",
				AccessKey:       "test-key",
				AccessKeySecret: "test-secret",
				Model:           "test-model",
				SmallFastModel:  "test-small-model",
			},
		},
		{
			name: "missing region",
			cfg: &config.AWSBedrockConfig{
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
			cfg: &config.AWSBedrockConfig{
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
			cfg: &config.AWSBedrockConfig{
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
			cfg: &config.AWSBedrockConfig{
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
			cfg: &config.AWSBedrockConfig{
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
			cfg:         &config.AWSBedrockConfig{},
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
			base := &interceptionBase{}
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
