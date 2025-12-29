package aibridge

import (
	"context"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/shared/constant"
	"github.com/coder/aibridge/mcp"
	mcpgo "github.com/mark3labs/mcp-go/mcp"
	"github.com/stretchr/testify/require"
)

func TestInjectTools_CacheBreakpoints(t *testing.T) {
	t.Parallel()

	t.Run("cache control preserved when no tools to inject", func(t *testing.T) {
		t.Parallel()

		// Request has existing tool with cache control, but no tools to inject.
		i := &AnthropicMessagesInterceptionBase{
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
		i := &AnthropicMessagesInterceptionBase{
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
		i := &AnthropicMessagesInterceptionBase{
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
		i := &AnthropicMessagesInterceptionBase{
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

		i := &AnthropicMessagesInterceptionBase{
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

		i := &AnthropicMessagesInterceptionBase{
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

		i := &AnthropicMessagesInterceptionBase{
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

		i := &AnthropicMessagesInterceptionBase{
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

		i := &AnthropicMessagesInterceptionBase{
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

		i := &AnthropicMessagesInterceptionBase{
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
