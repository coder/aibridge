package testutil

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"cdr.dev/slog"
	"github.com/coder/aibridge/mcp"
	mcplib "github.com/mark3labs/mcp-go/mcp"
	"github.com/mark3labs/mcp-go/server"
	"go.opentelemetry.io/otel/trace"
)

const (
	// ToolCoderListWorkspaces matches the tool name used throughout the current tests.
	ToolCoderListWorkspaces = "coder_list_workspaces"
)

func DefaultCoderToolNames() []string {
	return []string{
		ToolCoderListWorkspaces,
		"coder_list_templates",
		"coder_template_version_parameters",
		"coder_get_authenticated_user",
		"coder_create_workspace_build",
	}
}

type MCPToolResultFunc func(ctx context.Context, request mcplib.CallToolRequest) (*mcplib.CallToolResult, error)

type MCPServer struct {
	*httptest.Server

	callsMu sync.Mutex
	calls   map[string][]any
}

type MCPServerOption func(*mcpServerConfig)

type mcpServerConfig struct {
	toolResultFn MCPToolResultFunc
}

func WithMCPToolResult(fn MCPToolResultFunc) MCPServerOption {
	return func(cfg *mcpServerConfig) {
		cfg.toolResultFn = fn
	}
}

func NewMCPServer(t testing.TB, toolNames []string, opts ...MCPServerOption) *MCPServer {
	t.Helper()

	cfg := mcpServerConfig{
		toolResultFn: func(ctx context.Context, request mcplib.CallToolRequest) (*mcplib.CallToolResult, error) {
			return mcplib.NewToolResultText("mock"), nil
		},
	}
	for _, opt := range opts {
		if opt != nil {
			opt(&cfg)
		}
	}

	s := &MCPServer{calls: make(map[string][]any)}

	mcpSrv := server.NewMCPServer(
		"Mock MCP server",
		"1.0.0",
		server.WithToolCapabilities(true),
	)

	for _, name := range toolNames {
		name := name // capture
		tool := mcplib.NewTool(name, mcplib.WithDescription(fmt.Sprintf("Mock of the %s tool", name)))
		mcpSrv.AddTool(tool, func(ctx context.Context, request mcplib.CallToolRequest) (*mcplib.CallToolResult, error) {
			s.addCall(request.Params.Name, request.Params.Arguments)
			return cfg.toolResultFn(ctx, request)
		})
	}

	h := server.NewStreamableHTTPServer(mcpSrv)
	s.Server = httptest.NewServer(h)
	t.Cleanup(s.Server.Close)

	return s
}

func (s *MCPServer) addCall(tool string, args any) {
	s.callsMu.Lock()
	defer s.callsMu.Unlock()

	s.calls[tool] = append(s.calls[tool], args)
}

func (s *MCPServer) CallsByTool(name string) []any {
	s.callsMu.Lock()
	defer s.callsMu.Unlock()

	calls := s.calls[name]
	out := make([]any, len(calls))
	copy(out, calls)
	return out
}

func (s *MCPServer) Proxiers(t testing.TB, serverName string, logger slog.Logger, tracer trace.Tracer) map[string]mcp.ServerProxier {
	t.Helper()

	proxy, err := mcp.NewStreamableHTTPServerProxy(serverName, s.URL, nil, nil, nil, logger, tracer)
	mustNoError(t, err, "create MCP proxy")
	return map[string]mcp.ServerProxier{proxy.Name(): proxy}
}

func (s *MCPServer) Handler() http.Handler {
	if s.Server == nil {
		return nil
	}
	return s.Server.Config.Handler
}
