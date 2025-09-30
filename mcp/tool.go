package mcp

import (
	"context"
	"errors"
	"regexp"
	"strings"

	"cdr.dev/slog"
	"github.com/modelcontextprotocol/go-sdk/mcp"
)

const (
	injectedToolPrefix    = "bmcp" // "bridged MCP"
	injectedToolDelimiter = "_"
)

// ToolCaller is the narrowest interface which describes the behaviour required from [mcp.Client],
// which will normally be passed into [Tool] for interaction with an MCP server.
// TODO: don't expose github.com/modelcontextprotocol/go-sdk outside this package.
type ToolCaller interface {
	CallTool(ctx context.Context, params *mcp.CallToolParams) (*mcp.CallToolResult, error)
}

type Tool struct {
	Client ToolCaller

	ID          string
	Name        string
	ServerName  string
	ServerURL   string
	Description string
	Params      map[string]any
	Required    []string
}

func (t *Tool) Call(ctx context.Context, input any) (*mcp.CallToolResult, error) {
	if t == nil {
		return nil, errors.New("nil tool!")
	}
	if t.Client == nil {
		return nil, errors.New("nil client!")
	}

	return t.Client.CallTool(ctx, &mcp.CallToolParams{
		Name:      t.Name,
		Arguments: input,
	})
}

// EncodeToolID namespaces the given tool name with a prefix to identify tools injected by this library.
// Claude Code, for example, prefixes the tools it includes from defined MCP servers with the "mcp__" prefix.
// We have to namespace the tools we inject to prevent clashes.
//
// We stick to 5 prefix chars ("bmcp_") like "mcp__" since names can only be up to 64 chars:
//
// See:
// - https://community.openai.com/t/function-call-description-max-length/529902
// - https://github.com/anthropics/claude-code/issues/2326
func EncodeToolID(server, tool string) string {
	var sb strings.Builder
	sb.WriteString(injectedToolPrefix)
	sb.WriteString(injectedToolDelimiter)
	sb.WriteString(server)
	sb.WriteString(injectedToolDelimiter)
	sb.WriteString(tool)
	return sb.String()
}

// FilterAllowedTools filters tools based on the given allow/denylists.
// Filtering acts on tool names, and uses tool IDs for tracking.
// The denylist supersedes the allowlist in the case of any conflicts.
// If an allowlist is provided, tools must match it to be allowed.
// If only a denylist is provided, tools are allowed unless explicitly denied.
func FilterAllowedTools(logger slog.Logger, tools map[string]*Tool, allowlist *regexp.Regexp, denylist *regexp.Regexp) map[string]*Tool {
	if len(tools) == 0 {
		return tools
	}

	if allowlist == nil && denylist == nil {
		return tools
	}

	allowed := make(map[string]*Tool, len(tools))
	for id, tool := range tools {
		if tool == nil {
			continue
		}

		// Check denylist first since it can override allowlist.
		if denylist != nil && denylist.MatchString(tool.Name) {
			// Log conflict if also in allowlist.
			if allowlist != nil && allowlist.MatchString(tool.Name) {
				logger.Warn(context.Background(), "tool filtering conflict; marking tool disallowed", slog.F("name", tool.Name))
			}
			continue // Not allowed.
		}

		// Check allowlist if present.
		if allowlist != nil {
			if !allowlist.MatchString(tool.Name) {
				continue // Not allowed.
			}
		}

		// Tool is allowed.
		allowed[id] = tool
	}

	return allowed
}
