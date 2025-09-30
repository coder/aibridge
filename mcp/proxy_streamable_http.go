package mcp

import (
	"context"
	"fmt"
	"net/http"
	"regexp"

	"cdr.dev/slog"
	"github.com/modelcontextprotocol/go-sdk/mcp"
	"golang.org/x/exp/maps"
)

var _ ServerProxier = &StreamableHTTPServerProxy{}

type StreamableHTTPServerProxy struct {
	serverName string
	serverURL  string
	headers    map[string]string
	session    *mcp.ClientSession
	logger     slog.Logger
	tools      map[string]*Tool

	allowlistPattern, denylistPattern *regexp.Regexp
}

func NewStreamableHTTPServerProxy(logger slog.Logger, serverName, serverURL string, headers map[string]string, allowlist, denylist *regexp.Regexp) (*StreamableHTTPServerProxy, error) {
	return &StreamableHTTPServerProxy{
		serverName:       serverName,
		serverURL:        serverURL,
		headers:          headers,
		logger:           logger,
		allowlistPattern: allowlist,
		denylistPattern:  denylist,
	}, nil
}

func (p *StreamableHTTPServerProxy) Name() string {
	return p.serverName
}

func (p *StreamableHTTPServerProxy) Init(ctx context.Context) error {
	httpClient := newHTTPClientWithHeaders(p.headers)
	transport := &mcp.StreamableClientTransport{
		Endpoint:   p.serverURL,
		HTTPClient: httpClient,
	}

	impl := GetClientInfo()
	client := mcp.NewClient(&impl, nil)
	sess, err := client.Connect(ctx, transport, nil)
	if err != nil {
		return fmt.Errorf("connect MCP client: %w", err)
	}
	p.session = sess

	p.logger.Debug(ctx, "MCP client initialized")

	tools, err := p.fetchTools(ctx)
	if err != nil {
		return fmt.Errorf("fetch tools: %w", err)
	}

	// Only include allowed tools.
	p.tools = FilterAllowedTools(p.logger.Named("tool-filterer"), tools, p.allowlistPattern, p.denylistPattern)
	return nil
}

func (p *StreamableHTTPServerProxy) ListTools() []*Tool {
	return maps.Values(p.tools)
}

func (p *StreamableHTTPServerProxy) GetTool(name string) *Tool {
	if p.tools == nil {
		return nil
	}

	t, ok := p.tools[name]
	if !ok {
		return nil
	}
	return t
}

func (p *StreamableHTTPServerProxy) CallTool(ctx context.Context, name string, input any) (*mcp.CallToolResult, error) {
	tool := p.GetTool(name)
	if tool == nil {
		return nil, fmt.Errorf("%q tool not known", name)
	}
	if p.session == nil {
		return nil, fmt.Errorf("session not initialized")
	}

	return p.session.CallTool(ctx, &mcp.CallToolParams{
		Name:      tool.Name,
		Arguments: input,
	})
}

func (p *StreamableHTTPServerProxy) fetchTools(ctx context.Context) (map[string]*Tool, error) {
	if p.session == nil {
		return nil, fmt.Errorf("session not initialized")
	}

	res, err := p.session.ListTools(ctx, &mcp.ListToolsParams{})
	if err != nil {
		return nil, fmt.Errorf("list MCP tools: %w", err)
	}

	out := make(map[string]*Tool, len(res.Tools))
	for _, t := range res.Tools {
		encodedID := EncodeToolID(p.serverName, t.Name)

		var params map[string]any
		var required []string
		if schemaMap, ok := t.InputSchema.(map[string]any); ok {
			if props, ok := schemaMap["properties"].(map[string]any); ok {
				params = props
			}
			if req, ok := schemaMap["required"].([]any); ok {
				for _, v := range req {
					if s, ok := v.(string); ok {
						required = append(required, s)
					}
				}
			}
		}

		out[encodedID] = &Tool{
			Client:      p.session,
			ID:          encodedID,
			Name:        t.Name,
			ServerName:  p.serverName,
			ServerURL:   p.serverURL,
			Description: t.Description,
			Params:      params,
			Required:    required,
		}
	}
	return out, nil
}

func (p *StreamableHTTPServerProxy) Shutdown(ctx context.Context) error {
	if p.session == nil {
		return nil
	}
	return p.session.Close()
}

// newHTTPClientWithHeaders returns an http.Client that injects headers (including Accept for streamable) on each request.
func newHTTPClientWithHeaders(headers map[string]string) *http.Client {
	transport := http.DefaultTransport
	return &http.Client{Transport: roundTripperFunc(func(req *http.Request) (*http.Response, error) {
		for k, v := range headers {
			req.Header.Set(k, v)
		}
		// Ensure Accept header supports streamable http per go-sdk issues.
		if req.Header.Get("Accept") == "" {
			req.Header.Set("Accept", "text/event-stream,application/json")
		}
		return transport.RoundTrip(req)
	})}
}

type roundTripperFunc func(*http.Request) (*http.Response, error)

func (f roundTripperFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }
