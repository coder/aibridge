package mcp

import (
	"context"
	"fmt"
	"regexp"
	"slices"
	"strings"

	"cdr.dev/slog"
	"github.com/coder/aibridge/aibtrace"
	"github.com/mark3labs/mcp-go/client"
	"github.com/mark3labs/mcp-go/client/transport"
	"github.com/mark3labs/mcp-go/mcp"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
	"golang.org/x/exp/maps"
)

var _ ServerProxier = &StreamableHTTPServerProxy{}

type StreamableHTTPServerProxy struct {
	serverName string
	serverURL  string
	client     *client.Client
	logger     slog.Logger
	tracer     trace.Tracer
	tools      map[string]*Tool

	allowlistPattern, denylistPattern *regexp.Regexp
}

func NewStreamableHTTPServerProxy(logger slog.Logger, tracer trace.Tracer, serverName, serverURL string, headers map[string]string, allowlist, denylist *regexp.Regexp) (*StreamableHTTPServerProxy, error) {
	var opts []transport.StreamableHTTPCOption
	if headers != nil {
		opts = append(opts, transport.WithHTTPHeaders(headers))
	}

	mcpClient, err := client.NewStreamableHttpClient(serverURL, opts...)
	if err != nil {
		return nil, fmt.Errorf("create streamable http client: %w", err)
	}

	return &StreamableHTTPServerProxy{
		serverName:       serverName,
		serverURL:        serverURL,
		client:           mcpClient,
		logger:           logger,
		tracer:           tracer,
		allowlistPattern: allowlist,
		denylistPattern:  denylist,
	}, nil
}

func (p *StreamableHTTPServerProxy) Name() string {
	return p.serverName
}

func (p *StreamableHTTPServerProxy) Init(ctx context.Context) (outErr error) {
	ctx, span := p.tracer.Start(ctx, "StreamableHTTPServerProxy.Init", trace.WithAttributes(p.traceAttributes()...))
	defer aibtrace.EndSpanErr(span, &outErr)

	if err := p.client.Start(ctx); err != nil {
		return fmt.Errorf("start client: %w", err)
	}

	version := mcp.LATEST_PROTOCOL_VERSION
	initReq := mcp.InitializeRequest{
		Params: mcp.InitializeParams{
			ProtocolVersion: version,
			ClientInfo:      GetClientInfo(),
		},
	}

	result, err := p.client.Initialize(ctx, initReq)
	if err != nil {
		return fmt.Errorf("init MCP client: %w", err)
	}

	if !slices.Contains(mcp.ValidProtocolVersions, result.ProtocolVersion) {
		if err := p.client.Close(); err != nil {
			p.logger.Debug(ctx, "failed to close MCP client on unsuccessful version negotiation", slog.Error(err))
		}
		return fmt.Errorf("MCP version negotiation failed; requested %q, accepts %q, received %q", version, strings.Join(mcp.ValidProtocolVersions, ","), result.ProtocolVersion)
	}

	p.logger.Debug(ctx, "MCP client initialized", slog.F("name", result.ServerInfo.Name), slog.F("server_version", result.ServerInfo.Version))

	tools, err := p.fetchTools(ctx)
	if err != nil {
		return fmt.Errorf("fetch tools: %w", err)
	}

	// Only include allowed tools.
	p.tools = FilterAllowedTools(p.logger.Named("tool-filterer"), tools, p.allowlistPattern, p.denylistPattern)
	return nil
}

func (p *StreamableHTTPServerProxy) ListTools() []*Tool {
	tools := maps.Values(p.tools)
	slices.SortStableFunc(tools, func(a, b *Tool) int {
		return strings.Compare(a.ID, b.ID)
	})
	return tools
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

	return p.client.CallTool(ctx, mcp.CallToolRequest{
		Params: mcp.CallToolParams{
			Name:      tool.Name,
			Arguments: input,
		},
	})
}

func (p *StreamableHTTPServerProxy) fetchTools(ctx context.Context) (_ map[string]*Tool, outErr error) {
	ctx, span := p.tracer.Start(ctx, "StreamableHTTPServerProxy.Init.fetchTools", trace.WithAttributes(p.traceAttributes()...))
	defer aibtrace.EndSpanErr(span, &outErr)

	tools, err := p.client.ListTools(ctx, mcp.ListToolsRequest{})
	if err != nil {
		return nil, fmt.Errorf("list MCP tools: %w", err)
	}

	out := make(map[string]*Tool, len(tools.Tools))
	for _, tool := range tools.Tools {
		encodedID := EncodeToolID(p.serverName, tool.Name)
		out[encodedID] = &Tool{
			Client:      p.client,
			ID:          encodedID,
			Name:        tool.Name,
			ServerName:  p.serverName,
			ServerURL:   p.serverURL,
			Description: tool.Description,
			Params:      tool.InputSchema.Properties,
			Required:    tool.InputSchema.Required,
			Logger:      p.logger,
		}
	}
	span.SetAttributes(append(p.traceAttributes(), attribute.Int(aibtrace.MCPToolCount, len(out)))...)
	return out, nil
}

func (p *StreamableHTTPServerProxy) Shutdown(ctx context.Context) error {
	if p.client == nil {
		return nil
	}

	// NOTE: as of v0.38.0 the lib doesn't allow an outside context to be passed in;
	// it has an internal timeout of 5s, though.
	return p.client.Close()
}

func (p *StreamableHTTPServerProxy) traceAttributes() []attribute.KeyValue {
	return []attribute.KeyValue{
		attribute.String(aibtrace.MCPProxyName, p.Name()),
		attribute.String(aibtrace.MCPServerName, p.serverName),
		attribute.String(aibtrace.MCPServerURL, p.serverURL),
	}
}
