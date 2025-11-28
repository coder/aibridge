package aibtrace

import (
	"context"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"
)

type (
	traceInterceptionAttrsContextKey  struct{}
	traceRequestBridgeAttrsContextKey struct{}
)

const (
	// trace attribute key constants
	RequestPath = "request_path"

	InterceptionID = "interception_id"
	UserID         = "user_id"
	Provider       = "provider"
	Model          = "model"
	Streaming      = "streaming"
	IsBedrock      = "aws_bedrock"

	PassthroughURL    = "passthrough_url"
	PassthroughMethod = "passthrough_method"

	MCPInput      = "mcp_input"
	MCPProxyName  = "mcp_proxy_name"
	MCPToolName   = "mcp_tool_name"
	MCPServerName = "mcp_server_name"
	MCPServerURL  = "mcp_server_url"
	MCPToolCount  = "mcp_tool_count"

	APIKeyID = "api_key_id"
)

func WithInterceptionAttributesInContext(ctx context.Context, traceAttrs []attribute.KeyValue) context.Context {
	return context.WithValue(ctx, traceInterceptionAttrsContextKey{}, traceAttrs)
}

func InterceptionAttributesFromContext(ctx context.Context) []attribute.KeyValue {
	attrs, ok := ctx.Value(traceInterceptionAttrsContextKey{}).([]attribute.KeyValue)
	if !ok {
		return nil
	}

	return attrs
}

func WithRequestBridgeAttributesInContext(ctx context.Context, traceAttrs []attribute.KeyValue) context.Context {
	return context.WithValue(ctx, traceRequestBridgeAttrsContextKey{}, traceAttrs)
}

func RequestBridgeAttributesFromContext(ctx context.Context) []attribute.KeyValue {
	attrs, ok := ctx.Value(traceRequestBridgeAttrsContextKey{}).([]attribute.KeyValue)
	if !ok {
		return nil
	}

	return attrs
}

func EndSpanErr(span trace.Span, err *error) {
	if span == nil {
		return
	}

	if err != nil && *err != nil {
		span.SetStatus(codes.Error, (*err).Error())
	}
	span.End()
}
