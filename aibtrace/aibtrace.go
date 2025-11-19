package aibtrace

import (
	"context"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"
)

type traceInterceptionAttrsContextKey struct{}

const (
	// trace attribute key constants
	InterceptionID    = "interception_id"
	UserID            = "user_id"
	Provider          = "provider"
	Model             = "model"
	Streaming         = "streaming"
	IsBedrock         = "aws_bedrock"
	MCPToolName       = "mcp_tool_name"
	PassthroughURL    = "passthrough_url"
	PassthroughMethod = "passthrough_method"
)

func WithTraceInterceptionAttributesInContext(ctx context.Context, traceAttrs []attribute.KeyValue) context.Context {
	return context.WithValue(ctx, traceInterceptionAttrsContextKey{}, traceAttrs)
}

func TraceInterceptionAttributesFromContext(ctx context.Context) []attribute.KeyValue {
	attrs, ok := ctx.Value(traceInterceptionAttrsContextKey{}).([]attribute.KeyValue)
	if !ok {
		return nil
	}

	return attrs
}

func EndSpanErr(span trace.Span, err *error) {
	if err != nil && *err != nil {
		span.SetStatus(codes.Error, (*err).Error())
	}
	span.End()
}
