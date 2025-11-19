package aibtrace

import (
	"context"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"
)

type traceInterceptionAttrsContextKey struct{}

const (
	TraceInterceptionIDKey = "interception_id"
	TraceUserIDKey         = "user_id"
	TraceProviderKey       = "provider"
	TraceModelKey          = "model"
	TraceStreamingKey      = "streaming"
	TraceIsBedrockKey      = "aws_bedrock"
	TraceMCPToolName       = "mcp_tool_name"
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

func EndSpanErr(span trace.Span, err error) {
	if err != nil {
		span.SetStatus(codes.Error, err.Error())
	}
	span.End()
}
