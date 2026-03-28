package intercept

import (
	"net/http"

	"cdr.dev/slog/v3"
	"github.com/coder/aibridge/mcp"
	"github.com/coder/aibridge/recorder"
	"github.com/google/uuid"
	"go.opentelemetry.io/otel/attribute"
)

// Interceptor describes a (potentially) stateful interaction with an AI provider.
type Interceptor interface {
	// ID returns the unique identifier for this interception.
	ID() uuid.UUID
	// Setup injects some required dependencies. This MUST be called before using the interceptor
	// to process requests.
	Setup(logger slog.Logger, recorder recorder.Recorder, mcpProxy mcp.ServerProxier)
	// Model returns the model in use for this [Interceptor].
	Model() string
	// ProcessRequest handles the HTTP request.
	ProcessRequest(w http.ResponseWriter, r *http.Request) error
	// Specifies whether an interceptor handles streaming or not.
	Streaming() bool
	// TraceAttributes returns tracing attributes for this [Interceptor]
	TraceAttributes(*http.Request) []attribute.KeyValue
	// ProviderName returns the provider name for this interception.
	ProviderName() string
	// CorrelatingToolCallID returns the ID of a tool call result submitted
	// in the request, if present. This is used to correlate the current
	// interception back to the previous interception that issued those tool
	// calls. If multiple tool use results are present, we use the last one
	// (most recent). Both Anthropic's /v1/messages and OpenAI's /v1/responses
	// require that ALL tool results are submitted for tool choices returned
	// by the model, so any single tool call ID is sufficient to identify the
	// parent interception.
	CorrelatingToolCallID() *string
}

// ResolvedUpstream represents the resolved upstream destination for a request.
// Providers resolve this per-request, allowing a single provider to route to
// different upstreams based on request context.
type ResolvedUpstream struct {
	// Name identifies the upstream provider domain.
	Name string
	// URL is the URL of the upstream provider.
	URL string
}
