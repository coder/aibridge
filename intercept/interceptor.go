package intercept

import (
	"net/http"

	"cdr.dev/slog"
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
}
