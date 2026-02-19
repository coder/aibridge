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
	// LastToolUseID returns the ID of the last tool use result to be submitted, if present.
	// If multiple tool use results are present, we only need the first one.
	// Both Anthropic's /v1/messages and OpenAI's /v1/responses require that ALL tool results
	// are submitted for tool choices returned by the model.
	// Therefore we only need to pick one, and it doesn't matter which one, since all tool calls
	// will be associated with the previous interception.
	LastToolUseID() string
}
