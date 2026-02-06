package provider

import (
	"errors"
	"net/http"

	"github.com/coder/aibridge/config"
	"github.com/coder/aibridge/intercept"
	"go.opentelemetry.io/otel/trace"
)

var UnknownRoute = errors.New("unknown route")

// Provider describes an AI provider client's behaviour.
// Provider clients are responsible for interacting with upstream AI providers.
type Provider interface {
	// Name returns the provider's name.
	Name() string
	// BaseURL defines the base URL endpoint for this provider's API.
	BaseURL() string

	// CreateInterceptor starts a new [Interceptor] which is responsible for intercepting requests,
	// communicating with the upstream provider and formulating a response to be sent to the requesting client.
	CreateInterceptor(http.ResponseWriter, *http.Request, trace.Tracer) (intercept.Interceptor, error)

	// RoutePrefix returns a prefix on which the provider's bridged and passthroguh routes will be registered.
	// Must be unique across providers to avoid conflicts.
	RoutePrefix() string

	// BridgedRoutes returns a slice of [http.ServeMux]-compatible routes which will have special handling.
	// See https://pkg.go.dev/net/http#hdr-Patterns-ServeMux.
	BridgedRoutes() []string
	// PassthroughRoutes returns a slice of whitelisted [http.ServeMux]-compatible* routes which are
	// not currently intercepted and must be handled by the upstream directly.
	//
	// * only path routes can be specified, not ones containing HTTP methods. (i.e. GET /route).
	// By default, these passthrough routes will accept any HTTP method.
	PassthroughRoutes() []string

	// AuthHeader returns the name of the header which the provider expects to find its authentication
	// token in.
	AuthHeader() string
	// InjectAuthHeader allows [Provider]s to set its authentication header.
	InjectAuthHeader(*http.Header)

	// CircuitBreakerConfig returns the circuit breaker configuration for the provider.
	CircuitBreakerConfig() *config.CircuitBreaker
}
