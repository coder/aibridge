package aibridge

import (
	"fmt"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/sony/gobreaker/v2"
)

// CircuitBreakerConfig holds configuration for circuit breakers.
// Fields match gobreaker.Settings for clarity.
type CircuitBreakerConfig struct {
	// MaxRequests is the maximum number of requests allowed in half-open state.
	MaxRequests uint32
	// Interval is the cyclic period of the closed state for clearing internal counts.
	Interval time.Duration
	// Timeout is how long the circuit stays open before transitioning to half-open.
	Timeout time.Duration
	// FailureThreshold is the number of consecutive failures that triggers the circuit to open.
	FailureThreshold uint32
	// IsFailure determines if a status code should count as a failure.
	// If nil, defaults to 429, 503, and 529 (Anthropic overloaded).
	IsFailure func(statusCode int) bool
}

// DefaultCircuitBreakerConfig returns sensible defaults for circuit breaker configuration.
func DefaultCircuitBreakerConfig() CircuitBreakerConfig {
	return CircuitBreakerConfig{
		FailureThreshold: 5,
		Interval:         10 * time.Second,
		Timeout:          30 * time.Second,
		MaxRequests:      3,
		IsFailure:        DefaultIsFailure,
	}
}

// DefaultIsFailure returns true for status codes that typically indicate
// upstream overload: 429 (Too Many Requests), 503 (Service Unavailable),
// and 529 (Anthropic Overloaded).
func DefaultIsFailure(statusCode int) bool {
	switch statusCode {
	case http.StatusTooManyRequests, // 429
		http.StatusServiceUnavailable, // 503
		529:                           // Anthropic "Overloaded"
		return true
	default:
		return false
	}
}

// CircuitBreaker defines the interface for circuit breaker implementations.
type CircuitBreaker interface {
	// Allow returns true if the request should be allowed through.
	Allow() bool
	// RecordSuccess records a successful request.
	RecordSuccess()
	// RecordFailure records a failed request with the given status code.
	RecordFailure(statusCode int)
}

// NoopCircuitBreaker is a circuit breaker that always allows requests through.
// Used when circuit breaker is not configured for a provider.
type NoopCircuitBreaker struct{}

func (NoopCircuitBreaker) Allow() bool                  { return true }
func (NoopCircuitBreaker) RecordSuccess()               {}
func (NoopCircuitBreaker) RecordFailure(statusCode int) {}

// gobreakerCircuitBreaker wraps sony/gobreaker to implement CircuitBreaker.
type gobreakerCircuitBreaker struct {
	cb        *gobreaker.CircuitBreaker[any]
	isFailure func(statusCode int) bool
}

func (g *gobreakerCircuitBreaker) Allow() bool {
	return g.cb.State() != gobreaker.StateOpen
}

func (g *gobreakerCircuitBreaker) RecordSuccess() {
	_, _ = g.cb.Execute(func() (any, error) { return nil, nil })
}

func (g *gobreakerCircuitBreaker) RecordFailure(statusCode int) {
	if g.isFailure(statusCode) {
		_, _ = g.cb.Execute(func() (any, error) {
			return nil, fmt.Errorf("upstream error: %d", statusCode)
		})
	}
}

// CircuitBreakers manages per-endpoint circuit breakers using sony/gobreaker.
// Circuit breakers are keyed by "provider:endpoint" for per-endpoint isolation.
type CircuitBreakers struct {
	breakers sync.Map // map[string]CircuitBreaker
	configs  map[string]CircuitBreakerConfig
	onChange func(name string, from, to gobreaker.State)
}

// NewCircuitBreakers creates a new circuit breaker manager with per-provider configs.
// The configs map is keyed by provider name. Providers not in the map will use
// NoopCircuitBreaker (always allows requests).
func NewCircuitBreakers(configs map[string]CircuitBreakerConfig, onChange func(name string, from, to gobreaker.State)) *CircuitBreakers {
	return &CircuitBreakers{
		configs:  configs,
		onChange: onChange,
	}
}

// getConfig returns the config for a provider, or nil if not configured.
func (c *CircuitBreakers) getConfig(provider string) *CircuitBreakerConfig {
	if c.configs == nil {
		return nil
	}
	cfg, ok := c.configs[provider]
	if !ok {
		return nil
	}
	return &cfg
}

// Get returns the circuit breaker for a provider/endpoint.
// Returns NoopCircuitBreaker if the provider is not configured.
func (c *CircuitBreakers) Get(provider, endpoint string) CircuitBreaker {
	cfg := c.getConfig(provider)
	if cfg == nil {
		return NoopCircuitBreaker{}
	}

	key := provider + ":" + endpoint
	if v, ok := c.breakers.Load(key); ok {
		return v.(CircuitBreaker)
	}

	isFailure := cfg.IsFailure
	if isFailure == nil {
		isFailure = DefaultIsFailure
	}

	settings := gobreaker.Settings{
		Name:        key,
		MaxRequests: cfg.MaxRequests,
		Interval:    cfg.Interval,
		Timeout:     cfg.Timeout,
		ReadyToTrip: func(counts gobreaker.Counts) bool {
			return counts.ConsecutiveFailures >= cfg.FailureThreshold
		},
		OnStateChange: func(name string, from, to gobreaker.State) {
			if c.onChange != nil {
				c.onChange(name, from, to)
			}
		},
	}

	cb := &gobreakerCircuitBreaker{
		cb:        gobreaker.NewCircuitBreaker[any](settings),
		isFailure: isFailure,
	}
	actual, _ := c.breakers.LoadOrStore(key, cb)
	return actual.(CircuitBreaker)
}

// statusCapturingWriter wraps http.ResponseWriter to capture the status code.
// It also implements http.Flusher to support streaming responses.
type statusCapturingWriter struct {
	http.ResponseWriter
	statusCode    int
	headerWritten bool
}

func (w *statusCapturingWriter) WriteHeader(code int) {
	if !w.headerWritten {
		w.statusCode = code
		w.headerWritten = true
	}
	w.ResponseWriter.WriteHeader(code)
}

func (w *statusCapturingWriter) Write(b []byte) (int, error) {
	if !w.headerWritten {
		w.statusCode = http.StatusOK
		w.headerWritten = true
	}
	return w.ResponseWriter.Write(b)
}

func (w *statusCapturingWriter) Flush() {
	if f, ok := w.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}

// Unwrap returns the underlying ResponseWriter for interface checks.
func (w *statusCapturingWriter) Unwrap() http.ResponseWriter {
	return w.ResponseWriter
}

// CircuitBreakerMiddleware returns middleware that wraps handlers with circuit breaker protection.
// It captures the response status code to determine success/failure without provider-specific logic.
// If the provider is not configured, uses NoopCircuitBreaker (always allows requests).
func CircuitBreakerMiddleware(cbs *CircuitBreakers, metrics *Metrics, provider string) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			endpoint := strings.TrimPrefix(r.URL.Path, fmt.Sprintf("/%s", provider))
			cb := cbs.Get(provider, endpoint)

			// Check if circuit is open
			if !cb.Allow() {
				if metrics != nil {
					metrics.CircuitBreakerRejects.WithLabelValues(provider, endpoint).Inc()
				}
				http.Error(w, "circuit breaker is open", http.StatusServiceUnavailable)
				return
			}

			// Wrap response writer to capture status code
			sw := &statusCapturingWriter{ResponseWriter: w, statusCode: http.StatusOK}
			next.ServeHTTP(sw, r)

			// Record result - NoopCircuitBreaker methods are no-ops
			if sw.statusCode >= 400 {
				cb.RecordFailure(sw.statusCode)
			} else {
				cb.RecordSuccess()
			}
		})
	}
}

// stateToGaugeValue converts gobreaker.State to a gauge value.
// closed=0, half-open=0.5, open=1
func stateToGaugeValue(s gobreaker.State) float64 {
	switch s {
	case gobreaker.StateClosed:
		return 0
	case gobreaker.StateHalfOpen:
		return 0.5
	case gobreaker.StateOpen:
		return 1
	default:
		return 0
	}
}
