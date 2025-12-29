package aibridge

import (
	"errors"
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

// ProviderCircuitBreakers manages per-endpoint circuit breakers for a single provider.
type ProviderCircuitBreakers struct {
	provider string
	config   CircuitBreakerConfig
	breakers sync.Map // endpoint -> *gobreaker.CircuitBreaker[struct{}]
	onChange func(endpoint string, from, to gobreaker.State)
}

// NewProviderCircuitBreakers creates circuit breakers for a single provider.
// Returns nil if config is nil (no circuit breaker protection).
func NewProviderCircuitBreakers(provider string, config *CircuitBreakerConfig, onChange func(endpoint string, from, to gobreaker.State)) *ProviderCircuitBreakers {
	if config == nil {
		return nil
	}
	if config.IsFailure == nil {
		config.IsFailure = DefaultIsFailure
	}
	return &ProviderCircuitBreakers{
		provider: provider,
		config:   *config,
		onChange: onChange,
	}
}

// Get returns the circuit breaker for an endpoint, creating it if needed.
func (p *ProviderCircuitBreakers) Get(endpoint string) *gobreaker.CircuitBreaker[struct{}] {
	if v, ok := p.breakers.Load(endpoint); ok {
		return v.(*gobreaker.CircuitBreaker[struct{}])
	}

	settings := gobreaker.Settings{
		Name:        p.provider + ":" + endpoint,
		MaxRequests: p.config.MaxRequests,
		Interval:    p.config.Interval,
		Timeout:     p.config.Timeout,
		ReadyToTrip: func(counts gobreaker.Counts) bool {
			return counts.ConsecutiveFailures >= p.config.FailureThreshold
		},
		OnStateChange: func(_ string, from, to gobreaker.State) {
			if p.onChange != nil {
				p.onChange(endpoint, from, to)
			}
		},
	}

	cb := gobreaker.NewCircuitBreaker[struct{}](settings)
	actual, _ := p.breakers.LoadOrStore(endpoint, cb)
	return actual.(*gobreaker.CircuitBreaker[struct{}])
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
// If cbs is nil, requests pass through without circuit breaker protection.
func CircuitBreakerMiddleware(cbs *ProviderCircuitBreakers, metrics *Metrics) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		// No circuit breaker configured - pass through
		if cbs == nil {
			return next
		}

		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			endpoint := strings.TrimPrefix(r.URL.Path, "/"+cbs.provider)
			cb := cbs.Get(endpoint)

			// Wrap response writer to capture status code
			sw := &statusCapturingWriter{ResponseWriter: w, statusCode: http.StatusOK}

			_, err := cb.Execute(func() (struct{}, error) {
				next.ServeHTTP(sw, r)
				if cbs.config.IsFailure(sw.statusCode) {
					return struct{}{}, fmt.Errorf("upstream error: %d", sw.statusCode)
				}
				return struct{}{}, nil
			})

			if errors.Is(err, gobreaker.ErrOpenState) || errors.Is(err, gobreaker.ErrTooManyRequests) {
				if metrics != nil {
					metrics.CircuitBreakerRejects.WithLabelValues(cbs.provider, endpoint).Inc()
				}
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusServiceUnavailable)
				w.Write([]byte(`{"type":"error","error":{"type":"circuit_breaker_open","message":"circuit breaker is open"}}`))
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
