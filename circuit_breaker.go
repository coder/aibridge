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

// CircuitBreakers manages per-endpoint circuit breakers using sony/gobreaker.
// Circuit breakers are keyed by "provider:endpoint" for per-endpoint isolation.
type CircuitBreakers struct {
	breakers sync.Map // map[string]*gobreaker.CircuitBreaker[any]
	configs  map[string]CircuitBreakerConfig
	onChange func(name string, from, to gobreaker.State)
}

// NewCircuitBreakers creates a new circuit breaker manager with per-provider configs.
// The configs map is keyed by provider name. Providers not in the map will not have
// circuit breaker protection.
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

// getOrCreate returns the circuit breaker for a provider/endpoint, creating if needed.
// Returns nil if the provider is not configured.
func (c *CircuitBreakers) getOrCreate(provider, endpoint string) *gobreaker.CircuitBreaker[any] {
	cfg := c.getConfig(provider)
	if cfg == nil {
		return nil
	}

	key := provider + ":" + endpoint
	if v, ok := c.breakers.Load(key); ok {
		return v.(*gobreaker.CircuitBreaker[any])
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

	cb := gobreaker.NewCircuitBreaker[any](settings)
	actual, _ := c.breakers.LoadOrStore(key, cb)
	return actual.(*gobreaker.CircuitBreaker[any])
}

// statusCapturingWriter wraps http.ResponseWriter to capture the status code.
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

// CircuitBreakerMiddleware returns middleware that wraps handlers with circuit breaker protection.
// It captures the response status code to determine success/failure without provider-specific logic.
func CircuitBreakerMiddleware(cbs *CircuitBreakers, metrics *Metrics, provider string) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		cfg := cbs.getConfig(provider)
		if cfg == nil {
			// No config for this provider, pass through
			return next
		}

		isFailure := cfg.IsFailure
		if isFailure == nil {
			isFailure = DefaultIsFailure
		}

		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			endpoint := strings.TrimPrefix(r.URL.Path, fmt.Sprintf("/%s", provider))

			// Check if circuit is open
			cb := cbs.getOrCreate(provider, endpoint)
			if cb != nil && cb.State() == gobreaker.StateOpen {
				if metrics != nil {
					metrics.CircuitBreakerRejects.WithLabelValues(provider, endpoint).Inc()
				}
				http.Error(w, "circuit breaker is open", http.StatusServiceUnavailable)
				return
			}

			// Wrap response writer to capture status code
			sw := &statusCapturingWriter{ResponseWriter: w, statusCode: http.StatusOK}
			next.ServeHTTP(sw, r)

			// Record result
			if cb != nil {
				if isFailure(sw.statusCode) {
					_, _ = cb.Execute(func() (any, error) {
						return nil, fmt.Errorf("upstream error: %d", sw.statusCode)
					})
				} else {
					_, _ = cb.Execute(func() (any, error) { return nil, nil })
				}
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
