package circuitbreaker

import (
	"errors"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"sync"

	"cdr.dev/slog/v3"
	"github.com/coder/aibridge/config"
	"github.com/coder/aibridge/metrics"
	"github.com/sony/gobreaker/v2"
)

// DefaultIsFailure returns true for standard HTTP status codes that typically
// indicate upstream overload.
func DefaultIsFailure(statusCode int) bool {
	switch statusCode {
	case http.StatusTooManyRequests, // 429
		http.StatusServiceUnavailable, // 503
		http.StatusGatewayTimeout:     // 504
		return true
	default:
		return false
	}
}

// ProviderCircuitBreakers manages per-endpoint circuit breakers for a single provider.
type ProviderCircuitBreakers struct {
	provider string
	config   config.CircuitBreaker
	breakers sync.Map // endpoint -> *gobreaker.CircuitBreaker[struct{}]
	onChange func(endpoint string, from, to gobreaker.State)
}

// NewProviderCircuitBreakers creates circuit breakers for a single provider.
// Returns nil if cfg is nil (no circuit breaker protection).
func NewProviderCircuitBreakers(provider string, cfg *config.CircuitBreaker, onChange func(endpoint string, from, to gobreaker.State)) *ProviderCircuitBreakers {
	if cfg == nil {
		return nil
	}
	return &ProviderCircuitBreakers{
		provider: provider,
		config:   *cfg,
		onChange: onChange,
	}
}

// isFailure checks if the status code should count as a failure.
// Falls back to DefaultIsFailure if no custom function is configured.
func (p *ProviderCircuitBreakers) isFailure(statusCode int) bool {
	if p.config.IsFailure != nil {
		return p.config.IsFailure(statusCode)
	}
	return DefaultIsFailure(statusCode)
}

// openErrorResponse returns the error response body when the circuit is open.
func (p *ProviderCircuitBreakers) openErrorResponse() []byte {
	if p.config.OpenErrorResponse != nil {
		return p.config.OpenErrorResponse()
	}
	return []byte(`{"error":"circuit breaker is open"}`)
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

// Middleware returns middleware that wraps handlers with circuit breaker protection.
// It captures the response status code to determine success/failure without provider-specific logic.
// If cbs is nil, requests pass through without circuit breaker protection.
func Middleware(cbs *ProviderCircuitBreakers, m *metrics.Metrics, logger slog.Logger) func(http.Handler) http.Handler {
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
				if cbs.isFailure(sw.statusCode) {
					return struct{}{}, fmt.Errorf("upstream error: %d", sw.statusCode)
				}
				return struct{}{}, nil
			})

			if errors.Is(err, gobreaker.ErrOpenState) || errors.Is(err, gobreaker.ErrTooManyRequests) {
				if m != nil {
					m.CircuitBreakerRejects.WithLabelValues(cbs.provider, endpoint).Inc()
				}
				w.Header().Set("Content-Type", "application/json")
				w.Header().Set("Retry-After", strconv.FormatInt(int64(cbs.config.Timeout.Seconds()), 10))
				w.WriteHeader(http.StatusServiceUnavailable)
				w.Write(cbs.openErrorResponse())
			} else if err != nil {
				logger.Warn(r.Context(), "unexpected circuit breaker error", slog.Error(err))
			}
		})
	}
}

// StateToGaugeValue converts gobreaker.State to a gauge value.
// closed=0, half-open=0.5, open=1
func StateToGaugeValue(s gobreaker.State) float64 {
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
