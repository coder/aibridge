package aibridge

import (
	"fmt"
	"net/http"
	"sync"
	"time"

	"github.com/sony/gobreaker/v2"
)

// CircuitState represents the current state of a circuit breaker.
type CircuitState int

const (
	// CircuitClosed is the normal state - all requests pass through.
	CircuitClosed CircuitState = iota
	// CircuitOpen is the tripped state - requests are rejected immediately.
	CircuitOpen
	// CircuitHalfOpen is the testing state - limited requests pass through.
	CircuitHalfOpen
)

func (s CircuitState) String() string {
	switch s {
	case CircuitClosed:
		return "closed"
	case CircuitOpen:
		return "open"
	case CircuitHalfOpen:
		return "half-open"
	default:
		return "unknown"
	}
}

// toCircuitState converts gobreaker.State to our CircuitState.
func toCircuitState(s gobreaker.State) CircuitState {
	switch s {
	case gobreaker.StateClosed:
		return CircuitClosed
	case gobreaker.StateOpen:
		return CircuitOpen
	case gobreaker.StateHalfOpen:
		return CircuitHalfOpen
	default:
		return CircuitClosed
	}
}

// CircuitBreakerConfig holds configuration for circuit breakers.
// Fields match gobreaker.Settings for clarity.
type CircuitBreakerConfig struct {
	// Enabled controls whether circuit breakers are active.
	Enabled bool
	// MaxRequests is the maximum number of requests allowed in half-open state.
	MaxRequests uint32
	// Interval is the cyclic period of the closed state for clearing internal counts.
	Interval time.Duration
	// Timeout is how long the circuit stays open before transitioning to half-open.
	Timeout time.Duration
	// FailureThreshold is the number of consecutive failures that triggers the circuit to open.
	FailureThreshold uint32
}

// DefaultCircuitBreakerConfig returns sensible defaults for circuit breaker configuration.
func DefaultCircuitBreakerConfig() CircuitBreakerConfig {
	return CircuitBreakerConfig{
		Enabled:          false, // Disabled by default for backward compatibility
		FailureThreshold: 5,
		Interval:         10 * time.Second,
		Timeout:          30 * time.Second,
		MaxRequests:      3,
	}
}

// isCircuitBreakerFailure returns true if the given HTTP status code
// should count as a failure for circuit breaker purposes.
func isCircuitBreakerFailure(statusCode int) bool {
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
	config   CircuitBreakerConfig
	onChange func(name string, from, to CircuitState)
}

// NewCircuitBreakers creates a new circuit breaker manager.
func NewCircuitBreakers(config CircuitBreakerConfig, onChange func(name string, from, to CircuitState)) *CircuitBreakers {
	return &CircuitBreakers{
		config:   config,
		onChange: onChange,
	}
}

// Allow checks if a request to provider/endpoint should be allowed.
func (c *CircuitBreakers) Allow(provider, endpoint string) bool {
	if !c.config.Enabled {
		return true
	}
	cb := c.getOrCreate(provider, endpoint)
	return cb.State() != gobreaker.StateOpen
}

// RecordSuccess records a successful request.
func (c *CircuitBreakers) RecordSuccess(provider, endpoint string) {
	if !c.config.Enabled {
		return
	}
	cb := c.getOrCreate(provider, endpoint)
	_, _ = cb.Execute(func() (any, error) { return nil, nil })
}

// RecordFailure records a failed request. Returns true if this caused the circuit to open.
func (c *CircuitBreakers) RecordFailure(provider, endpoint string, statusCode int) bool {
	if !c.config.Enabled || !isCircuitBreakerFailure(statusCode) {
		return false
	}
	cb := c.getOrCreate(provider, endpoint)
	before := cb.State()
	_, _ = cb.Execute(func() (any, error) {
		return nil, fmt.Errorf("upstream error: %d", statusCode)
	})
	return before != gobreaker.StateOpen && cb.State() == gobreaker.StateOpen
}

// State returns the current state for a provider/endpoint.
func (c *CircuitBreakers) State(provider, endpoint string) CircuitState {
	if !c.config.Enabled {
		return CircuitClosed
	}
	cb := c.getOrCreate(provider, endpoint)
	return toCircuitState(cb.State())
}

func (c *CircuitBreakers) getOrCreate(provider, endpoint string) *gobreaker.CircuitBreaker[any] {
	key := provider + ":" + endpoint
	if v, ok := c.breakers.Load(key); ok {
		return v.(*gobreaker.CircuitBreaker[any])
	}

	settings := gobreaker.Settings{
		Name:        key,
		MaxRequests: c.config.MaxRequests,
		Interval:    c.config.Interval,
		Timeout:     c.config.Timeout,
		ReadyToTrip: func(counts gobreaker.Counts) bool {
			return counts.ConsecutiveFailures >= c.config.FailureThreshold
		},
		OnStateChange: func(name string, from, to gobreaker.State) {
			if c.onChange != nil {
				c.onChange(name, toCircuitState(from), toCircuitState(to))
			}
		},
	}

	cb := gobreaker.NewCircuitBreaker[any](settings)
	actual, _ := c.breakers.LoadOrStore(key, cb)
	return actual.(*gobreaker.CircuitBreaker[any])
}
