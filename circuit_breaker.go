package aibridge

import (
	"net/http"
	"sync"
	"time"
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

// CircuitBreakerConfig holds configuration for a circuit breaker.
type CircuitBreakerConfig struct {
	// Enabled controls whether the circuit breaker is active.
	// If false, all requests pass through regardless of failures.
	Enabled bool
	// FailureThreshold is the number of failures within the window that triggers the circuit to open.
	FailureThreshold int64
	// Window is the time window for counting failures.
	Window time.Duration
	// Cooldown is how long the circuit stays open before transitioning to half-open.
	Cooldown time.Duration
	// HalfOpenMaxRequests is the maximum number of requests allowed in half-open state
	// before deciding whether to close or re-open the circuit.
	HalfOpenMaxRequests int64
}

// DefaultCircuitBreakerConfig returns sensible defaults for circuit breaker configuration.
func DefaultCircuitBreakerConfig() CircuitBreakerConfig {
	return CircuitBreakerConfig{
		Enabled:             false, // Disabled by default for backward compatibility
		FailureThreshold:    5,
		Window:              10 * time.Second,
		Cooldown:            30 * time.Second,
		HalfOpenMaxRequests: 3,
	}
}

// CircuitBreaker implements the circuit breaker pattern to protect against
// upstream service failures. It tracks failures from upstream providers
// (like rate limit errors) and temporarily blocks requests when the
// failure threshold is exceeded.
type CircuitBreaker struct {
	mu sync.RWMutex

	// Current state
	state    CircuitState
	failures int64     // Failure count in current window
	windowStart time.Time // Start of current failure counting window
	openedAt time.Time // When circuit transitioned to open
	
	// Half-open state tracking
	halfOpenSuccesses int64
	halfOpenFailures  int64

	// Configuration
	config CircuitBreakerConfig

	// Provider name for logging/metrics
	provider string

	// Optional metrics callback
	onStateChange func(provider string, from, to CircuitState)
}

// NewCircuitBreaker creates a new circuit breaker for the given provider.
func NewCircuitBreaker(provider string, config CircuitBreakerConfig) *CircuitBreaker {
	return &CircuitBreaker{
		state:       CircuitClosed,
		windowStart: time.Now(),
		config:      config,
		provider:    provider,
	}
}

// SetStateChangeCallback sets a callback that is invoked when the circuit state changes.
// This is useful for metrics and logging.
func (cb *CircuitBreaker) SetStateChangeCallback(fn func(provider string, from, to CircuitState)) {
	cb.mu.Lock()
	defer cb.mu.Unlock()
	cb.onStateChange = fn
}

// Allow checks if a request should be allowed through.
// Returns true if the request can proceed, false if it should be rejected.
func (cb *CircuitBreaker) Allow() bool {
	if !cb.config.Enabled {
		return true
	}

	cb.mu.Lock()
	defer cb.mu.Unlock()

	now := time.Now()

	switch cb.state {
	case CircuitClosed:
		return true

	case CircuitOpen:
		// Check if cooldown period has elapsed
		if now.Sub(cb.openedAt) >= cb.config.Cooldown {
			cb.transitionTo(CircuitHalfOpen)
			return true
		}
		return false

	case CircuitHalfOpen:
		// Allow limited requests in half-open state
		totalHalfOpenRequests := cb.halfOpenSuccesses + cb.halfOpenFailures
		return totalHalfOpenRequests < cb.config.HalfOpenMaxRequests
	}

	return true
}

// RecordSuccess records a successful request.
// This is called after a request completes successfully.
func (cb *CircuitBreaker) RecordSuccess() {
	if !cb.config.Enabled {
		return
	}

	cb.mu.Lock()
	defer cb.mu.Unlock()

	switch cb.state {
	case CircuitHalfOpen:
		cb.halfOpenSuccesses++
		// If we've had enough successes in half-open, close the circuit
		if cb.halfOpenSuccesses >= cb.config.HalfOpenMaxRequests {
			cb.transitionTo(CircuitClosed)
		}
	case CircuitClosed:
		// Reset failure count on success (sliding window behavior)
		// This helps prevent false positives from old failures
		cb.maybeResetWindow()
	}
}

// RecordFailure records a failed request.
// statusCode is the HTTP status code from the upstream response.
// Returns true if this failure caused the circuit to trip open.
func (cb *CircuitBreaker) RecordFailure(statusCode int) bool {
	if !cb.config.Enabled {
		return false
	}

	// Only count specific error codes as circuit-breaker failures
	if !isCircuitBreakerFailure(statusCode) {
		return false
	}

	cb.mu.Lock()
	defer cb.mu.Unlock()

	switch cb.state {
	case CircuitClosed:
		cb.maybeResetWindow()
		cb.failures++
		if cb.failures >= cb.config.FailureThreshold {
			cb.transitionTo(CircuitOpen)
			return true
		}

	case CircuitHalfOpen:
		cb.halfOpenFailures++
		// Any failure in half-open state re-opens the circuit
		cb.transitionTo(CircuitOpen)
		return true
	}

	return false
}

// State returns the current state of the circuit breaker.
func (cb *CircuitBreaker) State() CircuitState {
	cb.mu.RLock()
	defer cb.mu.RUnlock()
	return cb.state
}

// Provider returns the provider name this circuit breaker is for.
func (cb *CircuitBreaker) Provider() string {
	return cb.provider
}

// Failures returns the current failure count.
func (cb *CircuitBreaker) Failures() int64 {
	cb.mu.RLock()
	defer cb.mu.RUnlock()
	return cb.failures
}

// transitionTo changes the circuit state. Must be called with lock held.
func (cb *CircuitBreaker) transitionTo(newState CircuitState) {
	oldState := cb.state
	if oldState == newState {
		return
	}

	cb.state = newState
	now := time.Now()

	switch newState {
	case CircuitOpen:
		cb.openedAt = now
	case CircuitHalfOpen:
		cb.halfOpenSuccesses = 0
		cb.halfOpenFailures = 0
	case CircuitClosed:
		cb.failures = 0
		cb.windowStart = now
	}

	if cb.onStateChange != nil {
		// Call callback without holding lock to avoid deadlocks
		callback := cb.onStateChange
		go callback(cb.provider, oldState, newState)
	}
}

// maybeResetWindow resets the failure count if the window has elapsed.
// Must be called with lock held.
func (cb *CircuitBreaker) maybeResetWindow() {
	now := time.Now()
	if now.Sub(cb.windowStart) >= cb.config.Window {
		cb.failures = 0
		cb.windowStart = now
	}
}

// isCircuitBreakerFailure returns true if the given HTTP status code
// should count as a failure for circuit breaker purposes.
// We specifically track rate limiting and overload errors from upstream.
func isCircuitBreakerFailure(statusCode int) bool {
	switch statusCode {
	case http.StatusTooManyRequests: // 429 - Rate limited
		return true
	case http.StatusServiceUnavailable: // 503 - Service unavailable
		return true
	case 529: // Anthropic-specific "Overloaded" error
		return true
	default:
		return false
	}
}

// CircuitBreakerManager manages circuit breakers for multiple providers.
type CircuitBreakerManager struct {
	mu       sync.RWMutex
	breakers map[string]*CircuitBreaker
	config   CircuitBreakerConfig

	// Metrics callbacks
	onStateChange func(provider string, from, to CircuitState)
}

// NewCircuitBreakerManager creates a new manager with the given configuration.
func NewCircuitBreakerManager(config CircuitBreakerConfig) *CircuitBreakerManager {
	return &CircuitBreakerManager{
		breakers: make(map[string]*CircuitBreaker),
		config:   config,
	}
}

// SetStateChangeCallback sets the callback for state changes on all circuit breakers.
func (m *CircuitBreakerManager) SetStateChangeCallback(fn func(provider string, from, to CircuitState)) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.onStateChange = fn
	
	// Update existing breakers
	for _, cb := range m.breakers {
		cb.SetStateChangeCallback(fn)
	}
}

// GetOrCreate returns the circuit breaker for the given provider,
// creating one if it doesn't exist.
func (m *CircuitBreakerManager) GetOrCreate(provider string) *CircuitBreaker {
	m.mu.RLock()
	if cb, ok := m.breakers[provider]; ok {
		m.mu.RUnlock()
		return cb
	}
	m.mu.RUnlock()

	m.mu.Lock()
	defer m.mu.Unlock()

	// Double-check after acquiring write lock
	if cb, ok := m.breakers[provider]; ok {
		return cb
	}

	cb := NewCircuitBreaker(provider, m.config)
	if m.onStateChange != nil {
		cb.SetStateChangeCallback(m.onStateChange)
	}
	m.breakers[provider] = cb
	return cb
}

// Get returns the circuit breaker for the given provider, or nil if not found.
func (m *CircuitBreakerManager) Get(provider string) *CircuitBreaker {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.breakers[provider]
}

// AllStates returns the current state of all circuit breakers.
func (m *CircuitBreakerManager) AllStates() map[string]CircuitState {
	m.mu.RLock()
	defer m.mu.RUnlock()

	states := make(map[string]CircuitState, len(m.breakers))
	for provider, cb := range m.breakers {
		states[provider] = cb.State()
	}
	return states
}

// Config returns the configuration used by this manager.
func (m *CircuitBreakerManager) Config() CircuitBreakerConfig {
	return m.config
}
