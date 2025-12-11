package aibridge

import (
	"net/http"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestCircuitBreaker_DefaultConfig(t *testing.T) {
	t.Parallel()

	cfg := DefaultCircuitBreakerConfig()
	assert.False(t, cfg.Enabled, "should be disabled by default")
	assert.Equal(t, int64(5), cfg.FailureThreshold)
	assert.Equal(t, 10*time.Second, cfg.Window)
	assert.Equal(t, 30*time.Second, cfg.Cooldown)
	assert.Equal(t, int64(3), cfg.HalfOpenMaxRequests)
}

func TestCircuitBreaker_DisabledByDefault(t *testing.T) {
	t.Parallel()

	cb := NewCircuitBreaker("test", DefaultCircuitBreakerConfig())

	// Should always allow when disabled
	assert.True(t, cb.Allow())

	// Recording failures should not affect state when disabled
	for i := 0; i < 100; i++ {
		cb.RecordFailure(http.StatusTooManyRequests)
	}
	assert.True(t, cb.Allow())
	assert.Equal(t, CircuitClosed, cb.State())
}

func TestCircuitBreaker_StateTransitions(t *testing.T) {
	t.Parallel()

	cfg := CircuitBreakerConfig{
		Enabled:             true,
		FailureThreshold:    3,
		Window:              time.Minute, // Long window so it doesn't reset during test
		Cooldown:            50 * time.Millisecond,
		HalfOpenMaxRequests: 2,
	}
	cb := NewCircuitBreaker("test", cfg)

	// Start in closed state
	assert.Equal(t, CircuitClosed, cb.State())
	assert.True(t, cb.Allow())

	// Record failures below threshold
	cb.RecordFailure(http.StatusTooManyRequests)
	cb.RecordFailure(http.StatusTooManyRequests)
	assert.Equal(t, CircuitClosed, cb.State())
	assert.True(t, cb.Allow())

	// Third failure should trip the circuit
	tripped := cb.RecordFailure(http.StatusTooManyRequests)
	assert.True(t, tripped)
	assert.Equal(t, CircuitOpen, cb.State())
	assert.False(t, cb.Allow())

	// Wait for cooldown
	time.Sleep(60 * time.Millisecond)

	// Should transition to half-open and allow request
	assert.True(t, cb.Allow())
	assert.Equal(t, CircuitHalfOpen, cb.State())

	// Success in half-open should eventually close
	cb.RecordSuccess()
	cb.RecordSuccess()
	assert.Equal(t, CircuitClosed, cb.State())
	assert.True(t, cb.Allow())
}

func TestCircuitBreaker_HalfOpenFailure(t *testing.T) {
	t.Parallel()

	cfg := CircuitBreakerConfig{
		Enabled:             true,
		FailureThreshold:    2,
		Window:              time.Minute,
		Cooldown:            50 * time.Millisecond,
		HalfOpenMaxRequests: 3,
	}
	cb := NewCircuitBreaker("test", cfg)

	// Trip the circuit
	cb.RecordFailure(http.StatusTooManyRequests)
	cb.RecordFailure(http.StatusTooManyRequests)
	assert.Equal(t, CircuitOpen, cb.State())

	// Wait for cooldown
	time.Sleep(60 * time.Millisecond)

	// Transition to half-open
	assert.True(t, cb.Allow())
	assert.Equal(t, CircuitHalfOpen, cb.State())

	// Failure in half-open should re-open circuit
	tripped := cb.RecordFailure(http.StatusServiceUnavailable)
	assert.True(t, tripped)
	assert.Equal(t, CircuitOpen, cb.State())
	assert.False(t, cb.Allow())
}

func TestCircuitBreaker_OnlyCountsRelevantStatusCodes(t *testing.T) {
	t.Parallel()

	cfg := CircuitBreakerConfig{
		Enabled:             true,
		FailureThreshold:    2,
		Window:              time.Minute,
		Cooldown:            time.Minute,
		HalfOpenMaxRequests: 2,
	}
	cb := NewCircuitBreaker("test", cfg)

	// Non-circuit-breaker status codes should not count
	cb.RecordFailure(http.StatusBadRequest)          // 400
	cb.RecordFailure(http.StatusUnauthorized)        // 401
	cb.RecordFailure(http.StatusInternalServerError) // 500
	cb.RecordFailure(http.StatusBadGateway)          // 502
	assert.Equal(t, CircuitClosed, cb.State())
	assert.Equal(t, int64(0), cb.Failures())

	// These should count
	cb.RecordFailure(http.StatusTooManyRequests)   // 429
	assert.Equal(t, int64(1), cb.Failures())
	
	cb.RecordFailure(http.StatusServiceUnavailable) // 503
	assert.Equal(t, CircuitOpen, cb.State())
}

func TestCircuitBreaker_Anthropic529(t *testing.T) {
	t.Parallel()

	cfg := CircuitBreakerConfig{
		Enabled:             true,
		FailureThreshold:    1,
		Window:              time.Minute,
		Cooldown:            time.Minute,
		HalfOpenMaxRequests: 1,
	}
	cb := NewCircuitBreaker("anthropic", cfg)

	// Anthropic-specific 529 "Overloaded" should trip the circuit
	tripped := cb.RecordFailure(529)
	assert.True(t, tripped)
	assert.Equal(t, CircuitOpen, cb.State())
}

func TestCircuitBreaker_WindowReset(t *testing.T) {
	t.Parallel()

	cfg := CircuitBreakerConfig{
		Enabled:             true,
		FailureThreshold:    3,
		Window:              50 * time.Millisecond, // Short window
		Cooldown:            time.Minute,
		HalfOpenMaxRequests: 2,
	}
	cb := NewCircuitBreaker("test", cfg)

	// Record failures
	cb.RecordFailure(http.StatusTooManyRequests)
	cb.RecordFailure(http.StatusTooManyRequests)
	assert.Equal(t, int64(2), cb.Failures())

	// Wait for window to expire
	time.Sleep(60 * time.Millisecond)

	// Next failure should reset counter (due to window expiry)
	cb.RecordFailure(http.StatusTooManyRequests)
	assert.Equal(t, int64(1), cb.Failures())
	assert.Equal(t, CircuitClosed, cb.State())
}

func TestCircuitBreaker_ConcurrentAccess(t *testing.T) {
	t.Parallel()

	cfg := CircuitBreakerConfig{
		Enabled:             true,
		FailureThreshold:    100,
		Window:              time.Minute,
		Cooldown:            time.Minute,
		HalfOpenMaxRequests: 10,
	}
	cb := NewCircuitBreaker("test", cfg)

	var wg sync.WaitGroup
	numGoroutines := 50
	opsPerGoroutine := 100

	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < opsPerGoroutine; j++ {
				cb.Allow()
				cb.RecordSuccess()
				cb.RecordFailure(http.StatusTooManyRequests)
				cb.State()
				cb.Failures()
			}
		}()
	}

	wg.Wait()
	// Should not panic or deadlock
}

func TestCircuitBreaker_StateChangeCallback(t *testing.T) {
	t.Parallel()

	cfg := CircuitBreakerConfig{
		Enabled:             true,
		FailureThreshold:    2,
		Window:              time.Minute,
		Cooldown:            50 * time.Millisecond,
		HalfOpenMaxRequests: 1,
	}
	cb := NewCircuitBreaker("test", cfg)

	var mu sync.Mutex
	var transitions []struct {
		from, to CircuitState
	}

	cb.SetStateChangeCallback(func(provider string, from, to CircuitState) {
		mu.Lock()
		defer mu.Unlock()
		transitions = append(transitions, struct{ from, to CircuitState }{from, to})
	})

	// Trip the circuit
	cb.RecordFailure(http.StatusTooManyRequests)
	cb.RecordFailure(http.StatusTooManyRequests)

	// Wait for callback
	time.Sleep(10 * time.Millisecond)

	// Wait for cooldown and trigger half-open
	time.Sleep(60 * time.Millisecond)
	cb.Allow()

	// Wait for callback
	time.Sleep(10 * time.Millisecond)

	// Success to close
	cb.RecordSuccess()

	// Wait for callback
	time.Sleep(10 * time.Millisecond)

	mu.Lock()
	defer mu.Unlock()
	require.Len(t, transitions, 3)
	assert.Equal(t, CircuitClosed, transitions[0].from)
	assert.Equal(t, CircuitOpen, transitions[0].to)
	assert.Equal(t, CircuitOpen, transitions[1].from)
	assert.Equal(t, CircuitHalfOpen, transitions[1].to)
	assert.Equal(t, CircuitHalfOpen, transitions[2].from)
	assert.Equal(t, CircuitClosed, transitions[2].to)
}

func TestCircuitBreakerManager_GetOrCreate(t *testing.T) {
	t.Parallel()

	cfg := CircuitBreakerConfig{
		Enabled:          true,
		FailureThreshold: 5,
		Window:           time.Minute,
		Cooldown:         time.Minute,
	}
	manager := NewCircuitBreakerManager(cfg)

	// First call should create
	cb1 := manager.GetOrCreate("anthropic")
	require.NotNil(t, cb1)
	assert.Equal(t, "anthropic", cb1.Provider())

	// Second call should return same instance
	cb2 := manager.GetOrCreate("anthropic")
	assert.Same(t, cb1, cb2)

	// Different provider gets different instance
	cb3 := manager.GetOrCreate("openai")
	require.NotNil(t, cb3)
	assert.NotSame(t, cb1, cb3)
	assert.Equal(t, "openai", cb3.Provider())
}

func TestCircuitBreakerManager_AllStates(t *testing.T) {
	t.Parallel()

	cfg := CircuitBreakerConfig{
		Enabled:          true,
		FailureThreshold: 1,
		Window:           time.Minute,
		Cooldown:         time.Minute,
	}
	manager := NewCircuitBreakerManager(cfg)

	manager.GetOrCreate("anthropic")
	manager.GetOrCreate("openai")

	// Trip one circuit
	manager.Get("anthropic").RecordFailure(http.StatusTooManyRequests)

	states := manager.AllStates()
	assert.Equal(t, CircuitOpen, states["anthropic"])
	assert.Equal(t, CircuitClosed, states["openai"])
}

func TestCircuitBreakerManager_ConcurrentGetOrCreate(t *testing.T) {
	t.Parallel()

	cfg := DefaultCircuitBreakerConfig()
	cfg.Enabled = true
	manager := NewCircuitBreakerManager(cfg)

	var wg sync.WaitGroup
	var results [100]*CircuitBreaker

	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			results[idx] = manager.GetOrCreate("test-provider")
		}(i)
	}

	wg.Wait()

	// All should be the same instance
	first := results[0]
	for i := 1; i < 100; i++ {
		assert.Same(t, first, results[i])
	}
}

func TestIsCircuitBreakerFailure(t *testing.T) {
	t.Parallel()

	tests := []struct {
		statusCode int
		isFailure  bool
	}{
		{http.StatusOK, false},
		{http.StatusBadRequest, false},
		{http.StatusUnauthorized, false},
		{http.StatusForbidden, false},
		{http.StatusNotFound, false},
		{http.StatusTooManyRequests, true}, // 429
		{http.StatusInternalServerError, false},
		{http.StatusBadGateway, false},
		{http.StatusServiceUnavailable, true}, // 503
		{529, true},                           // Anthropic Overloaded
	}

	for _, tt := range tests {
		t.Run(http.StatusText(tt.statusCode), func(t *testing.T) {
			assert.Equal(t, tt.isFailure, isCircuitBreakerFailure(tt.statusCode))
		})
	}
}

func TestCircuitState_String(t *testing.T) {
	t.Parallel()

	assert.Equal(t, "closed", CircuitClosed.String())
	assert.Equal(t, "open", CircuitOpen.String())
	assert.Equal(t, "half-open", CircuitHalfOpen.String())
	assert.Equal(t, "unknown", CircuitState(99).String())
}
