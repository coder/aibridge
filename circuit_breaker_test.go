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

func TestCircuitBreakers_DisabledByDefault(t *testing.T) {
	t.Parallel()

	cbs := NewCircuitBreakers(DefaultCircuitBreakerConfig(), nil)

	// Should always allow when disabled
	assert.True(t, cbs.Allow("anthropic", "/v1/messages"))

	// Recording failures should not affect state when disabled
	for i := 0; i < 100; i++ {
		cbs.RecordFailure("anthropic", "/v1/messages", http.StatusTooManyRequests)
	}
	assert.True(t, cbs.Allow("anthropic", "/v1/messages"))
	assert.Equal(t, CircuitClosed, cbs.State("anthropic", "/v1/messages"))
}

func TestCircuitBreakers_StateTransitions(t *testing.T) {
	t.Parallel()

	cfg := CircuitBreakerConfig{
		Enabled:             true,
		FailureThreshold:    3,
		Window:              time.Minute,
		Cooldown:            50 * time.Millisecond,
		HalfOpenMaxRequests: 2,
	}
	cbs := NewCircuitBreakers(cfg, nil)

	// Start in closed state
	assert.Equal(t, CircuitClosed, cbs.State("test", "/api"))
	assert.True(t, cbs.Allow("test", "/api"))

	// Record failures below threshold
	cbs.RecordFailure("test", "/api", http.StatusTooManyRequests)
	cbs.RecordFailure("test", "/api", http.StatusTooManyRequests)
	assert.Equal(t, CircuitClosed, cbs.State("test", "/api"))

	// Third failure should trip the circuit
	tripped := cbs.RecordFailure("test", "/api", http.StatusTooManyRequests)
	assert.True(t, tripped)
	assert.Equal(t, CircuitOpen, cbs.State("test", "/api"))
	assert.False(t, cbs.Allow("test", "/api"))

	// Wait for cooldown
	time.Sleep(60 * time.Millisecond)

	// Should transition to half-open and allow request
	assert.True(t, cbs.Allow("test", "/api"))
	assert.Equal(t, CircuitHalfOpen, cbs.State("test", "/api"))

	// Success in half-open should eventually close
	cbs.RecordSuccess("test", "/api")
	cbs.RecordSuccess("test", "/api")
	assert.Equal(t, CircuitClosed, cbs.State("test", "/api"))
}

func TestCircuitBreakers_PerEndpointIsolation(t *testing.T) {
	t.Parallel()

	cfg := CircuitBreakerConfig{
		Enabled:             true,
		FailureThreshold:    1,
		Window:              time.Minute,
		Cooldown:            time.Minute,
		HalfOpenMaxRequests: 1,
	}
	cbs := NewCircuitBreakers(cfg, nil)

	// Trip circuit for one endpoint
	cbs.RecordFailure("openai", "/v1/chat/completions", http.StatusTooManyRequests)
	assert.Equal(t, CircuitOpen, cbs.State("openai", "/v1/chat/completions"))

	// Other endpoints should still be closed
	assert.Equal(t, CircuitClosed, cbs.State("openai", "/v1/responses"))
	assert.Equal(t, CircuitClosed, cbs.State("anthropic", "/v1/messages"))
	assert.True(t, cbs.Allow("openai", "/v1/responses"))
	assert.True(t, cbs.Allow("anthropic", "/v1/messages"))
}

func TestCircuitBreakers_OnlyCountsRelevantStatusCodes(t *testing.T) {
	t.Parallel()

	cfg := CircuitBreakerConfig{
		Enabled:             true,
		FailureThreshold:    2,
		Window:              time.Minute,
		Cooldown:            time.Minute,
		HalfOpenMaxRequests: 2,
	}
	cbs := NewCircuitBreakers(cfg, nil)

	// Non-circuit-breaker status codes should not count
	cbs.RecordFailure("test", "/api", http.StatusBadRequest)          // 400
	cbs.RecordFailure("test", "/api", http.StatusUnauthorized)        // 401
	cbs.RecordFailure("test", "/api", http.StatusInternalServerError) // 500
	cbs.RecordFailure("test", "/api", http.StatusBadGateway)          // 502
	assert.Equal(t, CircuitClosed, cbs.State("test", "/api"))

	// These should count
	cbs.RecordFailure("test", "/api", http.StatusTooManyRequests)    // 429
	cbs.RecordFailure("test", "/api", http.StatusServiceUnavailable) // 503
	assert.Equal(t, CircuitOpen, cbs.State("test", "/api"))
}

func TestCircuitBreakers_Anthropic529(t *testing.T) {
	t.Parallel()

	cfg := CircuitBreakerConfig{
		Enabled:             true,
		FailureThreshold:    1,
		Window:              time.Minute,
		Cooldown:            time.Minute,
		HalfOpenMaxRequests: 1,
	}
	cbs := NewCircuitBreakers(cfg, nil)

	// Anthropic-specific 529 "Overloaded" should trip the circuit
	tripped := cbs.RecordFailure("anthropic", "/v1/messages", 529)
	assert.True(t, tripped)
	assert.Equal(t, CircuitOpen, cbs.State("anthropic", "/v1/messages"))
}

func TestCircuitBreakers_ConcurrentAccess(t *testing.T) {
	t.Parallel()

	cfg := CircuitBreakerConfig{
		Enabled:             true,
		FailureThreshold:    1000,
		Window:              time.Minute,
		Cooldown:            time.Minute,
		HalfOpenMaxRequests: 10,
	}
	cbs := NewCircuitBreakers(cfg, nil)

	var wg sync.WaitGroup
	for i := 0; i < 50; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 100; j++ {
				cbs.Allow("test", "/api")
				cbs.RecordSuccess("test", "/api")
				cbs.RecordFailure("test", "/api", http.StatusTooManyRequests)
				cbs.State("test", "/api")
			}
		}()
	}
	wg.Wait()
	// Should not panic or deadlock
}

func TestCircuitBreakers_StateChangeCallback(t *testing.T) {
	t.Parallel()

	cfg := CircuitBreakerConfig{
		Enabled:             true,
		FailureThreshold:    2,
		Window:              time.Minute,
		Cooldown:            50 * time.Millisecond,
		HalfOpenMaxRequests: 1,
	}

	var mu sync.Mutex
	var transitions []struct{ from, to CircuitState }

	cbs := NewCircuitBreakers(cfg, func(name string, from, to CircuitState) {
		mu.Lock()
		defer mu.Unlock()
		transitions = append(transitions, struct{ from, to CircuitState }{from, to})
	})

	// Trip the circuit
	cbs.RecordFailure("test", "/api", http.StatusTooManyRequests)
	cbs.RecordFailure("test", "/api", http.StatusTooManyRequests)

	// Wait for cooldown and trigger half-open
	time.Sleep(60 * time.Millisecond)
	cbs.Allow("test", "/api")

	// Success to close
	cbs.RecordSuccess("test", "/api")

	// Wait for callbacks
	time.Sleep(20 * time.Millisecond)

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

func TestIsCircuitBreakerFailure(t *testing.T) {
	t.Parallel()

	tests := []struct {
		statusCode int
		isFailure  bool
	}{
		{http.StatusOK, false},
		{http.StatusBadRequest, false},
		{http.StatusUnauthorized, false},
		{http.StatusTooManyRequests, true}, // 429
		{http.StatusInternalServerError, false},
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
