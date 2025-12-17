package aibridge

import (
	"io"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/sony/gobreaker/v2"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestCircuitBreakerMiddleware_TripsOnUpstreamErrors(t *testing.T) {
	t.Parallel()

	var upstreamCalls atomic.Int32

	// Mock upstream that returns 429
	upstream := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		upstreamCalls.Add(1)
		w.WriteHeader(http.StatusTooManyRequests)
	})

	// Create circuit breaker with low threshold
	cbs := NewCircuitBreakers(map[string]CircuitBreakerConfig{
		"test": {
			FailureThreshold: 2,
			Interval:         time.Minute,
			Timeout:          50 * time.Millisecond,
			MaxRequests:      1,
		},
	}, nil)

	// Wrap upstream with circuit breaker middleware
	handler := CircuitBreakerMiddleware(cbs, nil, "test")(upstream)
	server := httptest.NewServer(handler)
	defer server.Close()

	// First 2 requests hit upstream, get 429
	for i := 0; i < 2; i++ {
		resp, err := http.Get(server.URL + "/test/v1/messages")
		require.NoError(t, err)
		resp.Body.Close()
		assert.Equal(t, http.StatusTooManyRequests, resp.StatusCode)
	}
	assert.Equal(t, int32(2), upstreamCalls.Load())

	// Third request should get 503 "circuit breaker is open" without hitting upstream
	resp, err := http.Get(server.URL + "/test/v1/messages")
	require.NoError(t, err)
	body, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	assert.Equal(t, http.StatusServiceUnavailable, resp.StatusCode)
	assert.Contains(t, string(body), "circuit breaker is open")
	assert.Equal(t, int32(2), upstreamCalls.Load()) // No new upstream call

	// Wait for timeout, verify recovery (circuit transitions to half-open)
	require.Eventually(t, func() bool {
		resp, err = http.Get(server.URL + "/test/v1/messages")
		if err != nil {
			return false
		}
		resp.Body.Close()
		// Request hit upstream again (half-open state allows probe request)
		return upstreamCalls.Load() == 3
	}, 5*time.Second, 25*time.Millisecond)
}

func TestCircuitBreakerMiddleware_PerEndpointIsolation(t *testing.T) {
	t.Parallel()

	chatCalls := atomic.Int32{}
	responsesCalls := atomic.Int32{}

	// Mock upstream - /chat returns 429, /responses returns 200
	upstream := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/test/v1/chat/completions" {
			chatCalls.Add(1)
			w.WriteHeader(http.StatusTooManyRequests)
		} else {
			responsesCalls.Add(1)
			w.WriteHeader(http.StatusOK)
		}
	})

	cbs := NewCircuitBreakers(map[string]CircuitBreakerConfig{
		"test": {
			FailureThreshold: 1,
			Interval:         time.Minute,
			Timeout:          time.Minute,
			MaxRequests:      1,
		},
	}, nil)

	handler := CircuitBreakerMiddleware(cbs, nil, "test")(upstream)
	server := httptest.NewServer(handler)
	defer server.Close()

	// Trip circuit on /chat/completions
	resp, err := http.Get(server.URL + "/test/v1/chat/completions")
	require.NoError(t, err)
	resp.Body.Close()

	// /chat/completions should now be blocked
	resp, err = http.Get(server.URL + "/test/v1/chat/completions")
	require.NoError(t, err)
	resp.Body.Close()
	assert.Equal(t, http.StatusServiceUnavailable, resp.StatusCode)
	assert.Equal(t, int32(1), chatCalls.Load()) // Only 1 call, second was blocked

	// /responses should still work
	resp, err = http.Get(server.URL + "/test/v1/responses")
	require.NoError(t, err)
	resp.Body.Close()
	assert.Equal(t, http.StatusOK, resp.StatusCode)
	assert.Equal(t, int32(1), responsesCalls.Load())
}

func TestCircuitBreakerMiddleware_NotConfigured(t *testing.T) {
	t.Parallel()

	var upstreamCalls atomic.Int32

	upstream := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		upstreamCalls.Add(1)
		w.WriteHeader(http.StatusTooManyRequests)
	})

	// No config for "test" provider
	cbs := NewCircuitBreakers(nil, nil)

	handler := CircuitBreakerMiddleware(cbs, nil, "test")(upstream)
	server := httptest.NewServer(handler)
	defer server.Close()

	// All requests should pass through even with 429s
	for i := 0; i < 10; i++ {
		resp, err := http.Get(server.URL + "/test/v1/messages")
		require.NoError(t, err)
		resp.Body.Close()
		assert.Equal(t, http.StatusTooManyRequests, resp.StatusCode)
	}
	assert.Equal(t, int32(10), upstreamCalls.Load())
}

func TestCircuitBreakerMiddleware_RecoveryAfterSuccess(t *testing.T) {
	t.Parallel()

	var returnError atomic.Bool
	returnError.Store(true)

	upstream := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if returnError.Load() {
			w.WriteHeader(http.StatusTooManyRequests)
		} else {
			w.WriteHeader(http.StatusOK)
		}
	})

	cbs := NewCircuitBreakers(map[string]CircuitBreakerConfig{
		"test": {
			FailureThreshold: 2,
			Interval:         time.Minute,
			Timeout:          50 * time.Millisecond,
			MaxRequests:      1,
		},
	}, nil)

	handler := CircuitBreakerMiddleware(cbs, nil, "test")(upstream)
	server := httptest.NewServer(handler)
	defer server.Close()

	// Trip the circuit
	for i := 0; i < 2; i++ {
		resp, _ := http.Get(server.URL + "/test/v1/messages")
		resp.Body.Close()
	}

	// Circuit should be open
	resp, _ := http.Get(server.URL + "/test/v1/messages")
	assert.Equal(t, http.StatusServiceUnavailable, resp.StatusCode)
	resp.Body.Close()

	// Switch upstream to success before we start polling
	returnError.Store(false)

	// Wait for timeout (circuit transitions to half-open), then verify recovery
	require.Eventually(t, func() bool {
		resp, err := http.Get(server.URL + "/test/v1/messages")
		if err != nil {
			return false
		}
		defer resp.Body.Close()
		// Half-open: request goes through and succeeds
		return resp.StatusCode == http.StatusOK
	}, 5*time.Second, 25*time.Millisecond)

	// Circuit should be closed now, more requests allowed
	resp, _ = http.Get(server.URL + "/test/v1/messages")
	assert.Equal(t, http.StatusOK, resp.StatusCode)
	resp.Body.Close()
}

func TestCircuitBreakerMiddleware_CustomIsFailure(t *testing.T) {
	t.Parallel()

	upstream := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadGateway) // 502
	})

	// Custom IsFailure that treats 502 as failure
	cbs := NewCircuitBreakers(map[string]CircuitBreakerConfig{
		"test": {
			FailureThreshold: 1,
			Interval:         time.Minute,
			Timeout:          time.Minute,
			MaxRequests:      1,
			IsFailure: func(statusCode int) bool {
				return statusCode == http.StatusBadGateway
			},
		},
	}, nil)

	handler := CircuitBreakerMiddleware(cbs, nil, "test")(upstream)
	server := httptest.NewServer(handler)
	defer server.Close()

	// First request returns 502, trips circuit
	resp, _ := http.Get(server.URL + "/test/v1/messages")
	resp.Body.Close()

	// Second request should be blocked
	resp, _ = http.Get(server.URL + "/test/v1/messages")
	assert.Equal(t, http.StatusServiceUnavailable, resp.StatusCode)
	resp.Body.Close()
}

func TestDefaultIsFailure(t *testing.T) {
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
		{http.StatusBadGateway, false},
		{http.StatusServiceUnavailable, true}, // 503
		{529, true},                           // Anthropic Overloaded
	}

	for _, tt := range tests {
		assert.Equal(t, tt.isFailure, DefaultIsFailure(tt.statusCode), "status code %d", tt.statusCode)
	}
}

func TestStateToGaugeValue(t *testing.T) {
	t.Parallel()

	assert.Equal(t, float64(0), stateToGaugeValue(gobreaker.StateClosed))
	assert.Equal(t, float64(0.5), stateToGaugeValue(gobreaker.StateHalfOpen))
	assert.Equal(t, float64(1), stateToGaugeValue(gobreaker.StateOpen))
}
