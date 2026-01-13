package circuitbreaker

import (
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"cdr.dev/slog/v3"
	"cdr.dev/slog/v3/sloggers/slogtest"

	"github.com/sony/gobreaker/v2"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestMiddleware_PerEndpointIsolation(t *testing.T) {
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

	cbs := NewProviderCircuitBreakers("test", &Config{
		FailureThreshold: 1,
		Interval:         time.Minute,
		Timeout:          time.Minute,
		MaxRequests:      1,
	}, func(endpoint string, from, to gobreaker.State) {})

	logger := slogtest.Make(t, &slogtest.Options{IgnoreErrors: false}).Leveled(slog.LevelDebug)
	handler := Middleware(cbs, nil, logger)(upstream)
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
	assert.Equal(t, "60", resp.Header.Get("Retry-After")) // Timeout is 1 minute
	assert.Equal(t, int32(1), chatCalls.Load())           // Only 1 call, second was blocked

	// /responses should still work
	resp, err = http.Get(server.URL + "/test/v1/responses")
	require.NoError(t, err)
	resp.Body.Close()
	assert.Equal(t, http.StatusOK, resp.StatusCode)
	assert.Equal(t, int32(1), responsesCalls.Load())
}

func TestMiddleware_NotConfigured(t *testing.T) {
	t.Parallel()

	var upstreamCalls atomic.Int32

	upstream := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		upstreamCalls.Add(1)
		w.WriteHeader(http.StatusTooManyRequests)
	})

	// No circuit breaker configured (nil)
	logger := slogtest.Make(t, &slogtest.Options{IgnoreErrors: false}).Leveled(slog.LevelDebug)
	handler := Middleware(nil, nil, logger)(upstream)
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

func TestMiddleware_CustomIsFailure(t *testing.T) {
	t.Parallel()

	upstream := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadGateway) // 502
	})

	// Custom IsFailure that treats 502 as failure
	cbs := NewProviderCircuitBreakers("test", &Config{
		FailureThreshold: 1,
		Interval:         time.Minute,
		Timeout:          time.Minute,
		MaxRequests:      1,
		IsFailure: func(statusCode int) bool {
			return statusCode == http.StatusBadGateway
		},
	}, func(endpoint string, from, to gobreaker.State) {})

	logger := slogtest.Make(t, &slogtest.Options{IgnoreErrors: false}).Leveled(slog.LevelDebug)
	handler := Middleware(cbs, nil, logger)(upstream)
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
	}

	for _, tt := range tests {
		assert.Equal(t, tt.isFailure, DefaultIsFailure(tt.statusCode), "status code %d", tt.statusCode)
	}
}

func TestAnthropicIsFailure(t *testing.T) {
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
		assert.Equal(t, tt.isFailure, AnthropicIsFailure(tt.statusCode), "status code %d", tt.statusCode)
	}
}

func TestStateToGaugeValue(t *testing.T) {
	t.Parallel()

	assert.Equal(t, float64(0), StateToGaugeValue(gobreaker.StateClosed))
	assert.Equal(t, float64(0.5), StateToGaugeValue(gobreaker.StateHalfOpen))
	assert.Equal(t, float64(1), StateToGaugeValue(gobreaker.StateOpen))
}
