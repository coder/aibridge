package circuitbreaker

import (
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/coder/aibridge/config"
	"github.com/sony/gobreaker/v2"
	"github.com/stretchr/testify/assert"
)

func TestExecute_PerModelIsolation(t *testing.T) {
	t.Parallel()

	sonnetCalls := atomic.Int32{}
	haikuCalls := atomic.Int32{}

	cbs := NewProviderCircuitBreakers("test", &config.CircuitBreaker{
		FailureThreshold: 1,
		Interval:         time.Minute,
		Timeout:          time.Minute,
		MaxRequests:      1,
	}, func(endpoint, model string, from, to gobreaker.State) {})

	endpoint := "/v1/messages"
	sonnetModel := "claude-sonnet-4-20250514"
	haikuModel := "claude-3-5-haiku-20241022"

	// Trip circuit on sonnet model (returns 429)
	w := httptest.NewRecorder()
	result := cbs.Execute(endpoint, sonnetModel, w, func(rw http.ResponseWriter) {
		sonnetCalls.Add(1)
		rw.WriteHeader(http.StatusTooManyRequests)
	})
	assert.False(t, result.CircuitOpen)
	assert.Equal(t, http.StatusTooManyRequests, result.StatusCode)
	assert.Equal(t, int32(1), sonnetCalls.Load())

	// Second sonnet request should be blocked by circuit breaker
	w = httptest.NewRecorder()
	result = cbs.Execute(endpoint, sonnetModel, w, func(rw http.ResponseWriter) {
		sonnetCalls.Add(1)
		rw.WriteHeader(http.StatusOK)
	})
	assert.True(t, result.CircuitOpen)
	assert.Equal(t, int32(1), sonnetCalls.Load()) // No new call

	// Haiku model on same endpoint should still work (independent circuit)
	w = httptest.NewRecorder()
	result = cbs.Execute(endpoint, haikuModel, w, func(rw http.ResponseWriter) {
		haikuCalls.Add(1)
		rw.WriteHeader(http.StatusOK)
	})
	assert.False(t, result.CircuitOpen)
	assert.Equal(t, http.StatusOK, result.StatusCode)
	assert.Equal(t, int32(1), haikuCalls.Load())
}

func TestExecute_PerEndpointIsolation(t *testing.T) {
	t.Parallel()

	messagesCalls := atomic.Int32{}
	completionsCalls := atomic.Int32{}

	cbs := NewProviderCircuitBreakers("test", &config.CircuitBreaker{
		FailureThreshold: 1,
		Interval:         time.Minute,
		Timeout:          time.Minute,
		MaxRequests:      1,
	}, func(endpoint, model string, from, to gobreaker.State) {})

	model := "test-model"

	// Trip circuit on /v1/messages endpoint (returns 429)
	w := httptest.NewRecorder()
	result := cbs.Execute("/v1/messages", model, w, func(rw http.ResponseWriter) {
		messagesCalls.Add(1)
		rw.WriteHeader(http.StatusTooManyRequests)
	})
	assert.False(t, result.CircuitOpen)
	assert.Equal(t, int32(1), messagesCalls.Load())

	// Second /v1/messages request should be blocked
	w = httptest.NewRecorder()
	result = cbs.Execute("/v1/messages", model, w, func(rw http.ResponseWriter) {
		messagesCalls.Add(1)
		rw.WriteHeader(http.StatusOK)
	})
	assert.True(t, result.CircuitOpen)
	assert.Equal(t, int32(1), messagesCalls.Load()) // No new call

	// /v1/chat/completions on same model should still work (different endpoint)
	w = httptest.NewRecorder()
	result = cbs.Execute("/v1/chat/completions", model, w, func(rw http.ResponseWriter) {
		completionsCalls.Add(1)
		rw.WriteHeader(http.StatusOK)
	})
	assert.False(t, result.CircuitOpen)
	assert.Equal(t, http.StatusOK, result.StatusCode)
	assert.Equal(t, int32(1), completionsCalls.Load())
}

func TestExecute_CustomIsFailure(t *testing.T) {
	t.Parallel()

	var calls atomic.Int32

	// Custom IsFailure that treats 502 as failure
	cbs := NewProviderCircuitBreakers("test", &config.CircuitBreaker{
		FailureThreshold: 1,
		Interval:         time.Minute,
		Timeout:          time.Minute,
		MaxRequests:      1,
		IsFailure: func(statusCode int) bool {
			return statusCode == http.StatusBadGateway
		},
	}, func(endpoint, model string, from, to gobreaker.State) {})

	// First request returns 502, trips circuit
	w := httptest.NewRecorder()
	result := cbs.Execute("/v1/messages", "test-model", w, func(rw http.ResponseWriter) {
		calls.Add(1)
		rw.WriteHeader(http.StatusBadGateway)
	})
	assert.False(t, result.CircuitOpen)
	assert.Equal(t, http.StatusBadGateway, result.StatusCode)
	assert.Equal(t, int32(1), calls.Load())

	// Second request should be blocked
	w = httptest.NewRecorder()
	result = cbs.Execute("/v1/messages", "test-model", w, func(rw http.ResponseWriter) {
		calls.Add(1)
		rw.WriteHeader(http.StatusOK)
	})
	assert.True(t, result.CircuitOpen)
	assert.Equal(t, int32(1), calls.Load()) // No new call
}

func TestExecute_OnStateChange(t *testing.T) {
	t.Parallel()

	var stateChanges []struct {
		endpoint string
		model    string
		from     gobreaker.State
		to       gobreaker.State
	}

	cbs := NewProviderCircuitBreakers("test", &config.CircuitBreaker{
		FailureThreshold: 1,
		Interval:         time.Minute,
		Timeout:          time.Minute,
		MaxRequests:      1,
	}, func(endpoint, model string, from, to gobreaker.State) {
		stateChanges = append(stateChanges, struct {
			endpoint string
			model    string
			from     gobreaker.State
			to       gobreaker.State
		}{endpoint, model, from, to})
	})

	endpoint := "/v1/messages"
	model := "claude-sonnet-4-20250514"

	// Trip circuit
	w := httptest.NewRecorder()
	cbs.Execute(endpoint, model, w, func(rw http.ResponseWriter) {
		rw.WriteHeader(http.StatusTooManyRequests)
	})

	// Verify state change callback was called with correct parameters
	assert.Len(t, stateChanges, 1)
	assert.Equal(t, endpoint, stateChanges[0].endpoint)
	assert.Equal(t, model, stateChanges[0].model)
	assert.Equal(t, gobreaker.StateClosed, stateChanges[0].from)
	assert.Equal(t, gobreaker.StateOpen, stateChanges[0].to)
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
		{http.StatusGatewayTimeout, true},     // 504
	}

	for _, tt := range tests {
		assert.Equal(t, tt.isFailure, DefaultIsFailure(tt.statusCode), "status code %d", tt.statusCode)
	}
}

func TestStateToGaugeValue(t *testing.T) {
	t.Parallel()

	assert.Equal(t, float64(0), StateToGaugeValue(gobreaker.StateClosed))
	assert.Equal(t, float64(0.5), StateToGaugeValue(gobreaker.StateHalfOpen))
	assert.Equal(t, float64(1), StateToGaugeValue(gobreaker.StateOpen))
}
