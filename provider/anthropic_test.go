package provider

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"testing"

	"cdr.dev/slog/v3"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/coder/aibridge/config"
	"github.com/coder/aibridge/internal/testutil"
)

func TestAnthropic_CreateInterceptor(t *testing.T) {
	t.Parallel()

	provider := NewAnthropic(config.Anthropic{Key: "test-key"}, nil)

	t.Run("Messages_NonStreamingRequest_BlockingInterceptor", func(t *testing.T) {
		t.Parallel()

		body := `{"model": "claude-opus-4-5", "max_tokens": 1024, "messages": [{"role": "user", "content": "hello"}], "stream": false}`
		req := httptest.NewRequest(http.MethodPost, routeMessages, bytes.NewBufferString(body))
		w := httptest.NewRecorder()

		interceptor, err := provider.CreateInterceptor(w, req, testTracer)

		require.NoError(t, err)
		require.NotNil(t, interceptor)
		assert.False(t, interceptor.Streaming())
	})

	t.Run("Messages_StreamingRequest_StreamingInterceptor", func(t *testing.T) {
		t.Parallel()

		body := `{"model": "claude-opus-4-5", "max_tokens": 1024, "messages": [{"role": "user", "content": "hello"}], "stream": true}`
		req := httptest.NewRequest(http.MethodPost, routeMessages, bytes.NewBufferString(body))
		w := httptest.NewRecorder()

		interceptor, err := provider.CreateInterceptor(w, req, testTracer)

		require.NoError(t, err)
		require.NotNil(t, interceptor)
		assert.True(t, interceptor.Streaming())
	})

	t.Run("Messages_InvalidRequestBody", func(t *testing.T) {
		t.Parallel()

		body := `invalid json`
		req := httptest.NewRequest(http.MethodPost, routeMessages, bytes.NewBufferString(body))
		w := httptest.NewRecorder()

		interceptor, err := provider.CreateInterceptor(w, req, testTracer)

		require.Error(t, err)
		require.Nil(t, interceptor)
		assert.Contains(t, err.Error(), "unmarshal request body")
	})

	t.Run("Messages_ForwardsAnthropicBetaHeaderToUpstream", func(t *testing.T) {
		t.Parallel()

		var receivedHeaders http.Header

		// Mock upstream that captures headers.
		mockUpstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			receivedHeaders = r.Header.Clone()
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(`{"id":"msg-123","type":"message","role":"assistant","content":[{"type":"text","text":"Hello!"}],"model":"claude-opus-4-5","stop_reason":"end_turn","usage":{"input_tokens":10,"output_tokens":5}}`))
		}))
		t.Cleanup(mockUpstream.Close)

		provider := NewAnthropic(config.Anthropic{
			BaseURL: mockUpstream.URL,
			Key:     "test-key",
		}, nil)

		// Use a realistic multi-beta value as sent by Claude Code clients.
		betaHeader := "claude-code-20250219,adaptive-thinking-2026-01-28,context-management-2025-06-27,prompt-caching-scope-2026-01-05,effort-2025-11-24"

		body := `{"model": "claude-opus-4-5", "max_tokens": 1024, "messages": [{"role": "user", "content": "hello"}], "stream": false}`
		req := httptest.NewRequest(http.MethodPost, routeMessages, bytes.NewBufferString(body))
		req.Header.Set("Anthropic-Beta", betaHeader)
		req.Header.Set("X-Custom-Header", "should-not-forward")
		w := httptest.NewRecorder()

		interceptor, err := provider.CreateInterceptor(w, req, testTracer)
		require.NoError(t, err)
		require.NotNil(t, interceptor)

		logger := slog.Make()
		interceptor.Setup(logger, &testutil.MockRecorder{}, nil)

		processReq := httptest.NewRequest(http.MethodPost, routeMessages, nil)
		err = interceptor.ProcessRequest(w, processReq)
		require.NoError(t, err)

		// Verify the full Anthropic-Beta header (all betas) was forwarded unchanged.
		assert.Equal(t, betaHeader, receivedHeaders.Get("Anthropic-Beta"))

		// Verify non-Anthropic headers are not forwarded.
		assert.Empty(t, receivedHeaders.Get("X-Custom-Header"), "non-Anthropic headers should not be forwarded")
	})

	t.Run("UnknownRoute", func(t *testing.T) {
		t.Parallel()

		body := `{"model": "claude-opus-4-5", "max_tokens": 1024, "messages": [{"role": "user", "content": "hello"}]}`
		req := httptest.NewRequest(http.MethodPost, "/anthropic/unknown/route", bytes.NewBufferString(body))
		w := httptest.NewRecorder()

		interceptor, err := provider.CreateInterceptor(w, req, testTracer)

		require.ErrorIs(t, err, UnknownRoute)
		require.Nil(t, interceptor)
	})
}

func TestExtractAnthropicHeaders(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		headers  map[string]string
		expected http.Header
	}{
		{
			name:     "no headers",
			headers:  map[string]string{},
			expected: http.Header{},
		},
		{
			name:     "single beta",
			headers:  map[string]string{"Anthropic-Beta": "claude-code-20250219"},
			expected: http.Header{"Anthropic-Beta": {"claude-code-20250219"}},
		},
		{
			name:     "multiple betas in single header",
			headers:  map[string]string{"Anthropic-Beta": "claude-code-20250219,adaptive-thinking-2026-01-28,context-management-2025-06-27,prompt-caching-scope-2026-01-05,effort-2025-11-24"},
			expected: http.Header{"Anthropic-Beta": {"claude-code-20250219,adaptive-thinking-2026-01-28,context-management-2025-06-27,prompt-caching-scope-2026-01-05,effort-2025-11-24"}},
		},
		{
			name:     "ignores other headers",
			headers:  map[string]string{"Anthropic-Beta": "claude-code-20250219,context-management-2025-06-27", "X-Api-Key": "secret"},
			expected: http.Header{"Anthropic-Beta": {"claude-code-20250219,context-management-2025-06-27"}},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			req := httptest.NewRequest(http.MethodPost, "/", nil)
			for header, value := range tc.headers {
				req.Header.Set(header, value)
			}

			result := extractAnthropicHeaders(req)
			assert.Equal(t, tc.expected, result)
		})
	}
}

func Test_anthropicIsFailure(t *testing.T) {
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
		{529, true},                           // Anthropic Overloaded
	}

	for _, tt := range tests {
		assert.Equal(t, tt.isFailure, anthropicIsFailure(tt.statusCode), "status code %d", tt.statusCode)
	}
}
