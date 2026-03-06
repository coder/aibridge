package provider

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"strings"
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
		expected map[string]string
	}{
		{
			name:     "no headers",
			headers:  map[string]string{},
			expected: map[string]string{},
		},
		{
			name:     "single beta",
			headers:  map[string]string{"Anthropic-Beta": "claude-code-20250219"},
			expected: map[string]string{"Anthropic-Beta": "claude-code-20250219"},
		},
		{
			name:     "multiple betas in single header",
			headers:  map[string]string{"Anthropic-Beta": "claude-code-20250219,adaptive-thinking-2026-01-28,context-management-2025-06-27,prompt-caching-scope-2026-01-05,effort-2025-11-24"},
			expected: map[string]string{"Anthropic-Beta": "claude-code-20250219,adaptive-thinking-2026-01-28,context-management-2025-06-27,prompt-caching-scope-2026-01-05,effort-2025-11-24"},
		},
		{
			name:     "ignores other headers",
			headers:  map[string]string{"Anthropic-Beta": "claude-code-20250219,context-management-2025-06-27", "X-Api-Key": "secret"},
			expected: map[string]string{"Anthropic-Beta": "claude-code-20250219,context-management-2025-06-27"},
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

// TestAnthropic_ForwardsHeadersToUpstream verifies that custom client headers are forwarded
// to the upstream, while filtered headers (auth, hop-by-hop) are stripped and the
// configured API key is always used.
func TestAnthropic_ForwardsHeadersToUpstream(t *testing.T) {
	t.Parallel()

	var receivedHeaders http.Header

	mockUpstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		receivedHeaders = r.Header.Clone()
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"id":"msg-123","type":"message","role":"assistant","content":[{"type":"text","text":"Hello!"}],"model":"claude-3-haiku-20240307","stop_reason":"end_turn","usage":{"input_tokens":10,"output_tokens":5}}`))
	}))
	t.Cleanup(mockUpstream.Close)

	const configuredKey = "configured-key"
	p := NewAnthropic(config.Anthropic{
		BaseURL: mockUpstream.URL,
		Key:     configuredKey,
	}, nil)

	body := `{"model":"claude-3-haiku-20240307","max_tokens":1024,"messages":[{"role":"user","content":"hello"}],"stream":false}`
	req := httptest.NewRequest(http.MethodPost, routeMessages, bytes.NewBufferString(body))
	req.Header.Set("X-Custom-Header", "custom-value")         // should be forwarded
	req.Header.Set("Authorization", "Bearer client-fake-key") // should be stripped (re-injected by SDK)
	req.Header.Set("X-Api-Key", "client-fake-key")            // should be stripped (re-injected by SDK)
	req.Header.Set("Upgrade", "websocket")                    // should be stripped (hop-by-hop)
	w := httptest.NewRecorder()

	interceptor, err := p.CreateInterceptor(w, req, testTracer)
	require.NoError(t, err)
	require.NotNil(t, interceptor)

	interceptor.Setup(slog.Make(), &testutil.MockRecorder{}, nil)

	err = interceptor.ProcessRequest(w, httptest.NewRequest(http.MethodPost, routeMessages, nil))
	require.NoError(t, err)

	// Custom headers must reach the upstream.
	assert.Equal(t, "custom-value", receivedHeaders.Get("X-Custom-Header"))
	// Hop-by-hop headers must not reach the upstream.
	assert.Empty(t, receivedHeaders.Get("Upgrade"))
	// Configured key must be used, not the client-provided fake.
	assert.Equal(t, configuredKey, receivedHeaders.Get("X-Api-Key"))
	// User-Agent must be set by the SDK, not forwarded from the client.
	assert.True(t, strings.HasPrefix(receivedHeaders.Get("User-Agent"), "Anthropic/Go"),
		"upstream User-Agent should be set by the SDK")
}
