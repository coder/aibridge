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
	"go.opentelemetry.io/otel"

	"github.com/coder/aibridge/config"
	"github.com/coder/aibridge/internal/testutil"
)

var testTracer = otel.Tracer("copilot_test")

func TestCopilot_InjectAuthHeader(t *testing.T) {
	t.Parallel()

	// Copilot uses per-user key passed in the Authorization header,
	// so InjectAuthHeader should not modify any headers.
	provider := NewCopilot(config.Copilot{})

	t.Run("EmptyHeaders_NoneAdded", func(t *testing.T) {
		t.Parallel()

		headers := http.Header{}

		provider.InjectAuthHeader(&headers)

		assert.Empty(t, headers, "no headers should be added")
	})

	t.Run("ExistingHeaders_Unchanged", func(t *testing.T) {
		t.Parallel()

		headers := http.Header{}
		headers.Set("Authorization", "Bearer user-token")
		headers.Set("X-Custom-Header", "custom-value")

		provider.InjectAuthHeader(&headers)

		assert.Equal(t, "Bearer user-token", headers.Get("Authorization"),
			"Authorization header should remain unchanged")
		assert.Equal(t, "custom-value", headers.Get("X-Custom-Header"),
			"other headers should remain unchanged")
	})
}

func TestCopilot_CreateInterceptor(t *testing.T) {
	t.Parallel()

	provider := NewCopilot(config.Copilot{})

	t.Run("MissingAuthorizationHeader", func(t *testing.T) {
		t.Parallel()

		body := `{"model": "gpt-4.1", "messages": [{"role": "user", "content": "hello"}]}`
		req := httptest.NewRequest(http.MethodPost, routeCopilotChatCompletions, bytes.NewBufferString(body))
		w := httptest.NewRecorder()

		interceptor, err := provider.CreateInterceptor(w, req, testTracer)

		require.Error(t, err)
		require.Nil(t, interceptor)
		assert.Contains(t, err.Error(), "missing Copilot authorization: Authorization header not found or invalid")
	})

	t.Run("InvalidAuthorizationFormat", func(t *testing.T) {
		t.Parallel()

		body := `{"model": "claude-haiku-4.5", "messages": [{"role": "user", "content": "hello"}]}`
		req := httptest.NewRequest(http.MethodPost, routeCopilotChatCompletions, bytes.NewBufferString(body))
		req.Header.Set("Authorization", "InvalidFormat")
		w := httptest.NewRecorder()

		interceptor, err := provider.CreateInterceptor(w, req, testTracer)

		require.Error(t, err)
		require.Nil(t, interceptor)
		assert.Contains(t, err.Error(), "missing Copilot authorization: Authorization header not found or invalid")
	})

	t.Run("ChatCompletions_NonStreamingRequest_BlockingInterceptor", func(t *testing.T) {
		t.Parallel()

		body := `{"model": "claude-haiku-4.5", "messages": [{"role": "user", "content": "hello"}], "stream": false}`
		req := httptest.NewRequest(http.MethodPost, routeCopilotChatCompletions, bytes.NewBufferString(body))
		req.Header.Set("Authorization", "Bearer test-token")
		w := httptest.NewRecorder()

		interceptor, err := provider.CreateInterceptor(w, req, testTracer)

		require.NoError(t, err)
		require.NotNil(t, interceptor)
		assert.False(t, interceptor.Streaming())
	})

	t.Run("ChatCompletions_StreamingRequest_StreamingInterceptor", func(t *testing.T) {
		t.Parallel()

		body := `{"model": "gpt-4.1", "messages": [{"role": "user", "content": "hello"}], "stream": true}`
		req := httptest.NewRequest(http.MethodPost, routeCopilotChatCompletions, bytes.NewBufferString(body))
		req.Header.Set("Authorization", "Bearer test-token")
		w := httptest.NewRecorder()

		interceptor, err := provider.CreateInterceptor(w, req, testTracer)

		require.NoError(t, err)
		require.NotNil(t, interceptor)
		assert.True(t, interceptor.Streaming())
	})

	t.Run("ChatCompletions_InvalidRequestBody", func(t *testing.T) {
		t.Parallel()

		body := `invalid json`
		req := httptest.NewRequest(http.MethodPost, routeCopilotChatCompletions, bytes.NewBufferString(body))
		req.Header.Set("Authorization", "Bearer test-token")
		w := httptest.NewRecorder()

		interceptor, err := provider.CreateInterceptor(w, req, testTracer)

		require.Error(t, err)
		require.Nil(t, interceptor)
		assert.Contains(t, err.Error(), "unmarshal chat completions request body")
	})

	t.Run("Responses_NonStreamingRequest_BlockingInterceptor", func(t *testing.T) {
		t.Parallel()

		body := `{"model": "gpt-5-mini", "input": "hello", "stream": false}`
		req := httptest.NewRequest(http.MethodPost, routeCopilotResponses, bytes.NewBufferString(body))
		req.Header.Set("Authorization", "Bearer test-token")
		w := httptest.NewRecorder()

		interceptor, err := provider.CreateInterceptor(w, req, testTracer)

		require.NoError(t, err)
		require.NotNil(t, interceptor)
		assert.False(t, interceptor.Streaming())
	})

	t.Run("Responses_StreamingRequest_StreamingInterceptor", func(t *testing.T) {
		t.Parallel()

		body := `{"model": "gpt-5-mini", "input": "hello", "stream": true}`
		req := httptest.NewRequest(http.MethodPost, routeCopilotResponses, bytes.NewBufferString(body))
		req.Header.Set("Authorization", "Bearer test-token")
		w := httptest.NewRecorder()

		interceptor, err := provider.CreateInterceptor(w, req, testTracer)

		require.NoError(t, err)
		require.NotNil(t, interceptor)
		assert.True(t, interceptor.Streaming())
	})

	t.Run("Responses_InvalidRequestBody", func(t *testing.T) {
		t.Parallel()

		body := `invalid json`
		req := httptest.NewRequest(http.MethodPost, routeCopilotResponses, bytes.NewBufferString(body))
		req.Header.Set("Authorization", "Bearer test-token")
		w := httptest.NewRecorder()

		interceptor, err := provider.CreateInterceptor(w, req, testTracer)

		require.Error(t, err)
		require.Nil(t, interceptor)
		assert.Contains(t, err.Error(), "unmarshal responses request body")
	})

	t.Run("UnknownRoute", func(t *testing.T) {
		t.Parallel()

		body := `{"model": "gpt-4.1", "messages": [{"role": "user", "content": "hello"}]}`
		req := httptest.NewRequest(http.MethodPost, "/copilot/unknown/route", bytes.NewBufferString(body))
		req.Header.Set("Authorization", "Bearer test-token")
		w := httptest.NewRecorder()

		interceptor, err := provider.CreateInterceptor(w, req, testTracer)

		require.ErrorIs(t, err, UnknownRoute)
		require.Nil(t, interceptor)
	})
}

// TestCopilot_ForwardsHeadersToUpstream verifies that custom client headers are forwarded
// to the upstream, while filtered headers (hop-by-hop) are stripped, Copilot-specific
// headers are always forwarded, and the per-user token is used for auth.
func TestCopilot_ForwardsHeadersToUpstream(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name         string
		route        string
		body         string
		mockResponse string
	}{
		{
			name:         "chat-completions",
			route:        routeCopilotChatCompletions,
			body:         `{"model":"gpt-4","messages":[{"role":"user","content":"hello"}],"stream":false}`,
			mockResponse: `{"id":"chatcmpl-123","object":"chat.completion","created":1677652288,"model":"gpt-4","choices":[{"index":0,"message":{"role":"assistant","content":"Hello!"},"finish_reason":"stop"}],"usage":{"prompt_tokens":9,"completion_tokens":12,"total_tokens":21}}`,
		},
		{
			name:         "responses",
			route:        routeCopilotResponses,
			body:         `{"model":"gpt-5-mini","input":"hello","stream":false}`,
			mockResponse: `{"id":"resp-123","object":"responses.response","created":1677652288,"model":"gpt-5-mini","output":[],"usage":{"input_tokens":5,"output_tokens":10,"total_tokens":15}}`,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			var receivedHeaders http.Header

			mockUpstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				receivedHeaders = r.Header.Clone()
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusOK)
				_, _ = w.Write([]byte(tc.mockResponse))
			}))
			t.Cleanup(mockUpstream.Close)

			p := NewCopilot(config.Copilot{
				BaseURL: mockUpstream.URL,
			})

			req := httptest.NewRequest(http.MethodPost, tc.route, bytes.NewBufferString(tc.body))
			req.Header.Set("Authorization", "Bearer user-token") // per-user key, must reach upstream
			req.Header.Set("Editor-Version", "vscode/1.85.0")    // Copilot-specific, must reach upstream
			req.Header.Set("Copilot-Integration-Id", "test-id")  // Copilot-specific, must reach upstream
			req.Header.Set("X-Custom-Header", "custom-value")    // should be forwarded
			req.Header.Set("X-Api-Key", "client-fake-key")       // should be stripped (re-injected by SDK)
			req.Header.Set("Upgrade", "websocket")               // should be stripped (hop-by-hop)
			w := httptest.NewRecorder()

			interceptor, err := p.CreateInterceptor(w, req, testTracer)
			require.NoError(t, err)
			require.NotNil(t, interceptor)

			interceptor.Setup(slog.Make(), &testutil.MockRecorder{}, nil)

			err = interceptor.ProcessRequest(w, httptest.NewRequest(http.MethodPost, tc.route, nil))
			require.NoError(t, err)

			// Copilot-specific headers must reach the upstream.
			assert.Equal(t, "vscode/1.85.0", receivedHeaders.Get("Editor-Version"))
			assert.Equal(t, "test-id", receivedHeaders.Get("Copilot-Integration-Id"))
			// Custom headers must reach the upstream.
			assert.Equal(t, "custom-value", receivedHeaders.Get("X-Custom-Header"))
			// Hop-by-hop headers must not reach the upstream.
			assert.Empty(t, receivedHeaders.Get("Upgrade"))
			// Per-user token must be used for auth as Copilot uses the client's token, not a global key.
			assert.Equal(t, "Bearer user-token", receivedHeaders.Get("Authorization"))
			// User-Agent must be set by the SDK, not forwarded from the client.
			assert.True(t, strings.HasPrefix(receivedHeaders.Get("User-Agent"), "OpenAI/Go"),
				"upstream User-Agent should be set by the SDK")
		})
	}
}

func Test_extractBearerToken(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "Empty",
			input:    "",
			expected: "",
		},
		{
			name:     "Whitespace",
			input:    " ",
			expected: "",
		},
		{
			name:     "InvalidFormat",
			input:    "some-token",
			expected: "",
		},
		{
			name:     "BearerOnly",
			input:    "Bearer",
			expected: "",
		},
		{
			name:     "Valid",
			input:    "Bearer my-secret-token",
			expected: "my-secret-token",
		},
		{
			name:     "BearerMixedCase",
			input:    "BeArEr my-secret-token",
			expected: "my-secret-token",
		},
		{
			name:     "LeadingWhitespace",
			input:    "  Bearer my-secret-token",
			expected: "my-secret-token",
		},
		{
			name:     "TrailingWhitespace",
			input:    "Bearer my-secret-token  ",
			expected: "my-secret-token",
		},
		{
			name:     "TooManyParts",
			input:    "Bearer token extra",
			expected: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			result := extractBearerToken(tt.input)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestExtractCopilotHeaders(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		headers  map[string]string
		expected map[string]string
	}{
		{
			name:     "all headers present",
			headers:  map[string]string{"Editor-Version": "vscode/1.85.0", "Copilot-Integration-Id": "some-id"},
			expected: map[string]string{"Editor-Version": "vscode/1.85.0", "Copilot-Integration-Id": "some-id"},
		},
		{
			name:     "some headers present",
			headers:  map[string]string{"Editor-Version": "vscode/1.85.0"},
			expected: map[string]string{"Editor-Version": "vscode/1.85.0"},
		},
		{
			name:     "no headers",
			headers:  map[string]string{},
			expected: map[string]string{},
		},
		{
			name:     "ignores other headers",
			headers:  map[string]string{"Editor-Version": "vscode/1.85.0", "Authorization": "Bearer token"},
			expected: map[string]string{"Editor-Version": "vscode/1.85.0"},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			req := httptest.NewRequest(http.MethodPost, "/", nil)
			for header, value := range tc.headers {
				req.Header.Set(header, value)
			}

			result := extractCopilotHeaders(req)
			assert.Equal(t, tc.expected, result)
		})
	}
}
