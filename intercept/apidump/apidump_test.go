package apidump

import (
	"bytes"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"testing"

	"cdr.dev/slog/v3"
	"cdr.dev/slog/v3/sloggers/slogtest"
	"github.com/coder/quartz"
	"github.com/google/uuid"
	"github.com/stretchr/testify/require"
)

// findDumpFile finds a dump file matching the pattern in the given directory.
func findDumpFile(t *testing.T, dir, suffix string) string {
	t.Helper()
	pattern := filepath.Join(dir, "*"+suffix)
	matches, err := filepath.Glob(pattern)
	require.NoError(t, err)
	require.Len(t, matches, 1, "expected exactly one %s file in %s", suffix, dir)
	return matches[0]
}

func TestMiddleware_RedactsSensitiveRequestHeaders(t *testing.T) {
	t.Parallel()

	tmpDir := t.TempDir()
	logger := slogtest.Make(t, &slogtest.Options{IgnoreErrors: false}).Leveled(slog.LevelDebug)
	clk := quartz.NewMock(t)
	interceptionID := uuid.New()

	middleware := NewMiddleware(tmpDir, "openai", "gpt-4", interceptionID, logger, clk)
	require.NotNil(t, middleware)

	req, err := http.NewRequest(http.MethodPost, "https://api.openai.com/v1/chat/completions", bytes.NewReader([]byte(`{"test": true}`)))
	require.NoError(t, err)

	// Add sensitive headers that should be redacted
	req.Header.Set("Authorization", "Bearer sk-secret-key-12345")
	req.Header.Set("X-Api-Key", "secret-api-key-value")
	req.Header.Set("Cookie", "session=abc123")

	// Add non-sensitive headers that should be kept as-is
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("User-Agent", "test-client")

	// Call middleware with a mock next function
	_, err = middleware(req, func(r *http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode: http.StatusOK,
			Status:     "200 OK",
			Proto:      "HTTP/1.1",
			Header:     http.Header{"Content-Type": []string{"application/json"}},
			Body:       io.NopCloser(bytes.NewReader([]byte(`{"ok": true}`))),
		}, nil
	})
	require.NoError(t, err)

	// Read the request dump file
	modelDir := filepath.Join(tmpDir, "openai", "gpt-4")
	reqDumpPath := findDumpFile(t, modelDir, SuffixRequest)
	reqContent, err := os.ReadFile(reqDumpPath)
	require.NoError(t, err)

	content := string(reqContent)

	// Verify sensitive headers ARE present but redacted
	require.Contains(t, content, "Authorization: Bear...2345")
	require.Contains(t, content, "X-Api-Key: secr...alue")
	require.Contains(t, content, "Cookie: sess...c123") // "session=abc123" is 14 chars, so first 4 + last 4

	// Verify the full secret values are NOT present
	require.NotContains(t, content, "sk-secret-key-12345")
	require.NotContains(t, content, "secret-api-key-value")

	// Verify non-sensitive headers ARE present in full
	require.Contains(t, content, "Content-Type: application/json")
	require.Contains(t, content, "User-Agent: test-client")
}

func TestMiddleware_RedactsSensitiveResponseHeaders(t *testing.T) {
	t.Parallel()

	tmpDir := t.TempDir()
	logger := slogtest.Make(t, &slogtest.Options{IgnoreErrors: false}).Leveled(slog.LevelDebug)
	clk := quartz.NewMock(t)
	interceptionID := uuid.New()

	middleware := NewMiddleware(tmpDir, "openai", "gpt-4", interceptionID, logger, clk)
	require.NotNil(t, middleware)

	req, err := http.NewRequest(http.MethodPost, "https://api.openai.com/v1/chat/completions", bytes.NewReader([]byte(`{}`)))
	require.NoError(t, err)

	// Call middleware with a response containing sensitive headers
	resp, err := middleware(req, func(r *http.Request) (*http.Response, error) {
		resp := &http.Response{
			StatusCode: http.StatusOK,
			Status:     "200 OK",
			Proto:      "HTTP/1.1",
			Header:     make(http.Header),
			Body:       io.NopCloser(bytes.NewReader([]byte(`{"ok": true}`))),
		}
		// Add sensitive response headers
		resp.Header.Set("Set-Cookie", "session=secret123; HttpOnly; Secure")
		resp.Header.Set("WWW-Authenticate", "Bearer realm=\"api\"")
		// Add non-sensitive headers
		resp.Header.Set("Content-Type", "application/json")
		resp.Header.Set("X-Request-Id", "req-123")
		return resp, nil
	})
	require.NoError(t, err)

	// Must read and close response body to trigger the streaming dump
	_, err = io.ReadAll(resp.Body)
	require.NoError(t, err)
	require.NoError(t, resp.Body.Close())

	// Read the response dump file
	modelDir := filepath.Join(tmpDir, "openai", "gpt-4")
	respDumpPath := findDumpFile(t, modelDir, SuffixResponse)
	respContent, err := os.ReadFile(respDumpPath)
	require.NoError(t, err)

	content := string(respContent)

	// Verify sensitive headers are present but redacted
	require.Contains(t, content, "Set-Cookie: sess...cure")
	// Note: Go canonicalizes WWW-Authenticate to Www-Authenticate
	// "Bearer realm=\"api\"" = 18 chars, first 4 = "Bear", last 4 = "api\""
	require.Contains(t, content, "Www-Authenticate: Bear...api\"")

	// Verify full secret values are NOT present
	require.NotContains(t, content, "secret123")
	require.NotContains(t, content, "realm=\"api\"")

	// Verify non-sensitive headers ARE present in full
	require.Contains(t, content, "Content-Type: application/json")
	require.Contains(t, content, "X-Request-Id: req-123")
}

func TestMiddleware_EmptyBaseDir_ReturnsNil(t *testing.T) {
	t.Parallel()

	logger := slogtest.Make(t, &slogtest.Options{IgnoreErrors: false}).Leveled(slog.LevelDebug)
	middleware := NewMiddleware("", "openai", "gpt-4", uuid.New(), logger, quartz.NewMock(t))
	require.Nil(t, middleware)
}

func TestMiddleware_PreservesRequestBody(t *testing.T) {
	t.Parallel()

	tmpDir := t.TempDir()
	logger := slogtest.Make(t, &slogtest.Options{IgnoreErrors: false}).Leveled(slog.LevelDebug)
	clk := quartz.NewMock(t)
	interceptionID := uuid.New()

	middleware := NewMiddleware(tmpDir, "openai", "gpt-4", interceptionID, logger, clk)
	require.NotNil(t, middleware)

	originalBody := `{"messages": [{"role": "user", "content": "hello"}]}`
	req, err := http.NewRequest(http.MethodPost, "https://api.openai.com/v1/chat/completions", bytes.NewReader([]byte(originalBody)))
	require.NoError(t, err)

	var capturedBody []byte
	_, err = middleware(req, func(r *http.Request) (*http.Response, error) {
		// Read the body in the next handler to verify it's still available
		capturedBody, _ = io.ReadAll(r.Body)
		return &http.Response{
			StatusCode: http.StatusOK,
			Status:     "200 OK",
			Proto:      "HTTP/1.1",
			Header:     http.Header{},
			Body:       io.NopCloser(bytes.NewReader([]byte(`{}`))),
		}, nil
	})
	require.NoError(t, err)

	// Verify the body was preserved for the next handler
	require.Equal(t, originalBody, string(capturedBody))
}

func TestMiddleware_ModelWithSlash(t *testing.T) {
	t.Parallel()

	tmpDir := t.TempDir()
	logger := slogtest.Make(t, &slogtest.Options{IgnoreErrors: false}).Leveled(slog.LevelDebug)
	clk := quartz.NewMock(t)
	interceptionID := uuid.New()

	// Model with slash should have it replaced with dash
	middleware := NewMiddleware(tmpDir, "google", "gemini/1.5-pro", interceptionID, logger, clk)
	require.NotNil(t, middleware)

	req, err := http.NewRequest(http.MethodPost, "https://api.google.com/v1/chat", bytes.NewReader([]byte(`{}`)))
	require.NoError(t, err)

	_, err = middleware(req, func(r *http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode: http.StatusOK,
			Status:     "200 OK",
			Proto:      "HTTP/1.1",
			Header:     http.Header{},
			Body:       io.NopCloser(bytes.NewReader([]byte(`{}`))),
		}, nil
	})
	require.NoError(t, err)

	// Verify files are created with sanitized model name
	modelDir := filepath.Join(tmpDir, "google", "gemini-1.5-pro")
	reqDumpPath := findDumpFile(t, modelDir, SuffixRequest)
	_, err = os.Stat(reqDumpPath)
	require.NoError(t, err)
}

func TestPrettyPrintJSON(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		input    []byte
		expected string
	}{
		{
			name:     "empty",
			input:    []byte{},
			expected: "",
		},
		{
			name:     "valid JSON",
			input:    []byte(`{"key":"value"}`),
			expected: "{\n  \"key\": \"value\"\n}",
		},
		{
			name:     "invalid JSON returns as-is",
			input:    []byte("not json"),
			expected: "not json",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			result := prettyPrintJSON(tc.input)
			require.Equal(t, tc.expected, string(result))
		})
	}
}

func TestMiddleware_AllSensitiveRequestHeaders(t *testing.T) {
	t.Parallel()

	tmpDir := t.TempDir()
	logger := slogtest.Make(t, &slogtest.Options{IgnoreErrors: false}).Leveled(slog.LevelDebug)
	clk := quartz.NewMock(t)
	interceptionID := uuid.New()

	middleware := NewMiddleware(tmpDir, "openai", "gpt-4", interceptionID, logger, clk)
	require.NotNil(t, middleware)

	req, err := http.NewRequest(http.MethodPost, "https://api.openai.com/v1/chat/completions", bytes.NewReader([]byte(`{}`)))
	require.NoError(t, err)

	// Set all sensitive headers
	req.Header.Set("Authorization", "Bearer sk-secret-key")
	req.Header.Set("X-Api-Key", "secret-api-key")
	req.Header.Set("Api-Key", "another-secret")
	req.Header.Set("X-Auth-Token", "auth-token-val")
	req.Header.Set("Cookie", "session=abc123def")
	req.Header.Set("Proxy-Authorization", "Basic proxy-creds")
	req.Header.Set("X-Amz-Security-Token", "aws-security-token")

	_, err = middleware(req, func(r *http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode: http.StatusOK,
			Status:     "200 OK",
			Proto:      "HTTP/1.1",
			Header:     http.Header{},
			Body:       io.NopCloser(bytes.NewReader([]byte(`{}`))),
		}, nil
	})
	require.NoError(t, err)

	modelDir := filepath.Join(tmpDir, "openai", "gpt-4")
	reqDumpPath := findDumpFile(t, modelDir, SuffixRequest)
	reqContent, err := os.ReadFile(reqDumpPath)
	require.NoError(t, err)

	content := string(reqContent)

	// Verify none of the full secret values are present
	require.NotContains(t, content, "sk-secret-key")
	require.NotContains(t, content, "secret-api-key")
	require.NotContains(t, content, "another-secret")
	require.NotContains(t, content, "auth-token-val")
	require.NotContains(t, content, "abc123def")
	require.NotContains(t, content, "proxy-creds")
	require.NotContains(t, content, "aws-security-token")
	require.NotContains(t, content, "google-api-key")

	// But headers themselves are present (redacted)
	require.Contains(t, content, "Authorization:")
	require.Contains(t, content, "X-Api-Key:")
	require.Contains(t, content, "Api-Key:")
	require.Contains(t, content, "X-Auth-Token:")
	require.Contains(t, content, "Cookie:")
	require.Contains(t, content, "Proxy-Authorization:")
	require.Contains(t, content, "X-Amz-Security-Token:")
}
