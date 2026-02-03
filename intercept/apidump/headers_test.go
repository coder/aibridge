package apidump

import (
	"bytes"
	"net/http"
	"testing"

	"cdr.dev/slog/v3"

	"github.com/coder/quartz"
	"github.com/google/uuid"
	"github.com/stretchr/testify/require"
)

func TestRedactHeaderValue(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "empty string",
			input:    "",
			expected: "",
		},
		{
			name:     "single char",
			input:    "a",
			expected: "a",
		},
		{
			name:     "two chars",
			input:    "ab",
			expected: "a...b",
		},
		{
			name:     "seven chars",
			input:    "abcdefg",
			expected: "a...g",
		},
		{
			name:     "eight chars - threshold",
			input:    "abcdefgh",
			expected: "abcd...efgh",
		},
		{
			name:     "long value",
			input:    "Bearer sk-secret-key-12345",
			expected: "Bear...2345",
		},
		{
			name:     "realistic api key",
			input:    "sk-proj-abc123xyz789",
			expected: "sk-p...z789",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			result := redactHeaderValue(tc.input)
			require.Equal(t, tc.expected, result)
		})
	}
}

func TestSensitiveHeaderLists(t *testing.T) {
	t.Parallel()

	// Verify all expected sensitive request headers are in the list
	expectedRequestHeaders := []string{
		"Authorization",
		"X-Api-Key",
		"Api-Key",
		"X-Auth-Token",
		"Cookie",
		"Proxy-Authorization",
		"X-Amz-Security-Token",
	}
	for _, h := range expectedRequestHeaders {
		_, ok := sensitiveRequestHeaders[h]
		require.True(t, ok, "expected %q to be in sensitiveRequestHeaders", h)
	}

	// Verify all expected sensitive response headers are in the list
	// Note: header names use Go's canonical form (http.CanonicalHeaderKey)
	expectedResponseHeaders := []string{
		"Set-Cookie",
		"Www-Authenticate",
		"Proxy-Authenticate",
	}
	for _, h := range expectedResponseHeaders {
		_, ok := sensitiveResponseHeaders[h]
		require.True(t, ok, "expected %q to be in sensitiveResponseHeaders", h)
	}
}

func TestWriteRedactedHeaders(t *testing.T) {
	t.Parallel()

	d := &dumper{
		baseDir:        "/tmp",
		provider:       "test",
		model:          "test",
		interceptionID: uuid.New(),
		clk:            quartz.NewMock(t),
		logger:         slog.Make(),
	}

	tests := []struct {
		name      string
		headers   http.Header
		sensitive map[string]struct{}
		overrides map[string]string
		expected  string
	}{
		{
			name:     "empty headers",
			headers:  http.Header{},
			expected: "",
		},
		{
			name:     "single header",
			headers:  http.Header{"Content-Type": {"application/json"}},
			expected: "Content-Type: application/json\r\n",
		},
		{
			name: "sorted alphabetically",
			headers: http.Header{
				"Zebra": {"last"},
				"Alpha": {"first"},
			},
			expected: "Alpha: first\r\nZebra: last\r\n",
		},
		{
			name:      "override applied",
			headers:   http.Header{"Content-Length": {"100"}},
			overrides: map[string]string{"Content-Length": "200"},
			expected:  "Content-Length: 200\r\n",
		},
		{
			name:      "sensitive header redacted",
			headers:   http.Header{"Set-Cookie": {"session=abcdefghij"}},
			sensitive: sensitiveResponseHeaders,
			expected:  "Set-Cookie: sess...ghij\r\n",
		},
		{
			name: "multi-value header",
			headers: http.Header{
				"Accept": {"text/html", "application/json"},
			},
			expected: "Accept: text/html\r\nAccept: application/json\r\n",
		},
		{
			name:      "override for non-existent header",
			headers:   http.Header{},
			overrides: map[string]string{"Host": "example.com"},
			expected:  "Host: example.com\r\n",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			var buf bytes.Buffer
			d.writeRedactedHeaders(&buf, tc.headers, tc.sensitive, tc.overrides)
			require.Equal(t, tc.expected, buf.String())
		})
	}
}
