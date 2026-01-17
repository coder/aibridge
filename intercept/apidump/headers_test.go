package apidump

import (
	"testing"

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
