package intercept

import (
	"net/http"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestSanitizeClientHeaders(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name          string
		input         http.Header
		expectAbsent  []string
		expectPresent map[string][]string
		expectEmpty   bool
	}{
		{
			name:        "no headers returns empty header",
			input:       nil,
			expectEmpty: true,
		},
		{
			name: "hop-by-hop headers are removed",
			input: http.Header{
				"Connection":        []string{"keep-alive"},
				"Keep-Alive":        []string{"timeout=5"},
				"Transfer-Encoding": []string{"chunked"},
				"Upgrade":           []string{"websocket"},
			},
			expectAbsent: []string{"Connection", "Keep-Alive", "Transfer-Encoding", "Upgrade"},
		},
		{
			name: "bridge headers are removed",
			input: http.Header{
				"Host":            []string{"example.com"},
				"Accept-Encoding": []string{"gzip"},
				"Content-Length":  []string{"42"},
				"Content-Type":    []string{"application/json"},
				"User-Agent":      []string{"client/1.0"},
			},
			expectAbsent: []string{"Host", "Accept-Encoding", "Content-Length", "Content-Type", "User-Agent"},
		},
		{
			name: "auth headers are removed",
			input: http.Header{
				"Authorization": []string{"Bearer some-token"},
				"X-Api-Key":     []string{"some-key"},
			},
			expectAbsent: []string{"Authorization", "X-Api-Key"},
		},
		{
			name: "custom headers are preserved",
			input: http.Header{
				"X-Custom-Header": []string{"custom-value"},
				"X-Request-Id":    []string{"req-123"},
			},
			expectPresent: map[string][]string{
				"X-Custom-Header": {"custom-value"},
				"X-Request-Id":    {"req-123"},
			},
		},
		{
			name: "multi-value headers are preserved",
			input: http.Header{
				"X-Custom-Header": []string{"value-1", "value-2"},
			},
			expectPresent: map[string][]string{
				"X-Custom-Header": {"value-1", "value-2"},
			},
		},
		{
			name: "input is not mutated",
			input: http.Header{
				"Connection":      []string{"keep-alive"},
				"X-Custom-Header": []string{"custom-value"},
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			// Capture original state to verify no mutation.
			originalCopy := tc.input.Clone()

			result := SanitizeClientHeaders(tc.input)

			if tc.expectEmpty {
				require.Empty(t, result)
				return
			}

			for _, h := range tc.expectAbsent {
				require.Empty(t, result.Get(h), "expected header %q to be absent", h)
			}

			for h, vals := range tc.expectPresent {
				require.Equal(t, vals, result[h], "expected header %q to be present", h)
			}

			// Verify input was not mutated.
			require.Equal(t, originalCopy, tc.input)
		})
	}
}
