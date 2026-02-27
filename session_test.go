package aibridge

import (
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestGuessSessionID(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name      string
		client    Client
		body      string
		headers   map[string]string
		sessionID string
	}{
		// Claude Code.
		{
			name:      "claude_code_with_valid_session",
			client:    ClientClaudeCode,
			body:      `{"metadata":{"user_id":"user_abc123_account_456_session_f47ac10b-58cc-4372-a567-0e02b2c3d479"}}`,
			sessionID: "f47ac10b-58cc-4372-a567-0e02b2c3d479",
		},
		{
			name:      "claude_code_missing_metadata",
			client:    ClientClaudeCode,
			body:      `{"model":"claude-3"}`,
			sessionID: "",
		},
		{
			name:      "claude_code_missing_user_id",
			client:    ClientClaudeCode,
			body:      `{"metadata":{}}`,
			sessionID: "",
		},
		{
			name:      "claude_code_user_id_without_session",
			client:    ClientClaudeCode,
			body:      `{"metadata":{"user_id":"user_abc123_account_456"}}`,
			sessionID: "",
		},
		{
			name:      "claude_code_empty_body",
			client:    ClientClaudeCode,
			body:      ``,
			sessionID: "",
		},
		{
			name:      "claude_code_invalid_json",
			client:    ClientClaudeCode,
			body:      `not json at all`,
			sessionID: "",
		},
		// Codex.
		{
			name:      "codex_with_session_header",
			client:    ClientCodex,
			headers:   map[string]string{"session_id": "codex-session-123"},
			sessionID: "codex-session-123",
		},
		{
			name:      "codex_with_whitespace_in_header",
			client:    ClientCodex,
			headers:   map[string]string{"session_id": "  codex-session-123  "},
			sessionID: "codex-session-123",
		},
		{
			name:      "codex_without_session_header",
			client:    ClientCodex,
			sessionID: "",
		},
		// Other clients shouldn't use others' logic.
		{
			name:      "unknown_client_returns_empty",
			client:    ClientUnknown,
			body:      `{"metadata":{"user_id":"user_abc_account_456_session_some-id"}}`,
			sessionID: "",
		},
		{
			name:      "zed_returns_empty",
			client:    ClientZed,
			headers:   map[string]string{"session_id": "zed-session"},
			body:      `{"metadata":{"user_id":"user_abc_account_456_session_some-id"}}`,
			sessionID: "",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			body := tc.body
			req, err := http.NewRequest(http.MethodPost, "http://localhost", strings.NewReader(body))
			require.NoError(t, err)

			for key, value := range tc.headers {
				req.Header.Set(key, value)
			}

			got := guessSessionID(tc.client, req)
			require.Equal(t, tc.sessionID, got)

			// Verify the body was restored and can be read again.
			restored, err := io.ReadAll(req.Body)
			require.NoError(t, err)
			require.Equal(t, body, string(restored))
		})
	}
}

func TestUnreadableBody(t *testing.T) {
	t.Parallel()

	req, err := http.NewRequest(http.MethodPost, "http://localhost", &errReader{})
	require.NoError(t, err)

	got := guessSessionID(ClientClaudeCode, req)
	require.Equal(t, "", got)
}

// errReader is an io.Reader that always returns an error.
type errReader struct{}

func (e *errReader) Read([]byte) (int, error) {
	return 0, io.ErrUnexpectedEOF
}
