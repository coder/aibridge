package aibridge

import (
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/coder/aibridge/utils"
	"github.com/stretchr/testify/require"
)

func TestGuessSessionID(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name      string
		client    Client
		body      string
		headers   map[string]string
		sessionID *string
	}{
		// Claude Code.
		{
			name:      "claude_code_with_valid_session",
			client:    ClientClaudeCode,
			body:      `{"metadata":{"user_id":"user_abc123_account_456_session_f47ac10b-58cc-4372-a567-0e02b2c3d479"}}`,
			sessionID: utils.PtrTo("f47ac10b-58cc-4372-a567-0e02b2c3d479"),
		},
		{
			name:      "claude_code_with_valid_session_new_format",
			client:    ClientClaudeCode,
			body:      `{"metadata":{"user_id":"{\"device_id\":\"45aa15c8c244ea2582f8144dde91a50ec3815851f6f648abef4ee15b173cc927\",\"account_uuid\":\"\",\"session_id\":\"54c1eb09-bc4c-4d2f-98eb-6d2ab2d5e2fe\"}"}}`,
			sessionID: utils.PtrTo("54c1eb09-bc4c-4d2f-98eb-6d2ab2d5e2fe"),
		},
		{
			name:   "claude_code_new_format_empty_session_id",
			client: ClientClaudeCode,
			body:   `{"metadata":{"user_id":"{\"device_id\":\"abc\",\"account_uuid\":\"\",\"session_id\":\"\"}"}}`,
		},
		{
			name:   "claude_code_new_format_no_session_id_field",
			client: ClientClaudeCode,
			body:   `{"metadata":{"user_id":"{\"device_id\":\"abc\",\"account_uuid\":\"\"}"}}`,
		},
		{
			name:   "claude_code_missing_metadata",
			client: ClientClaudeCode,
			body:   `{"model":"claude-3"}`,
		},
		{
			name:   "claude_code_missing_user_id",
			client: ClientClaudeCode,
			body:   `{"metadata":{}}`,
		},
		{
			name:   "claude_code_user_id_without_session",
			client: ClientClaudeCode,
			body:   `{"metadata":{"user_id":"user_abc123_account_456"}}`,
		},
		{
			name:   "claude_code_empty_body",
			client: ClientClaudeCode,
			body:   ``,
		},
		{
			name:   "claude_code_invalid_json",
			client: ClientClaudeCode,
			body:   `not json at all`,
		},
		// Codex.
		{
			name:      "codex_with_session_header",
			client:    ClientCodex,
			headers:   map[string]string{"session_id": "codex-session-123"},
			sessionID: utils.PtrTo("codex-session-123"),
		},
		{
			name:      "codex_with_whitespace_in_header",
			client:    ClientCodex,
			headers:   map[string]string{"session_id": "  codex-session-123  "},
			sessionID: utils.PtrTo("codex-session-123"),
		},
		{
			name:   "codex_without_session_header",
			client: ClientCodex,
		},
		// Other clients shouldn't use others' logic.
		{
			name:   "unknown_client_returns_empty",
			client: ClientUnknown,
			body:   `{"metadata":{"user_id":"user_abc_account_456_session_some-id"}}`,
		},
		{
			name:    "zed_returns_empty",
			client:  ClientZed,
			headers: map[string]string{"session_id": "zed-session"},
			body:    `{"metadata":{"user_id":"user_abc_account_456_session_some-id"}}`,
		},
		// Mux.
		{
			name:      "mux_with_workspace_header",
			client:    ClientMux,
			headers:   map[string]string{"X-Mux-Workspace-Id": "ws-abc-123"},
			sessionID: utils.PtrTo("ws-abc-123"),
		},
		{
			name:   "mux_without_workspace_header",
			client: ClientMux,
		},
		// Copilot VS Code.
		{
			name:      "copilot_vsc_with_interaction_id",
			client:    ClientCopilotVSC,
			headers:   map[string]string{"x-interaction-id": "interaction-xyz"},
			sessionID: utils.PtrTo("interaction-xyz"),
		},
		{
			name:   "copilot_vsc_without_interaction_id",
			client: ClientCopilotVSC,
		},
		// Copilot CLI.
		{
			name:      "copilot_cli_with_session_header",
			client:    ClientCopilotCLI,
			headers:   map[string]string{"X-Client-Session-Id": "cli-sess-456"},
			sessionID: utils.PtrTo("cli-sess-456"),
		},
		{
			name:   "copilot_cli_without_session_header",
			client: ClientCopilotCLI,
		},
		// Kilo.
		{
			name:      "kilo_with_task_id",
			client:    ClientKilo,
			headers:   map[string]string{"X-KILOCODE-TASKID": "task-789"},
			sessionID: utils.PtrTo("task-789"),
		},
		{
			name:   "kilo_without_task_id",
			client: ClientKilo,
		},
		// Coder Agents.
		{
			name:   "coder_agents_returns_empty",
			client: ClientCoderAgents,
		},
		// Roo.
		{
			name:   "roo_returns_empty",
			client: ClientRoo,
		},
		// Cursor.
		{
			name:   "cursor_returns_empty",
			client: ClientCursor,
		},
		// Other cases.
		{
			name:      "empty session ID value",
			client:    ClientKilo,
			headers:   map[string]string{"X-KILOCODE-TASKID": " "},
			sessionID: nil,
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
	require.Nil(t, got)
}

// errReader is an io.Reader that always returns an error.
type errReader struct{}

func (e *errReader) Read([]byte) (int, error) {
	return 0, io.ErrUnexpectedEOF
}
