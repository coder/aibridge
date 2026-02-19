package aibridge

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"cdr.dev/slog/v3/sloggers/slogtest"
	"github.com/coder/aibridge/config"
	"github.com/coder/aibridge/internal/testutil"
	"github.com/coder/aibridge/provider"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestPassthroughRoutesForProviders(t *testing.T) {
	t.Parallel()

	upstreamRespBody := "upstream response"
	tests := []struct {
		name        string
		baseURLPath string
		requestPath string
		provider    func(string) provider.Provider
		expectPath  string
	}{
		{
			name:        "openAI_no_base_path",
			requestPath: "/openai/v1/conversations",
			provider: func(baseURL string) provider.Provider {
				return NewOpenAIProvider(config.OpenAI{BaseURL: baseURL})
			},
			expectPath: "/conversations",
		},
		{
			name:        "openAI_with_base_path",
			baseURLPath: "/v1",
			requestPath: "/openai/v1/conversations",
			provider: func(baseURL string) provider.Provider {
				return NewOpenAIProvider(config.OpenAI{BaseURL: baseURL})
			},
			expectPath: "/v1/conversations",
		},
		{
			name:        "anthropic_no_base_path",
			requestPath: "/anthropic/v1/models",
			provider: func(baseURL string) provider.Provider {
				return NewAnthropicProvider(config.Anthropic{BaseURL: baseURL}, nil)
			},
			expectPath: "/v1/models",
		},
		{
			name:        "anthropic_with_base_path",
			baseURLPath: "/v1",
			requestPath: "/anthropic/v1/models",
			provider: func(baseURL string) provider.Provider {
				return NewAnthropicProvider(config.Anthropic{BaseURL: baseURL}, nil)
			},
			expectPath: "/v1/v1/models",
		},
		{
			name:        "copilot_no_base_path",
			requestPath: "/copilot/models",
			provider: func(baseURL string) provider.Provider {
				return NewCopilotProvider(config.Copilot{BaseURL: baseURL})
			},
			expectPath: "/models",
		},
		{
			name:        "copilot_with_base_path",
			baseURLPath: "/v1",
			requestPath: "/copilot/models",
			provider: func(baseURL string) provider.Provider {
				return NewCopilotProvider(config.Copilot{BaseURL: baseURL})
			},
			expectPath: "/v1/models",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			logger := slogtest.Make(t, nil)

			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				assert.Equal(t, tc.expectPath, r.URL.Path)
				w.WriteHeader(http.StatusOK)
				_, _ = w.Write([]byte(upstreamRespBody))
			}))
			t.Cleanup(upstream.Close)

			recorder := testutil.MockRecorder{}
			prov := tc.provider(upstream.URL + tc.baseURLPath)
			bridge, err := NewRequestBridge(t.Context(), []provider.Provider{prov}, &recorder, nil, logger, nil, testTracer)
			require.NoError(t, err)

			req := httptest.NewRequest("", tc.requestPath, nil)
			resp := httptest.NewRecorder()
			bridge.mux.ServeHTTP(resp, req)

			assert.Equal(t, http.StatusOK, resp.Code)
			assert.Contains(t, resp.Body.String(), upstreamRespBody)
		})
	}
}

func TestGuessClient(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name       string
		userAgent  string
		headers    map[string]string
		wantClient string
	}{
		{
			name:       "claude_code",
			userAgent:  "claude-cli/2.0.67 (external, cli)",
			wantClient: ClientClaude,
		},
		{
			name:       "codex_cli",
			userAgent:  "codex_cli_rs/0.87.0 (Mac OS 26.2.0; arm64) ghostty/1.3.0-main_250877ef",
			wantClient: ClientCodex,
		},
		{
			name:       "zed",
			userAgent:  "Zed/0.219.4+stable.119.abc123 (macos; aarch64)",
			wantClient: ClientZed,
		},
		{
			name:       "github_copilot_vsc",
			userAgent:  "GitHubCopilotChat/0.37.2026011603",
			wantClient: ClientCopilotVSC,
		},
		{
			name:       "github_copilot_cli",
			userAgent:  "copilot/0.0.403 (client/cli linux v24.11.1)",
			wantClient: ClientCopilotCLI,
		},
		{
			name:       "kilo_code_user_agent",
			userAgent:  "kilo-code/5.1.0 (darwin 25.2.0; arm64) node/22.21.1",
			wantClient: ClientKilo,
		},
		{
			name:       "kilo_code_originator",
			headers:    map[string]string{"Originator": "kilo-code"},
			wantClient: ClientKilo,
		},
		{
			name:       "roo_code_user_agent",
			userAgent:  "roo-code/3.45.0 (darwin 25.2.0; arm64) node/22.21.1",
			wantClient: ClientRoo,
		},
		{
			name:       "roo_code_originator",
			headers:    map[string]string{"Originator": "roo-code"},
			wantClient: ClientRoo,
		},
		{
			name:       "cursor_x_cursor_client_version",
			userAgent:  "connect-es/1.6.1",
			headers:    map[string]string{"X-Cursor-client-version": "0.50.0"},
			wantClient: ClientCursor,
		},
		{
			name:       "cursor_x_cursor_some_other_header",
			headers:    map[string]string{"x-cursor-client-version": "abc123"},
			wantClient: ClientCursor,
		},
		{
			name:       "unknown_client",
			userAgent:  "ccclaude-cli/calude-with-wrong-prefix",
			wantClient: ClientUnknown,
		},
		{
			name:       "empty_user_agent",
			userAgent:  "",
			wantClient: ClientUnknown,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			req, err := http.NewRequest(http.MethodGet, "", nil)
			require.NoError(t, err)

			req.Header.Set("User-Agent", tt.userAgent)
			for key, value := range tt.headers {
				req.Header.Set(key, value)
			}

			got := guessClient(req)
			require.Equal(t, tt.wantClient, got)
		})
	}
}
