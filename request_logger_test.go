package aibridge_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"cdr.dev/slog"
	"cdr.dev/slog/sloggers/slogtest"
	"github.com/coder/aibridge"
	"github.com/coder/aibridge/mcp"
	"github.com/stretchr/testify/require"
	"golang.org/x/tools/txtar"
)

func TestRequestLogging(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		provider       string
		fixture        []byte
		route          string
		createProvider func(*aibridge.AnthropicConfig) aibridge.Provider
	}{
		{
			provider: aibridge.ProviderAnthropic,
			fixture:  antSimple,
			route:    "/anthropic/v1/messages",
			createProvider: func(cfg *aibridge.AnthropicConfig) aibridge.Provider {
				return aibridge.NewAnthropicProvider(cfg, nil)
			},
		},
		{
			provider: aibridge.ProviderOpenAI,
			fixture:  oaiSimple,
			route:    "/openai/v1/chat/completions",
			createProvider: func(cfg *aibridge.AnthropicConfig) aibridge.Provider {
				return aibridge.NewOpenAIProvider(cfg)
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.provider, func(t *testing.T) {
			t.Parallel()

			logger := slogtest.Make(t, nil).Leveled(slog.LevelDebug)

			// Use a temp dir for this test
			tmpDir := t.TempDir()

			// Parse fixture
			arc := txtar.Parse(tc.fixture)
			files := filesMap(arc)
			require.Contains(t, files, fixtureRequest)
			require.Contains(t, files, fixtureNonStreamingResponse)

			// Create mock server
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusOK)
				_, _ = w.Write(files[fixtureNonStreamingResponse])
			}))
			t.Cleanup(srv.Close)

			cfg := aibridge.NewProviderConfig(srv.URL, apiKey, tmpDir)
			cfg.SetEnableUpstreamLogging(true)

			provider := tc.createProvider(cfg)
			client := &mockRecorderClient{}
			mcpProxy := mcp.NewServerProxyManager(nil, testTracer)

			bridge, err := aibridge.NewRequestBridge(context.Background(), []aibridge.Provider{provider}, client, mcpProxy, logger, nil, testTracer)
			require.NoError(t, err)
			t.Cleanup(func() {
				_ = bridge.Shutdown(context.Background())
			})

			// Make a request
			req, err := http.NewRequestWithContext(t.Context(), "POST", tc.route, strings.NewReader(string(files[fixtureRequest])))
			require.NoError(t, err)
			req.Header.Set("Content-Type", "application/json")
			req = req.WithContext(aibridge.AsActor(req.Context(), userID, nil))
			rec := httptest.NewRecorder()
			bridge.ServeHTTP(rec, req)
			require.Equal(t, 200, rec.Code)

			// Check that log files were created
			// Parse the request to get the model name
			var reqData map[string]any
			require.NoError(t, json.Unmarshal(files[fixtureRequest], &reqData))
			model := reqData["model"].(string)

			logDir := filepath.Join(tmpDir, tc.provider, model)
			entries, err := os.ReadDir(logDir)
			require.NoError(t, err, "log directory should exist")
			require.NotEmpty(t, entries, "log directory should contain files")

			// Should have at least one .req.log and one .res.log file
			var hasReq, hasRes bool
			for _, entry := range entries {
				name := entry.Name()
				if strings.HasSuffix(name, ".req.log") {
					hasReq = true
					// Verify the file has content
					content, err := os.ReadFile(filepath.Join(logDir, name))
					require.NoError(t, err)
					require.NotEmpty(t, content, "request log should have content")
					require.Contains(t, string(content), "POST")
				} else if strings.HasSuffix(name, ".res.log") {
					hasRes = true
					// Verify the file has content
					content, err := os.ReadFile(filepath.Join(logDir, name))
					require.NoError(t, err)
					require.NotEmpty(t, content, "response log should have content")
					require.Contains(t, string(content), "200")
				}
			}
			require.True(t, hasReq, "should have request log file")
			require.True(t, hasRes, "should have response log file")
		})
	}
}

func TestSanitizeModelName(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "simple model",
			input:    "gpt-4o",
			expected: "gpt-4o",
		},
		{
			name:     "model with slash",
			input:    "gpt-4o/mini",
			expected: "gpt-4o_mini",
		},
		{
			name:     "model with colon",
			input:    "o1:2024-12-17",
			expected: "o1_2024-12-17",
		},
		{
			name:     "model with backslash",
			input:    "model\\name",
			expected: "model_name",
		},
		{
			name:     "model with multiple special chars",
			input:    "model:name/version?",
			expected: "model_name_version_",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := aibridge.SanitizeModelName(tt.input)
			require.Equal(t, tt.expected, result)
		})
	}
}
