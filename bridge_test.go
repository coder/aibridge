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
