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

func TestValidateProviders(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name      string
		providers []provider.Provider
		expectErr string
	}{
		{
			name: "all_supported_providers",
			providers: []provider.Provider{
				NewOpenAIProvider(config.OpenAI{Name: "openai", BaseURL: "https://api.openai.com/v1/"}),
				NewAnthropicProvider(config.Anthropic{Name: "anthropic", BaseURL: "https://api.anthropic.com/"}, nil),
				NewCopilotProvider(config.Copilot{Name: "copilot", BaseURL: "https://api.individual.githubcopilot.com"}),
				NewCopilotProvider(config.Copilot{Name: "copilot-business", BaseURL: "https://api.business.githubcopilot.com"}),
				NewCopilotProvider(config.Copilot{Name: "copilot-enterprise", BaseURL: "https://api.enterprise.githubcopilot.com"}),
			},
		},
		{
			name: "default_names_and_base_urls",
			providers: []provider.Provider{
				NewOpenAIProvider(config.OpenAI{}),
				NewAnthropicProvider(config.Anthropic{}, nil),
				NewCopilotProvider(config.Copilot{}),
			},
		},
		{
			name: "multiple_copilot_instances",
			providers: []provider.Provider{
				NewCopilotProvider(config.Copilot{}),
				NewCopilotProvider(config.Copilot{Name: "copilot-business", BaseURL: "https://api.business.githubcopilot.com"}),
				NewCopilotProvider(config.Copilot{Name: "copilot-enterprise", BaseURL: "https://api.enterprise.githubcopilot.com"}),
			},
		},
		{
			name: "duplicate_name",
			providers: []provider.Provider{
				NewCopilotProvider(config.Copilot{Name: "copilot", BaseURL: "https://api.individual.githubcopilot.com"}),
				NewCopilotProvider(config.Copilot{Name: "copilot", BaseURL: "https://api.business.githubcopilot.com"}),
			},
			expectErr: "duplicate provider name",
		},
		{
			name: "duplicate_base_url_different_names",
			providers: []provider.Provider{
				NewCopilotProvider(config.Copilot{Name: "copilot", BaseURL: "https://api.individual.githubcopilot.com"}),
				NewCopilotProvider(config.Copilot{Name: "copilot-business", BaseURL: "https://api.individual.githubcopilot.com"}),
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			err := validateProviders(tc.providers)
			if tc.expectErr != "" {
				require.Error(t, err)
				assert.Contains(t, err.Error(), tc.expectErr)
			} else {
				require.NoError(t, err)
			}
		})
	}
}

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
