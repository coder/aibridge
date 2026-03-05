package integrationtest

import (
	"bytes"
	"net/http"
	"testing"

	"github.com/coder/aibridge/config"
	"github.com/stretchr/testify/require"
)

// apiKey is the default API key used across integration tests.
const apiKey = "api-key"

// createAnthropicMessagesReq builds an HTTP request targeting the Anthropic messages endpoint.
func createAnthropicMessagesReq(t *testing.T, baseURL string, input []byte) *http.Request {
	t.Helper()

	req, err := http.NewRequestWithContext(t.Context(), "POST", baseURL+"/anthropic/v1/messages", bytes.NewReader(input))
	require.NoError(t, err)
	req.Header.Set("Content-Type", "application/json")

	return req
}

// createOpenAIChatCompletionsReq builds an HTTP request targeting the OpenAI chat completions endpoint.
func createOpenAIChatCompletionsReq(t *testing.T, baseURL string, input []byte) *http.Request {
	t.Helper()

	req, err := http.NewRequestWithContext(t.Context(), "POST", baseURL+"/openai/v1/chat/completions", bytes.NewReader(input))
	require.NoError(t, err)
	req.Header.Set("Content-Type", "application/json")

	return req
}

// createOpenAIResponsesReq builds an HTTP request targeting the OpenAI responses endpoint.
func createOpenAIResponsesReq(t *testing.T, baseURL string, input []byte) *http.Request {
	t.Helper()

	req, err := http.NewRequestWithContext(t.Context(), "POST", baseURL+"/openai/v1/responses", bytes.NewReader(input))
	require.NoError(t, err)
	req.Header.Set("Content-Type", "application/json")
	return req
}

// openAICfg creates a minimal OpenAI config for testing.
func openAICfg(url, key string) config.OpenAI {
	return config.OpenAI{
		BaseURL: url,
		Key:     key,
	}
}

// anthropicCfg creates a minimal Anthropic config for testing.
func anthropicCfg(url, key string) config.Anthropic {
	return config.Anthropic{
		BaseURL: url,
		Key:     key,
	}
}
