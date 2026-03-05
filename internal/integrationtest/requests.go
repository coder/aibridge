package integrationtest

import (
	"bytes"
	"net/http"
	"testing"

	"github.com/coder/aibridge/config"
	"github.com/stretchr/testify/require"
)

// APIKey is the default API key used across integration tests.
const APIKey = "api-key"

// CreateAnthropicMessagesReq builds an HTTP request targeting the Anthropic messages endpoint.
func CreateAnthropicMessagesReq(t *testing.T, baseURL string, input []byte) *http.Request {
	t.Helper()

	req, err := http.NewRequestWithContext(t.Context(), "POST", baseURL+"/anthropic/v1/messages", bytes.NewReader(input))
	require.NoError(t, err)
	req.Header.Set("Content-Type", "application/json")

	return req
}

// CreateOpenAIChatCompletionsReq builds an HTTP request targeting the OpenAI chat completions endpoint.
func CreateOpenAIChatCompletionsReq(t *testing.T, baseURL string, input []byte) *http.Request {
	t.Helper()

	req, err := http.NewRequestWithContext(t.Context(), "POST", baseURL+"/openai/v1/chat/completions", bytes.NewReader(input))
	require.NoError(t, err)
	req.Header.Set("Content-Type", "application/json")

	return req
}

// CreateOpenAIResponsesReq builds an HTTP request targeting the OpenAI responses endpoint.
func CreateOpenAIResponsesReq(t *testing.T, baseURL string, input []byte) *http.Request {
	t.Helper()

	req, err := http.NewRequestWithContext(t.Context(), "POST", baseURL+"/openai/v1/responses", bytes.NewReader(input))
	require.NoError(t, err)
	req.Header.Set("Content-Type", "application/json")
	return req
}

// OpenAICfg creates a minimal OpenAI config for testing.
func OpenAICfg(url, key string) config.OpenAI {
	return config.OpenAI{
		BaseURL: url,
		Key:     key,
	}
}

// AnthropicCfg creates a minimal Anthropic config for testing.
func AnthropicCfg(url, key string) config.Anthropic {
	return config.Anthropic{
		BaseURL: url,
		Key:     key,
	}
}
