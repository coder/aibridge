package provider

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"cdr.dev/slog/v3"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.opentelemetry.io/otel/trace/noop"
	"golang.org/x/sync/errgroup"

	"github.com/coder/aibridge/config"
	"github.com/coder/aibridge/internal/testutil"
)

type message struct {
	Role    string
	Content string
}

type providerStrategy interface {
	DefaultModel() string
	formatMessages(messages []message) []any
	buildRequestBody(model string, messages []any, stream bool) map[string]any
}
type responsesProvider struct{}

func (*responsesProvider) DefaultModel() string {
	return "gpt-5"
}

func (*responsesProvider) formatMessages(messages []message) []any {
	formatted := make([]any, 0, len(messages))
	for _, msg := range messages {
		formatted = append(formatted, map[string]any{
			"type":    "message",
			"role":    msg.Role,
			"content": msg.Content,
		})
	}
	return formatted
}

func (*responsesProvider) buildRequestBody(model string, messages []any, stream bool) map[string]any {
	return map[string]any{
		"model":  model,
		"input":  messages,
		"stream": stream,
	}
}

type chatCompletionsProvider struct{}

func (*chatCompletionsProvider) DefaultModel() string {
	return "gpt-4"
}

func (*chatCompletionsProvider) formatMessages(messages []message) []any {
	formatted := make([]any, 0, len(messages))
	for _, msg := range messages {
		formatted = append(formatted, map[string]string{
			"role":    msg.Role,
			"content": msg.Content,
		})
	}
	return formatted
}

func (*chatCompletionsProvider) buildRequestBody(model string, messages []any, stream bool) map[string]any {
	return map[string]any{
		"model":    model,
		"messages": messages,
		"stream":   stream,
	}
}

func generateConversation(provider providerStrategy, targetSize int, numMessages int) []any {
	if targetSize <= 0 {
		return nil
	}
	if numMessages < 1 {
		numMessages = 1
	}

	roles := []string{"user", "assistant"}
	messages := make([]message, numMessages)
	for i := range messages {
		messages[i].Role = roles[i%2]
	}
	// Ensure last message is from user (required for LLM APIs).
	if messages[len(messages)-1].Role != "user" {
		messages[len(messages)-1].Role = "user"
	}

	overhead := measureJSONSize(provider.formatMessages(messages))

	bytesPerMessage := targetSize - overhead
	if bytesPerMessage < 0 {
		bytesPerMessage = 0
	}

	perMessage := bytesPerMessage / len(messages)
	remainder := bytesPerMessage % len(messages)

	for i := range messages {
		size := perMessage
		if i == len(messages)-1 {
			size += remainder
		}
		messages[i].Content = strings.Repeat("x", size)
	}

	return provider.formatMessages(messages)
}

func measureJSONSize(v any) int {
	data, err := json.Marshal(v)
	if err != nil {
		return 0
	}
	return len(data)
}

// generateChatCompletionsPayload creates a JSON payload with the specified number of messages.
// Messages alternate between user and assistant roles to simulate a conversation.
func generateChatCompletionsPayload(payloadSize int, messageCount int, stream bool) []byte {
	provider := &chatCompletionsProvider{}
	messages := generateConversation(provider, payloadSize, messageCount)

	body := provider.buildRequestBody(provider.DefaultModel(), messages, stream)
	bodyBytes, err := json.Marshal(body)
	if err != nil {
		panic(err)
	}
	return bodyBytes
}

// generateResponsesPayload creates a JSON payload for the responses API with the specified number of input items.
// Input items alternate between user and assistant roles to simulate a conversation.
func generateResponsesPayload(payloadSize int, inputCount int, stream bool) []byte {
	provider := &responsesProvider{}
	inputs := generateConversation(provider, payloadSize, inputCount)

	body := provider.buildRequestBody(provider.DefaultModel(), inputs, stream)
	bodyBytes, err := json.Marshal(body)
	if err != nil {
		panic(err)
	}
	return bodyBytes
}

// TestOpenAI_ForwardsHeadersToUpstream verifies that custom client headers are forwarded
// to the upstream, while filtered headers (auth, hop-by-hop) are stripped and the
// configured API key is always used.
func TestOpenAI_ForwardsHeadersToUpstream(t *testing.T) {
	t.Parallel()

	const configuredKey = "configured-key"

	testCases := []struct {
		name         string
		route        string
		body         string
		mockResponse string
	}{
		{
			name:         "chat-completions",
			route:        routeChatCompletions,
			body:         `{"model":"gpt-4","messages":[{"role":"user","content":"hello"}],"stream":false}`,
			mockResponse: `{"id":"chatcmpl-123","object":"chat.completion","created":1677652288,"model":"gpt-4","choices":[{"index":0,"message":{"role":"assistant","content":"Hello!"},"finish_reason":"stop"}],"usage":{"prompt_tokens":9,"completion_tokens":12,"total_tokens":21}}`,
		},
		{
			name:         "responses",
			route:        routeResponses,
			body:         `{"model":"gpt-5-mini","input":"hello","stream":false}`,
			mockResponse: `{"id":"resp-123","object":"realtime.response","created":1677652288,"model":"gpt-5-mini","output":[],"usage":{"input_tokens":5,"output_tokens":10,"total_tokens":15}}`,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			var receivedHeaders http.Header

			mockUpstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				receivedHeaders = r.Header.Clone()
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusOK)
				_, _ = w.Write([]byte(tc.mockResponse))
			}))
			t.Cleanup(mockUpstream.Close)

			p := NewOpenAI(config.OpenAI{
				BaseURL: mockUpstream.URL,
				Key:     configuredKey,
			})

			req := httptest.NewRequest(http.MethodPost, tc.route, bytes.NewBufferString(tc.body))
			req.Header.Set("X-Custom-Header", "custom-value")         // should be forwarded
			req.Header.Set("Authorization", "Bearer client-fake-key") // should be stripped (re-injected by SDK)
			req.Header.Set("X-Api-Key", "client-fake-key")            // should be stripped (re-injected by SDK)
			req.Header.Set("Upgrade", "websocket")                    // should be stripped (hop-by-hop)
			w := httptest.NewRecorder()

			interceptor, err := p.CreateInterceptor(w, req, testTracer)
			require.NoError(t, err)
			require.NotNil(t, interceptor)

			interceptor.Setup(slog.Make(), &testutil.MockRecorder{}, nil)

			err = interceptor.ProcessRequest(w, httptest.NewRequest(http.MethodPost, tc.route, nil))
			require.NoError(t, err)

			// Custom headers must reach the upstream.
			assert.Equal(t, "custom-value", receivedHeaders.Get("X-Custom-Header"))
			// Hop-by-hop headers must not reach the upstream.
			assert.Empty(t, receivedHeaders.Get("Upgrade"))
			// Configured key must be used, not the client-provided fake.
			assert.Equal(t, "Bearer "+configuredKey, receivedHeaders.Get("Authorization"))
			// User-Agent must be set by the SDK, not forwarded from the client.
			assert.True(t, strings.HasPrefix(receivedHeaders.Get("User-Agent"), "OpenAI/Go"),
				"upstream User-Agent should be set by the SDK")
		})
	}
}

func BenchmarkOpenAI_CreateInterceptor_ChatCompletions(b *testing.B) {
	provider := NewOpenAI(config.OpenAI{
		BaseURL: "https://api.openai.com/v1/",
		Key:     "test-key",
	})

	tracer := noop.NewTracerProvider().Tracer("test")
	messagesPerRequest := 50
	requestCount := 100
	maxConcurrentRequests := 10
	payloadSizes := []int{2000, 10000, 50000, 100000, 2000000}
	for _, payloadSize := range payloadSizes {
		for _, stream := range []bool{true, false} {
			payload := generateChatCompletionsPayload(payloadSize, messagesPerRequest, stream)
			name := fmt.Sprintf("stream=%t/payloadSize=%d/requests=%d", stream, payloadSize, requestCount)

			b.Run(name, func(b *testing.B) {
				b.ResetTimer()
				for range b.N {
					eg := errgroup.Group{}
					eg.SetLimit(maxConcurrentRequests)
					for i := 0; i < requestCount; i++ {
						eg.Go(func() error {
							req := httptest.NewRequest(http.MethodPost, routeChatCompletions, bytes.NewReader(payload))
							w := httptest.NewRecorder()
							_, err := provider.CreateInterceptor(w, req, tracer)
							if err != nil {
								return err
							}
							return nil
						})
					}
				}
			})
		}
	}
}

func BenchmarkOpenAI_CreateInterceptor_Responses(b *testing.B) {
	provider := NewOpenAI(config.OpenAI{
		BaseURL: "https://api.openai.com/v1/",
		Key:     "test-key",
	})

	tracer := noop.NewTracerProvider().Tracer("test")
	messagesPerRequest := 50
	requestCount := 100
	maxConcurrentRequests := 10
	// payloadSizes := []int{2000, 10000, 50000, 100000, 2000000}
	payloadSizes := []int{2000000}
	for _, payloadSize := range payloadSizes {
		for _, stream := range []bool{true, false} {
			payload := generateResponsesPayload(payloadSize, messagesPerRequest, stream)
			name := fmt.Sprintf("stream=%t/payloadSize=%d/requests=%d", stream, payloadSize, requestCount)

			b.Run(name, func(b *testing.B) {
				b.ResetTimer()
				for range b.N {
					eg := errgroup.Group{}
					eg.SetLimit(maxConcurrentRequests)
					for i := 0; i < requestCount; i++ {
						eg.Go(func() error {
							req := httptest.NewRequest(http.MethodPost, routeResponses, bytes.NewReader(payload))
							w := httptest.NewRecorder()
							interceptor, err := provider.CreateInterceptor(w, req, tracer)
							if err != nil {
								return err
							}
							err = interceptor.ProcessRequest(w, req)
							if err != nil {
								return err
							}
							return nil
						})
					}
				}
			})
		}
	}
}
