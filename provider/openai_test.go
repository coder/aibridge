package provider

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/coder/aibridge/config"
	"go.opentelemetry.io/otel/trace/noop"
	"golang.org/x/sync/errgroup"
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
