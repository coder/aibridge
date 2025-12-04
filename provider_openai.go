package aibridge

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"

	"github.com/coder/aibridge/tracing"
	"github.com/google/uuid"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"
)

var _ Provider = &OpenAIProvider{}

// OpenAIProvider allows for interactions with the OpenAI API.
type OpenAIProvider struct {
	baseURL, key string
}

const (
	ProviderOpenAI = "openai"

	routeChatCompletions = "/openai/v1/chat/completions" // https://platform.openai.com/docs/api-reference/chat
)

func NewOpenAIProvider(cfg OpenAIConfig) *OpenAIProvider {
	if cfg.BaseURL == "" {
		cfg.BaseURL = "https://api.openai.com/v1/"
	}

	if cfg.Key == "" {
		cfg.Key = os.Getenv("OPENAI_API_KEY")
	}

	return &OpenAIProvider{
		baseURL: cfg.BaseURL,
		key:     cfg.Key,
	}
}

func (p *OpenAIProvider) Name() string {
	return ProviderOpenAI
}

func (p *OpenAIProvider) BridgedRoutes() []string {
	return []string{routeChatCompletions}
}

// PassthroughRoutes define the routes which are not currently intercepted
// but must be passed through to the upstream.
// The /v1/completions legacy API is deprecated and will not be passed through.
// See https://platform.openai.com/docs/api-reference/completions.
func (p *OpenAIProvider) PassthroughRoutes() []string {
	return []string{
		"/v1/models",
		"/v1/models/",   // See https://pkg.go.dev/net/http#hdr-Trailing_slash_redirection-ServeMux.
		"/v1/responses", // TODO: support Responses API.
	}
}

func (p *OpenAIProvider) CreateInterceptor(w http.ResponseWriter, r *http.Request, tracer trace.Tracer) (_ Interceptor, outErr error) {
	id := uuid.New()

	_, span := tracer.Start(r.Context(), "Intercept.CreateInterceptor")
	defer tracing.EndSpanErr(span, &outErr)

	payload, err := io.ReadAll(r.Body)
	if err != nil {
		return nil, fmt.Errorf("read body: %w", err)
	}

	switch r.URL.Path {
	case routeChatCompletions:
		var req ChatCompletionNewParamsWrapper
		if err := json.Unmarshal(payload, &req); err != nil {
			return nil, fmt.Errorf("unmarshal request body: %w", err)
		}

		var interceptor Interceptor
		if req.Stream {
			interceptor = NewOpenAIStreamingChatInterception(id, &req, p.baseURL, p.key, tracer)
		} else {
			interceptor = NewOpenAIBlockingChatInterception(id, &req, p.baseURL, p.key, tracer)
		}
		span.SetAttributes(interceptor.TraceAttributes(r)...)
		return interceptor, nil
	}

	span.SetStatus(codes.Error, "unknown route: "+r.URL.Path)
	return nil, UnknownRoute
}

func (p *OpenAIProvider) BaseURL() string {
	return p.baseURL
}

func (p *OpenAIProvider) AuthHeader() string {
	return "Authorization"
}

func (p *OpenAIProvider) InjectAuthHeader(headers *http.Header) {
	if headers == nil {
		headers = &http.Header{}
	}

	headers.Set(p.AuthHeader(), "Bearer "+p.key)
}
