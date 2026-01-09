package provider

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"

	"github.com/coder/aibridge/config"
	"github.com/coder/aibridge/intercept"
	"github.com/coder/aibridge/intercept/chatcompletions"
	"github.com/coder/aibridge/intercept/responses"
	"github.com/coder/aibridge/tracing"
	"github.com/google/uuid"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"
)

const (
	routeChatCompletions = "/openai/v1/chat/completions" // https://platform.openai.com/docs/api-reference/chat
	routeResponses       = "/openai/v1/responses"        // https://platform.openai.com/docs/api-reference/responses
)

// OpenAI allows for interactions with the OpenAI API.
type OpenAI struct {
	baseURL string
	key     string
}

var _ Provider = &OpenAI{}

func NewOpenAI(cfg config.OpenAI) *OpenAI {
	if cfg.BaseURL == "" {
		cfg.BaseURL = "https://api.openai.com/v1/"
	}

	if cfg.Key == "" {
		cfg.Key = os.Getenv("OPENAI_API_KEY")
	}

	return &OpenAI{
		baseURL: cfg.BaseURL,
		key:     cfg.Key,
	}
}

func (p *OpenAI) Name() string {
	return config.ProviderOpenAI
}

func (p *OpenAI) BridgedRoutes() []string {
	return []string{
		routeChatCompletions,
		routeResponses,
	}
}

// PassthroughRoutes define the routes which are not currently intercepted
// but must be passed through to the upstream.
// The /v1/completions legacy API is deprecated and will not be passed through.
// See https://platform.openai.com/docs/api-reference/completions.
func (p *OpenAI) PassthroughRoutes() []string {
	return []string{
		"/v1/models",
		"/v1/models/",    // See https://pkg.go.dev/net/http#hdr-Trailing_slash_redirection-ServeMux.
		"/v1/responses/", // Forwards other responses API endpoints, eg: https://platform.openai.com/docs/api-reference/responses/get
		"/v1/conversations",
		"/v1/conversations/",
	}
}

func (p *OpenAI) CreateInterceptor(w http.ResponseWriter, r *http.Request, tracer trace.Tracer) (_ intercept.Interceptor, outErr error) {
	id := uuid.New()

	_, span := tracer.Start(r.Context(), "Intercept.CreateInterceptor")
	defer tracing.EndSpanErr(span, &outErr)

	payload, err := io.ReadAll(r.Body)
	if err != nil {
		return nil, fmt.Errorf("read body: %w", err)
	}

	var interceptor intercept.Interceptor

	switch r.URL.Path {
	case routeChatCompletions:
		var req chatcompletions.ChatCompletionNewParamsWrapper
		if err := json.Unmarshal(payload, &req); err != nil {
			return nil, fmt.Errorf("unmarshal request body: %w", err)
		}

		if req.Stream {
			interceptor = chatcompletions.NewStreamingInterceptor(id, &req, p.baseURL, p.key, tracer)
		} else {
			interceptor = chatcompletions.NewBlockingInterceptor(id, &req, p.baseURL, p.key, tracer)
		}

	case routeResponses:
		var req responses.ResponsesNewParamsWrapper
		if err := json.Unmarshal(payload, &req); err != nil {
			return nil, fmt.Errorf("unmarshal request body: %w", err)
		}
		if req.Stream {
			interceptor = responses.NewStreamingInterceptor(id, &req, payload, p.baseURL, p.key, string(req.Model))
		} else {
			interceptor = responses.NewBlockingInterceptor(id, &req, payload, p.baseURL, p.key, string(req.Model))
		}

	default:
		span.SetStatus(codes.Error, "unknown route: "+r.URL.Path)
		return nil, UnknownRoute
	}
	span.SetAttributes(interceptor.TraceAttributes(r)...)
	return interceptor, nil
}

func (p *OpenAI) BaseURL() string {
	return p.baseURL
}

func (p *OpenAI) AuthHeader() string {
	return "Authorization"
}

func (p *OpenAI) InjectAuthHeader(headers *http.Header) {
	if headers == nil {
		headers = &http.Header{}
	}

	headers.Set(p.AuthHeader(), "Bearer "+p.key)
}
