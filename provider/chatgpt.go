package provider

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"

	"github.com/coder/aibridge/config"
	"github.com/coder/aibridge/intercept"
	"github.com/coder/aibridge/intercept/chatcompletions"
	"github.com/coder/aibridge/intercept/responses"
	"github.com/coder/aibridge/tracing"
	"github.com/coder/aibridge/utils"
	"github.com/google/uuid"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"
)

const (
	chatGPTBaseURL = "https://chatgpt.com/backend-api/codex"

	routeChatGPTChatCompletions = "/chat/completions"
	routeChatGPTResponses       = "/responses"
)

var chatGPTOpenErrorResponse = func() []byte {
	return []byte(`{"error":{"message":"circuit breaker is open","type":"server_error","code":"service_unavailable"}}`)
}

// ChatGPT implements the Provider interface for the ChatGPT backend.
// ChatGPT uses per-user credentials passed through the request headers
// rather than a statically configured API key.
// The implementation mirrors the OpenAI provider. The ChatGPT backend API is not
// publicly documented, but manual testing suggests it follows the same route
// structure as the OpenAI API.
type ChatGPT struct {
	cfg            config.ChatGPT
	circuitBreaker *config.CircuitBreaker
}

var _ Provider = &ChatGPT{}

func NewChatGPT(cfg config.ChatGPT) *ChatGPT {
	if cfg.BaseURL == "" {
		cfg.BaseURL = chatGPTBaseURL
	}
	if cfg.APIDumpDir == "" {
		cfg.APIDumpDir = os.Getenv("BRIDGE_DUMP_DIR")
	}
	if cfg.CircuitBreaker != nil {
		cfg.CircuitBreaker.OpenErrorResponse = chatGPTOpenErrorResponse
	}

	return &ChatGPT{
		cfg:            cfg,
		circuitBreaker: cfg.CircuitBreaker,
	}
}

func (p *ChatGPT) Name() string {
	return config.ProviderChatGPT
}

func (p *ChatGPT) RoutePrefix() string {
	return fmt.Sprintf("/%s/v1", p.Name())
}

func (p *ChatGPT) BridgedRoutes() []string {
	return []string{
		routeChatGPTChatCompletions,
		routeChatGPTResponses,
	}
}

func (p *ChatGPT) PassthroughRoutes() []string {
	return []string{
		"/conversations",
		"/conversations/",
		"/models",
		"/models/",
		"/responses/",
	}
}

func (p *ChatGPT) CreateInterceptor(w http.ResponseWriter, r *http.Request, tracer trace.Tracer) (_ intercept.Interceptor, outErr error) {
	id := uuid.New()

	_, span := tracer.Start(r.Context(), "Intercept.CreateInterceptor")
	defer tracing.EndSpanErr(span, &outErr)

	var interceptor intercept.Interceptor

	// Extract the per-user ChatGPT token from the Authorization header.
	token := utils.ExtractBearerToken(r.Header.Get("Authorization"))
	if token == "" {
		span.SetStatus(codes.Error, "missing authorization")
		return nil, fmt.Errorf("missing ChatGPT authorization: Authorization header not found or invalid")
	}

	// Build config for the interceptor using the per-request token.
	// ChatGPT's API is OpenAI-compatible, so it uses the OpenAI interceptors
	// that require a config.OpenAI.
	openAICfg := config.OpenAI{
		BaseURL:          p.cfg.BaseURL,
		Key:              token,
		APIDumpDir:       p.cfg.APIDumpDir,
		CircuitBreaker:   p.cfg.CircuitBreaker,
		SendActorHeaders: p.cfg.SendActorHeaders,
		ExtraHeaders:     p.cfg.ExtraHeaders,
	}

	path := strings.TrimPrefix(r.URL.Path, p.RoutePrefix())
	switch path {
	case routeChatGPTChatCompletions:
		var req chatcompletions.ChatCompletionNewParamsWrapper
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			return nil, fmt.Errorf("unmarshal request body: %w", err)
		}

		if req.Stream {
			interceptor = chatcompletions.NewStreamingInterceptor(id, &req, openAICfg, r.Header, p.AuthHeader(), tracer)
		} else {
			interceptor = chatcompletions.NewBlockingInterceptor(id, &req, openAICfg, r.Header, p.AuthHeader(), tracer)
		}

	case routeChatGPTResponses:
		payload, err := io.ReadAll(r.Body)
		if err != nil {
			return nil, fmt.Errorf("read body: %w", err)
		}
		reqPayload, err := responses.NewResponsesRequestPayload(payload)
		if err != nil {
			return nil, fmt.Errorf("unmarshal request body: %w", err)
		}
		if reqPayload.Stream() {
			interceptor = responses.NewStreamingInterceptor(id, reqPayload, openAICfg, r.Header, p.AuthHeader(), tracer)
		} else {
			interceptor = responses.NewBlockingInterceptor(id, reqPayload, openAICfg, r.Header, p.AuthHeader(), tracer)
		}

	default:
		span.SetStatus(codes.Error, "unknown route: "+r.URL.Path)
		return nil, UnknownRoute
	}
	span.SetAttributes(interceptor.TraceAttributes(r)...)
	return interceptor, nil
}

func (p *ChatGPT) BaseURL() string {
	return p.cfg.BaseURL
}

func (p *ChatGPT) AuthHeader() string {
	return "Authorization"
}

func (p *ChatGPT) InjectAuthHeader(headers *http.Header) {}

func (p *ChatGPT) CircuitBreakerConfig() *config.CircuitBreaker {
	return p.circuitBreaker
}

func (p *ChatGPT) APIDumpDir() string {
	return p.cfg.APIDumpDir
}
