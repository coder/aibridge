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
	"github.com/google/uuid"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"
)

const (
	copilotBaseURL = "https://api.individual.githubcopilot.com"

	// Copilot exposes an OpenAI-compatible API, including for Anthropic models.
	routeCopilotChatCompletions = "/chat/completions"
	routeCopilotResponses       = "/responses"
)

var copilotOpenErrorResponse = func() []byte {
	return []byte(`{"error":{"message":"circuit breaker is open","type":"server_error","code":"service_unavailable"}}`)
}

// Headers that need to be forwarded to Copilot API.
// These were determined through manual testing as there is no reference
// of the headers in the official documentation.
// LiteLLM uses the same headers:
// https://docs.litellm.ai/docs/providers/github_copilot
var copilotForwardHeaders = []string{
	"Editor-Version",
	"Copilot-Integration-Id",
}

// Copilot implements the Provider interface for GitHub Copilot.
// Unlike other providers, Copilot uses per-user API keys that are passed through
// the request headers rather than configured statically.
type Copilot struct {
	cfg            config.Copilot
	circuitBreaker *config.CircuitBreaker
}

var _ Provider = &Copilot{}

func NewCopilot(cfg config.Copilot) *Copilot {
	if cfg.BaseURL == "" {
		cfg.BaseURL = copilotBaseURL
	}
	if cfg.APIDumpDir == "" {
		cfg.APIDumpDir = os.Getenv("BRIDGE_DUMP_DIR")
	}
	if cfg.CircuitBreaker != nil {
		cfg.CircuitBreaker.OpenErrorResponse = copilotOpenErrorResponse
	}
	return &Copilot{
		cfg:            cfg,
		circuitBreaker: cfg.CircuitBreaker,
	}
}

func (p *Copilot) Name() string {
	return config.ProviderCopilot
}

func (p *Copilot) BaseURL() string {
	return p.cfg.BaseURL
}

func (p *Copilot) RoutePrefix() string {
	return fmt.Sprintf("/%s", p.Name())
}

func (p *Copilot) BridgedRoutes() []string {
	return []string{
		routeCopilotChatCompletions,
		routeCopilotResponses,
	}
}

func (p *Copilot) PassthroughRoutes() []string {
	return []string{
		"/models",
		"/models/",
		"/agents/",
		"/mcp/",
		"/.well-known/",
	}
}

func (p *Copilot) AuthHeader() string {
	return "Authorization"
}

// InjectAuthHeader is a no-op for Copilot.
// Copilot uses per-user tokens passed in the original Authorization header,
// rather than a global key configured at the provider level.
// The original Authorization header flows through untouched from the client.
func (p *Copilot) InjectAuthHeader(_ *http.Header) {}

func (p *Copilot) CircuitBreakerConfig() *config.CircuitBreaker {
	return p.circuitBreaker
}

func (p *Copilot) CreateInterceptor(_ http.ResponseWriter, r *http.Request, tracer trace.Tracer) (_ intercept.Interceptor, outErr error) {
	_, span := tracer.Start(r.Context(), "Intercept.CreateInterceptor")
	defer tracing.EndSpanErr(span, &outErr)

	// Extract the per-user Copilot key from the Authorization header.
	key := extractBearerToken(r.Header.Get("Authorization"))
	if key == "" {
		span.SetStatus(codes.Error, "missing authorization")
		return nil, fmt.Errorf("missing Copilot authorization: Authorization header not found or invalid")
	}

	id := uuid.New()

	// Build config for the interceptor using the per-request key.
	// Copilot's API is OpenAI-compatible, so it uses the OpenAI interceptors
	// that require a config.OpenAI.
	cfg := config.OpenAI{
		BaseURL:        p.cfg.BaseURL,
		Key:            key,
		APIDumpDir:     p.cfg.APIDumpDir,
		CircuitBreaker: p.cfg.CircuitBreaker,
		ExtraHeaders:   extractCopilotHeaders(r),
	}

	var interceptor intercept.Interceptor

	switch r.URL.Path {
	case routeCopilotChatCompletions:
		var req chatcompletions.ChatCompletionNewParamsWrapper
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			return nil, fmt.Errorf("unmarshal chat completions request body: %w", err)
		}

		if req.Stream {
			interceptor = chatcompletions.NewStreamingInterceptor(id, &req, cfg, tracer)
		} else {
			interceptor = chatcompletions.NewBlockingInterceptor(id, &req, cfg, tracer)
		}

	case routeCopilotResponses:
		payload, err := io.ReadAll(r.Body)
		if err != nil {
			return nil, fmt.Errorf("read body: %w", err)
		}
		var req responses.ResponsesNewParamsWrapper
		if err := json.Unmarshal(payload, &req); err != nil {
			return nil, fmt.Errorf("unmarshal responses request body: %w", err)
		}

		if req.Stream {
			interceptor = responses.NewStreamingInterceptor(id, &req, payload, cfg, req.Model, tracer)
		} else {
			interceptor = responses.NewBlockingInterceptor(id, &req, payload, cfg, req.Model, tracer)
		}

	default:
		span.SetStatus(codes.Error, "unknown route: "+r.URL.Path)
		return nil, UnknownRoute
	}

	span.SetAttributes(interceptor.TraceAttributes(r)...)
	return interceptor, nil
}

// extractBearerToken extracts the token from a "Bearer <token>" authorization header.
func extractBearerToken(auth string) string {
	if auth := strings.TrimSpace(auth); auth != "" {
		fields := strings.Fields(auth)
		if len(fields) == 2 && strings.EqualFold(fields[0], "Bearer") {
			return fields[1]
		}
	}
	return ""
}

// extractCopilotHeaders extracts headers required by the Copilot API from the
// incoming request. Copilot requires certain client headers to be forwarded.
func extractCopilotHeaders(r *http.Request) map[string]string {
	headers := make(map[string]string, len(copilotForwardHeaders))
	for _, h := range copilotForwardHeaders {
		if v := r.Header.Get(h); v != "" {
			headers[h] = v
		}
	}
	return headers
}
