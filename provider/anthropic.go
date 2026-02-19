package provider

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"

	"github.com/coder/aibridge/circuitbreaker"
	"github.com/coder/aibridge/config"
	"github.com/coder/aibridge/intercept"
	"github.com/coder/aibridge/intercept/messages"
	"github.com/coder/aibridge/tracing"
	"github.com/google/uuid"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"
)

var _ Provider = &Anthropic{}

// Anthropic allows for interactions with the Anthropic API.
type Anthropic struct {
	cfg        config.Anthropic
	bedrockCfg *config.AWSBedrock
}

const routeMessages = "/v1/messages" // https://docs.anthropic.com/en/api/messages

var anthropicOpenErrorResponse = func() []byte {
	return []byte(`{"type":"error","error":{"type":"overloaded_error","message":"circuit breaker is open"}}`)
}

var anthropicIsFailure = func(statusCode int) bool {
	// https://platform.claude.com/docs/en/api/errors
	if statusCode == 529 {
		return true
	}
	return circuitbreaker.DefaultIsFailure(statusCode)
}

func NewAnthropic(cfg config.Anthropic, bedrockCfg *config.AWSBedrock) *Anthropic {
	if cfg.BaseURL == "" {
		cfg.BaseURL = "https://api.anthropic.com/"
	}
	if cfg.Key == "" {
		cfg.Key = os.Getenv("ANTHROPIC_API_KEY")
	}
	if cfg.APIDumpDir == "" {
		cfg.APIDumpDir = os.Getenv("BRIDGE_DUMP_DIR")
	}
	if cfg.CircuitBreaker != nil {
		cfg.CircuitBreaker.IsFailure = anthropicIsFailure
		cfg.CircuitBreaker.OpenErrorResponse = anthropicOpenErrorResponse
	}

	return &Anthropic{
		cfg:        cfg,
		bedrockCfg: bedrockCfg,
	}
}

func (p *Anthropic) Name() string {
	return config.ProviderAnthropic
}

func (p *Anthropic) RoutePrefix() string {
	return fmt.Sprintf("/%s", p.Name())
}

func (p *Anthropic) BridgedRoutes() []string {
	return []string{routeMessages}
}

func (p *Anthropic) PassthroughRoutes() []string {
	return []string{
		"/v1/models",
		"/v1/models/", // See https://pkg.go.dev/net/http#hdr-Trailing_slash_redirection-ServeMux.
		"/v1/messages/count_tokens",
		"/api/event_logging/",
	}
}

func (p *Anthropic) CreateInterceptor(w http.ResponseWriter, r *http.Request, tracer trace.Tracer) (_ intercept.Interceptor, outErr error) {
	id := uuid.New()
	_, span := tracer.Start(r.Context(), "Intercept.CreateInterceptor")
	defer tracing.EndSpanErr(span, &outErr)

	path := strings.TrimPrefix(r.URL.Path, p.RoutePrefix())
	switch path {
	case routeMessages:
		payload, err := io.ReadAll(r.Body)
		if err != nil {
			return nil, fmt.Errorf("read body: %w", err)
		}
		var req messages.MessageNewParamsWrapper
		if err := json.Unmarshal(payload, &req); err != nil {
			return nil, fmt.Errorf("unmarshal request body: %w", err)
		}

		var interceptor intercept.Interceptor
		if req.Stream {
			interceptor = messages.NewStreamingInterceptor(id, &req, payload, p.cfg, p.bedrockCfg, tracer)
		} else {
			interceptor = messages.NewBlockingInterceptor(id, &req, payload, p.cfg, p.bedrockCfg, tracer)
		}
		span.SetAttributes(interceptor.TraceAttributes(r)...)
		return interceptor, nil
	}

	span.SetStatus(codes.Error, "unknown route: "+r.URL.Path)
	return nil, UnknownRoute
}

func (p *Anthropic) BaseURL() string {
	return p.cfg.BaseURL
}

func (p *Anthropic) AuthHeader() string {
	return "X-Api-Key"
}

func (p *Anthropic) InjectAuthHeader(headers *http.Header) {
	if headers == nil {
		headers = &http.Header{}
	}

	headers.Set(p.AuthHeader(), p.cfg.Key)
}

func (p *Anthropic) CircuitBreakerConfig() *config.CircuitBreaker {
	return p.cfg.CircuitBreaker
}
