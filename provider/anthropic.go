package provider

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"

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

const routeMessages = "/anthropic/v1/messages" // https://docs.anthropic.com/en/api/messages

func NewAnthropic(cfg config.Anthropic, bedrockCfg *config.AWSBedrock) *Anthropic {
	if cfg.BaseURL == "" {
		cfg.BaseURL = "https://api.anthropic.com/"
	}
	if cfg.Key == "" {
		cfg.Key = os.Getenv("ANTHROPIC_API_KEY")
	}

	return &Anthropic{
		cfg:        cfg,
		bedrockCfg: bedrockCfg,
	}
}

func (p *Anthropic) Name() string {
	return config.ProviderAnthropic
}

func (p *Anthropic) BridgedRoutes() []string {
	return []string{routeMessages}
}

func (p *Anthropic) PassthroughRoutes() []string {
	return []string{
		"/v1/models",
		"/v1/models/", // See https://pkg.go.dev/net/http#hdr-Trailing_slash_redirection-ServeMux.
		"/v1/messages/count_tokens",
	}
}

func (p *Anthropic) CreateInterceptor(w http.ResponseWriter, r *http.Request, tracer trace.Tracer) (_ intercept.Interceptor, outErr error) {
	id := uuid.New()
	_, span := tracer.Start(r.Context(), "Intercept.CreateInterceptor")
	defer tracing.EndSpanErr(span, &outErr)

	payload, err := io.ReadAll(r.Body)
	if err != nil {
		return nil, fmt.Errorf("read body: %w", err)
	}

	switch r.URL.Path {
	case routeMessages:
		var req messages.MessageNewParamsWrapper
		if err := json.Unmarshal(payload, &req); err != nil {
			return nil, fmt.Errorf("failed to unmarshal request: %w", err)
		}

		var interceptor intercept.Interceptor
		if req.Stream {
			interceptor = messages.NewStreamingInterceptor(id, &req, p.cfg, p.bedrockCfg, tracer)
		} else {
			interceptor = messages.NewBlockingInterceptor(id, &req, p.cfg, p.bedrockCfg, tracer)
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

func (p *Anthropic) CircuitBreakerConfig() *circuitbreaker.Config {
	if p.cfg.CircuitBreaker != nil && p.cfg.CircuitBreaker.IsFailure == nil {
		p.cfg.CircuitBreaker.IsFailure = circuitbreaker.AnthropicIsFailure
	}
	return p.cfg.CircuitBreaker
}
