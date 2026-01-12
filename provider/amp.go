package provider

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/coder/aibridge/config"
	"github.com/coder/aibridge/intercept"
	"github.com/coder/aibridge/intercept/messages"
	"github.com/coder/aibridge/tracing"
	"github.com/google/uuid"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"
)

var _ Provider = &Amp{}

const routeAmpMessages = "/amp/v1/messages"

type Amp struct {
	cfg config.Amp
}

func NewAmp(cfg config.Amp) *Amp {
	if cfg.BaseURL == "" {
		cfg.BaseURL = "https://ampcode.com/api/provider/anthropic"
	}
	return &Amp{cfg: cfg}
}

func (p *Amp) Name() string {
	return config.ProviderAmp
}

func (p *Amp) BridgedRoutes() []string {
	return []string{routeAmpMessages}
}

func (p *Amp) PassthroughRoutes() []string {
	return []string{
		"/v1/models",
		"/v1/models/", // See https://pkg.go.dev/net/http#hdr-Trailing_slash_redirection-ServeMux.
		"/v1/messages/count_tokens",
	}
}

func (p *Amp) CreateInterceptor(w http.ResponseWriter, r *http.Request, tracer trace.Tracer) (_ intercept.Interceptor, outErr error) {
	id := uuid.New()
	_, span := tracer.Start(r.Context(), "Intercept.CreateInterceptor")
	defer tracing.EndSpanErr(span, &outErr)

	// Capture the API key from the incoming request.
	// Amp sends requests with X-Api-Key containing the authenticated user's API key.
	// One key per user instead of a global key.
	apiKey := r.Header.Get("X-Api-Key")

	payload, err := io.ReadAll(r.Body)
	if err != nil {
		return nil, fmt.Errorf("read body: %w", err)
	}

	switch r.URL.Path {
	case routeAmpMessages:
		var req messages.MessageNewParamsWrapper
		if err := json.Unmarshal(payload, &req); err != nil {
			return nil, fmt.Errorf("failed to unmarshal request: %w", err)
		}

		// Reuse Anthropic interceptors since Amp uses the same API format.
		// Override the config with the user's API key from the request.
		anthropicCfg := config.Anthropic{
			BaseURL: p.cfg.BaseURL,
			Key:     apiKey,
		}

		var interceptor intercept.Interceptor
		if req.Stream {
			interceptor = messages.NewStreamingInterceptor(id, &req, anthropicCfg, nil, tracer)
		} else {
			interceptor = messages.NewBlockingInterceptor(id, &req, anthropicCfg, nil, tracer)
		}
		span.SetAttributes(interceptor.TraceAttributes(r)...)
		return interceptor, nil
	}

	span.SetStatus(codes.Error, "unknown route: "+r.URL.Path)
	return nil, UnknownRoute
}

func (p *Amp) BaseURL() string {
	return p.cfg.BaseURL
}

func (p *Amp) AuthHeader() string {
	return "X-Api-Key"
}

// InjectAuthHeader is a no-op for Amp.
// Amp requests already include X-Api-Key with the authenticated user's API key.
// Unlike Anthropic/OpenAI where we inject a global server-side key, Amp uses per-user keys
// that come directly from the client request.
func (p *Amp) InjectAuthHeader(headers *http.Header) {
	// No-op: Amp uses per-user API keys from the incoming request, not a global key.
}
