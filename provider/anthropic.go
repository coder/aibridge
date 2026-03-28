package provider

import (
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
	"github.com/coder/aibridge/utils"
	"github.com/google/uuid"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"
)

// anthropicForwardHeaders lists headers from incoming requests that should be
// forwarded to the Anthropic API.
// TODO(ssncferreira): remove as part of https://github.com/coder/aibridge/issues/192
var anthropicForwardHeaders = []string{
	"Anthropic-Beta",
}

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

		reqPayload, err := messages.NewMessagesRequestPayload(payload)
		if err != nil {
			return nil, fmt.Errorf("unmarshal request body: %w", err)
		}

		cfg := p.cfg
		cfg.ExtraHeaders = extractAnthropicHeaders(r)

		// At this point the request contains only LLM provider headers.
		// Any Coder-specific authentication has already been stripped.
		//
		// In centralized mode neither Authorization nor X-Api-Key is
		// present, so cfg keeps the centralized key unchanged.
		//
		// In BYOK mode the user's LLM credentials survive intact.
		// If X-Api-Key is present the user has a personal API key;
		// overwrite the centralized key with it. If Authorization is
		// present the user authenticated directly with provider;
		// set BYOKBearerToken and clear the centralized key.
		// When both are present, X-Api-Key takes priority to match
		// claude-code behavior.
		authHeaderName := p.AuthHeader()
		if apiKey := r.Header.Get("X-Api-Key"); apiKey != "" {
			cfg.Key = apiKey
			authHeaderName = "X-Api-Key"
		} else if token := utils.ExtractBearerToken(r.Header.Get("Authorization")); token != "" {
			cfg.BYOKBearerToken = token
			cfg.Key = ""
			authHeaderName = "Authorization"
		}

		// TODO(ssncferreira): when Bedrock is added as a separate provider, pass the
		//   resolved provider name instead of p.Name() here.
		upstream := p.ResolveUpstream(r)

		var interceptor intercept.Interceptor
		if reqPayload.Stream() {
			interceptor = messages.NewStreamingInterceptor(id, reqPayload, upstream, cfg, p.bedrockCfg, r.Header, authHeaderName, tracer)
		} else {
			interceptor = messages.NewBlockingInterceptor(id, reqPayload, upstream, cfg, p.bedrockCfg, r.Header, authHeaderName, tracer)
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

func (p *Anthropic) ResolveUpstream(_ *http.Request) intercept.ResolvedUpstream {
	return intercept.ResolvedUpstream{Name: p.Name(), URL: p.cfg.BaseURL}
}

func (p *Anthropic) AuthHeader() string {
	return "X-Api-Key"
}

func (p *Anthropic) InjectAuthHeader(headers *http.Header) {
	if headers == nil {
		headers = &http.Header{}
	}

	// BYOK: if the request already carries user-supplied credentials,
	// do not overwrite them with the centralized key.
	if headers.Get("X-Api-Key") != "" || headers.Get("Authorization") != "" {
		return
	}

	headers.Set(p.AuthHeader(), p.cfg.Key)
}

func (p *Anthropic) CircuitBreakerConfig() *config.CircuitBreaker {
	return p.cfg.CircuitBreaker
}

func (p *Anthropic) APIDumpDir() string {
	return p.cfg.APIDumpDir
}

// extractAnthropicHeaders extracts headers required by the Anthropic API from
// the incoming request.
// TODO(ssncferreira): remove as part of https://github.com/coder/aibridge/issues/192
func extractAnthropicHeaders(r *http.Request) map[string]string {
	headers := make(map[string]string, len(anthropicForwardHeaders))
	for _, h := range anthropicForwardHeaders {
		if v := r.Header.Get(h); v != "" {
			headers[h] = v
		}
	}
	return headers
}
