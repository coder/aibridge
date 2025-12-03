package aibridge

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"

	"github.com/google/uuid"
)

var _ Provider = &OpenAIProvider{}

// OpenAIProvider allows for interactions with the OpenAI API.
type OpenAIProvider struct {
	cfg *OpenAIConfig
}

const (
	ProviderOpenAI = "openai"

	routeChatCompletions = "/openai/v1/chat/completions" // https://platform.openai.com/docs/api-reference/chat
)

func NewOpenAIProvider(cfg *OpenAIConfig) *OpenAIProvider {
	if cfg == nil {
		panic("ProviderConfig cannot be nil")
	}

	if cfg.BaseURL() == "" {
		cfg.SetBaseURL("https://api.openai.com/v1/")
	}

	if cfg.Key() == "" {
		cfg.SetKey(os.Getenv("OPENAI_API_KEY"))
	}

	return &OpenAIProvider{
		cfg: cfg,
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

func (p *OpenAIProvider) CreateInterceptor(w http.ResponseWriter, r *http.Request) (Interceptor, error) {
	payload, err := io.ReadAll(r.Body)
	if err != nil {
		return nil, fmt.Errorf("read body: %w", err)
	}

	id := uuid.New()

	switch r.URL.Path {
	case routeChatCompletions:
		var req ChatCompletionNewParamsWrapper
		if err := json.Unmarshal(payload, &req); err != nil {
			return nil, fmt.Errorf("unmarshal request body: %w", err)
		}

		if req.Stream {
			return NewOpenAIStreamingChatInterception(id, &req, p.cfg), nil
		} else {
			return NewOpenAIBlockingChatInterception(id, &req, p.cfg), nil
		}
	}

	return nil, UnknownRoute
}

func (p *OpenAIProvider) BaseURL() string {
	return p.cfg.BaseURL()
}

func (p *OpenAIProvider) AuthHeader() string {
	return "Authorization"
}

func (p *OpenAIProvider) InjectAuthHeader(headers *http.Header) {
	if headers == nil {
		headers = &http.Header{}
	}

	headers.Set(p.AuthHeader(), "Bearer "+p.cfg.Key())
}
