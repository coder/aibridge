package aibridge

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/http/httputil"
	"os"

	"github.com/google/uuid"
	"github.com/openai/openai-go/v2"
	"github.com/openai/openai-go/v2/option"
)

var _ Provider = &OpenAIProvider{}

// OpenAIProvider allows for interactions with the OpenAI API.
type OpenAIProvider struct {
	cfg ProviderConfig
}

const (
	ProviderOpenAI = "openai"

	routeChatCompletions = "/openai/v1/chat/completions" // https://platform.openai.com/docs/api-reference/chat
)

func NewOpenAIProvider(cfg ProviderConfig) *OpenAIProvider {
	if cfg.BaseURL == "" {
		cfg.BaseURL = "https://api.openai.com/v1/"
	}

	if cfg.Key == "" {
		cfg.Key = os.Getenv("OPENAI_API_KEY")
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
	return p.cfg.BaseURL
}

func (p *OpenAIProvider) AuthHeader() string {
	return "Authorization"
}

func (p *OpenAIProvider) InjectAuthHeader(headers *http.Header) {
	if headers == nil {
		headers = &http.Header{}
	}

	headers.Set(p.AuthHeader(), "Bearer "+p.cfg.Key)
}

func newOpenAIClient(cfg ProviderConfig, id string) openai.Client {
	var opts []option.RequestOption
	opts = append(opts, option.WithAPIKey(cfg.Key))
	opts = append(opts, option.WithBaseURL(cfg.BaseURL))

	if cfg.EnableUpstreamLogging {
		reqLogFile, err := os.OpenFile("/tmp/openai-req.log", os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
		if err == nil {
			reqLogger := log.New(reqLogFile, "", log.LstdFlags)

			resLogFile, err := os.OpenFile("/tmp/openai-res.log", os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
			if err == nil {
				resLogger := log.New(resLogFile, "", log.LstdFlags)

				opts = append(opts, option.WithMiddleware(func(req *http.Request, next option.MiddlewareNext) (*http.Response, error) {
					if reqDump, err := httputil.DumpRequest(req, true); err == nil {
						reqLogger.Printf("[req] [%s] %s", id, reqDump)
					}

					resp, err := next(req)
					if err != nil {
						resLogger.Printf("[res] [%s] Error: %v", id, err)
						return resp, err
					}

					if respDump, err := httputil.DumpResponse(resp, true); err == nil {
						resLogger.Printf("[res] [%s] %s", id, respDump)
					}

					return resp, err
				}))
			}
		}
	}

	return openai.NewClient(opts...)
}
