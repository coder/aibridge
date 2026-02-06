package testutil

import (
	"fmt"
	"net/http"

	"github.com/coder/aibridge/config"
	"github.com/coder/aibridge/intercept"
	"go.opentelemetry.io/otel/trace"
)

type MockProvider struct {
	Name_           string
	URL             string
	Bridged         []string
	Passthrough     []string
	InterceptorFunc func(w http.ResponseWriter, r *http.Request, tracer trace.Tracer) (intercept.Interceptor, error)
}

func (m *MockProvider) Name() string                                 { return m.Name_ }
func (m *MockProvider) BaseURL() string                              { return m.URL }
func (m *MockProvider) RoutePrefix() string                          { return fmt.Sprintf("/%s", m.Name_) }
func (m *MockProvider) BridgedRoutes() []string                      { return m.Bridged }
func (m *MockProvider) PassthroughRoutes() []string                  { return m.Passthrough }
func (m *MockProvider) AuthHeader() string                           { return "Authorization" }
func (m *MockProvider) InjectAuthHeader(h *http.Header)              {}
func (m *MockProvider) CircuitBreakerConfig() *config.CircuitBreaker { return nil }
func (m *MockProvider) CreateInterceptor(w http.ResponseWriter, r *http.Request, tracer trace.Tracer) (intercept.Interceptor, error) {
	if m.InterceptorFunc != nil {
		return m.InterceptorFunc(w, r, tracer)
	}
	return nil, nil
}
