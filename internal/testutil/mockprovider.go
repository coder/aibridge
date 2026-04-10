package testutil

import (
	"fmt"
	"net/http"

	"go.opentelemetry.io/otel/trace"

	"github.com/coder/aibridge/config"
	"github.com/coder/aibridge/intercept"
)

type MockProvider struct {
	NameStr         string
	URL             string
	Bridged         []string
	Passthrough     []string
	InterceptorFunc func(w http.ResponseWriter, r *http.Request, tracer trace.Tracer) (intercept.Interceptor, error)
}

func (m *MockProvider) Type() string                               { return m.NameStr }
func (m *MockProvider) Name() string                               { return m.NameStr }
func (m *MockProvider) BaseURL() string                            { return m.URL }
func (m *MockProvider) RoutePrefix() string                        { return fmt.Sprintf("/%s", m.NameStr) }
func (m *MockProvider) BridgedRoutes() []string                    { return m.Bridged }
func (m *MockProvider) PassthroughRoutes() []string                { return m.Passthrough }
func (*MockProvider) AuthHeader() string                           { return "Authorization" }
func (*MockProvider) InjectAuthHeader(_ *http.Header)              {}
func (*MockProvider) CircuitBreakerConfig() *config.CircuitBreaker { return nil }
func (*MockProvider) APIDumpDir() string                           { return "" }
func (m *MockProvider) CreateInterceptor(w http.ResponseWriter, r *http.Request, tracer trace.Tracer) (intercept.Interceptor, error) {
	if m.InterceptorFunc != nil {
		return m.InterceptorFunc(w, r, tracer)
	}
	return nil, nil
}
