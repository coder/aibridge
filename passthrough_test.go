package aibridge

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"cdr.dev/slog/v3/sloggers/slogtest"
	"github.com/coder/aibridge/internal/testutil"
	"github.com/stretchr/testify/assert"
	"go.opentelemetry.io/otel"
)

var testTracer = otel.Tracer("bridge_test")

func TestPassthroughRoutes(t *testing.T) {
	t.Parallel()

	upstreamRespBody := "upstream response"
	tests := []struct {
		name              string
		baseURLPath       string
		passthroughRoute  string
		expectRequestPath string
		expectRespStatus  int
		expectRespBody    string
	}{
		{
			name:              "passthrough_route_no_path",
			passthroughRoute:  "/v1/conversations",
			expectRequestPath: "/v1/conversations",
			expectRespStatus:  http.StatusOK,
			expectRespBody:    upstreamRespBody,
		},
		{
			name:              "base_URL_path_is_preserved_in_passthrough_routes",
			baseURLPath:       "/api/v2",
			passthroughRoute:  "/v1/models",
			expectRequestPath: "/api/v2/v1/models",
			expectRespStatus:  http.StatusOK,
			expectRespBody:    upstreamRespBody,
		},
		{
			name:             "passthrough_route_break_parse_base_url",
			baseURLPath:      "/%zz",
			passthroughRoute: "/v1/models/",
			expectRespStatus: http.StatusBadGateway,
			expectRespBody:   "request error",
		},
		{
			name:             "passthrough_route_break_join_path",
			baseURLPath:      "/%25",
			passthroughRoute: "/v1/models",
			expectRespStatus: http.StatusInternalServerError,
			expectRespBody:   "failed to join upstream path",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			logger := slogtest.Make(t, nil)

			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				assert.Equal(t, tc.expectRequestPath, r.URL.Path)
				w.WriteHeader(http.StatusOK)
				_, _ = w.Write([]byte(upstreamRespBody))
			}))
			t.Cleanup(upstream.Close)

			prov := &testutil.MockProvider{
				URL: upstream.URL + tc.baseURLPath,
			}

			handler := newPassthroughRouter(prov, logger, nil, testTracer)

			req := httptest.NewRequest("", tc.passthroughRoute, nil)
			resp := httptest.NewRecorder()
			handler.ServeHTTP(resp, req)

			assert.Equal(t, tc.expectRespStatus, resp.Code)
			assert.Contains(t, resp.Body.String(), tc.expectRespBody)
		})
	}
}
