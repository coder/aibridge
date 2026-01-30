package aibridge

import (
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"time"

	"cdr.dev/slog/v3"
	"github.com/coder/aibridge/metrics"
	"github.com/coder/aibridge/provider"
	"github.com/coder/aibridge/tracing"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
)

// newPassthroughRouter returns a simple reverse-proxy implementation which will be used when a route is not handled specifically
// by a [intercept.Provider].
func newPassthroughRouter(provider provider.Provider, logger slog.Logger, m *metrics.Metrics, tracer trace.Tracer) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if m != nil {
			m.PassthroughCount.WithLabelValues(provider.Name(), r.URL.Path, r.Method).Add(1)
		}

		ctx, span := tracer.Start(r.Context(), "Passthrough", trace.WithAttributes(
			attribute.String(tracing.PassthroughURL, r.URL.Path),
			attribute.String(tracing.PassthroughMethod, r.Method),
		))
		defer span.End()

		upURL, err := url.Parse(provider.BaseURL())
		if err != nil {
			logger.Warn(ctx, "failed to parse provider base URL", slog.Error(err))
			http.Error(w, "request error", http.StatusBadGateway)
			return
		}

		// Append the request path to the upstream base path.
		reqPath, err := url.JoinPath(upURL.Path, r.URL.Path)
		if err != nil {
			logger.Warn(ctx, "failed to join upstream path", slog.Error(err), slog.F("upstreamPath", upURL.Path), slog.F("requestPath", r.URL.Path))
			http.Error(w, "failed to join upstream path", http.StatusInternalServerError)
			return
		}
		// Ensure leading slash.
		if len(reqPath) == 0 || reqPath[0] != '/' {
			reqPath = "/" + reqPath
		}

		// Build a reverse proxy to the upstream.
		proxy := &httputil.ReverseProxy{
			Director: func(req *http.Request) {
				// Set scheme/host to upstream.
				req.URL.Scheme = upURL.Scheme
				req.URL.Host = upURL.Host
				req.URL.Path = reqPath
				req.URL.RawPath = ""

				// Preserve query string.
				req.URL.RawQuery = r.URL.RawQuery

				// Set Host header for upstream.
				req.Host = upURL.Host

				// Copy headers from client.
				req.Header = r.Header.Clone()

				// Standard proxy headers.
				host, _, herr := net.SplitHostPort(r.RemoteAddr)
				if herr != nil {
					host = r.RemoteAddr
				}
				if prior := req.Header.Get("X-Forwarded-For"); prior != "" {
					req.Header.Set("X-Forwarded-For", prior+", "+host)
				} else {
					req.Header.Set("X-Forwarded-For", host)
				}
				req.Header.Set("X-Forwarded-Host", r.Host)
				if r.TLS != nil {
					req.Header.Set("X-Forwarded-Proto", "https")
				} else {
					req.Header.Set("X-Forwarded-Proto", "http")
				}
				// Avoid default Go user-agent if none provided.
				if _, ok := req.Header["User-Agent"]; !ok {
					req.Header.Set("User-Agent", "aibridge") // TODO: use build tag.
				}

				// Inject provider auth.
				provider.InjectAuthHeader(&req.Header)
			},
			ErrorHandler: func(rw http.ResponseWriter, req *http.Request, e error) {
				logger.Warn(req.Context(), "reverse proxy error", slog.Error(e), slog.F("path", req.URL.Path))
				http.Error(rw, "upstream proxy error", http.StatusBadGateway)
			},
		}

		// Transport tuned for streaming (no response header timeout).
		proxy.Transport = &http.Transport{
			Proxy:                 http.ProxyFromEnvironment,
			ForceAttemptHTTP2:     true,
			MaxIdleConns:          100,
			IdleConnTimeout:       90 * time.Second,
			TLSHandshakeTimeout:   10 * time.Second,
			ExpectContinueTimeout: 1 * time.Second,
		}

		proxy.ServeHTTP(w, r)
	}
}
