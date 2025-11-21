package aibridge

import (
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"time"

	"cdr.dev/slog"
)

// newPassthroughRouter returns a simple reverse-proxy implementation which will be used when a route is not handled specifically
// by a [Provider].
func newPassthroughRouter(provider Provider, logger slog.Logger, metrics *Metrics) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if metrics != nil {
			metrics.PassthroughCount.WithLabelValues(provider.Name(), r.URL.Path, r.Method).Add(1)
		}

		upURL, err := url.Parse(provider.BaseURL())
		if err != nil {
			logger.Warn(r.Context(), "failed to parse provider base URL", slog.Error(err))
			http.Error(w, "request error", http.StatusBadGateway)
			return
		}

		// Build a reverse proxy to the upstream.
		proxy := &httputil.ReverseProxy{
			Director: func(req *http.Request) {
				// Set scheme/host to upstream.
				req.URL.Scheme = upURL.Scheme
				req.URL.Host = upURL.Host

				// Preserve the stripped path from the incoming request and ensure leading slash.
				p := r.URL.Path
				if len(p) == 0 || p[0] != '/' {
					p = "/" + p
				}
				req.URL.Path = p
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
