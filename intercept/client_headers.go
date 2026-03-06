package intercept

import "net/http"

// hopByHopHeaders are connection-level headers specific to the connection
// between client and AI Bridge, not meant for the upstream.
// See https://www.rfc-editor.org/rfc/rfc2616#section-13.5.1.
var hopByHopHeaders = []string{
	"Connection",
	"Keep-Alive",
	"Proxy-Authenticate",
	"Proxy-Authorization",
	"Te",
	"Trailer",
	"Transfer-Encoding",
	"Upgrade",
}

// nonForwardedHeaders are headers that should not be forwarded to the upstream provider:
//   - Connection-specific headers managed by AI Bridge or the HTTP transport.
//   - Auth headers that are re-injected by the SDK from the provider configuration.
//   - User-Agent is set by the SDK to identify AI Bridge as the upstream client.
var nonForwardedHeaders = []string{
	"Host",
	"Accept-Encoding",
	"Content-Length",
	"Content-Type",
	"Authorization",
	"X-Api-Key",
	"User-Agent",
}

// SanitizeClientHeaders clones headers and returns a sanitized copy suitable
// for forwarding to an upstream provider.
//
// It removes:
//   - Hop-by-hop headers
//   - Non-forwarded headers (connection-specific, transport-managed, auth, and SDK-managed headers)
//
// Callers should apply these headers first, so that any subsequently
// added headers take priority in case of conflict.
func SanitizeClientHeaders(headers http.Header) http.Header {
	if headers == nil {
		return http.Header{}
	}

	outHeaders := headers.Clone()

	for _, h := range hopByHopHeaders {
		outHeaders.Del(h)
	}

	for _, h := range nonForwardedHeaders {
		outHeaders.Del(h)
	}

	return outHeaders
}
