package apidump

// sensitiveRequestHeaders are headers that should be redacted from request dumps.
var sensitiveRequestHeaders = map[string]struct{}{
	"Authorization":        {},
	"X-Api-Key":            {},
	"Api-Key":              {},
	"X-Auth-Token":         {},
	"Cookie":               {},
	"Proxy-Authorization":  {},
	"X-Amz-Security-Token": {},
}

// sensitiveResponseHeaders are headers that should be redacted from response dumps.
// Note: header names use Go's canonical form (http.CanonicalHeaderKey).
var sensitiveResponseHeaders = map[string]struct{}{
	"Set-Cookie":         {},
	"Www-Authenticate":   {},
	"Proxy-Authenticate": {},
}

// redactHeaderValue redacts a sensitive header value, showing only partial content.
// For values >= 8 bytes: shows first 4 and last 4 bytes with "..." in between.
// For values < 8 bytes: shows first and last byte with "..." in between.
func redactHeaderValue(value string) string {
	if len(value) >= 8 {
		return value[:4] + "..." + value[len(value)-4:]
	}
	if len(value) >= 2 {
		return value[:1] + "..." + value[len(value)-1:]
	}
	// Single character or empty - just return as-is
	return value
}
