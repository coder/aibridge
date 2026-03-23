package utils

import "strings"

// ExtractBearerToken extracts the token from a "Bearer <token>"
// Authorization header value. Returns empty string if the header
// is not a valid Bearer token.
func ExtractBearerToken(auth string) string {
	auth = strings.TrimSpace(auth)
	if auth == "" {
		return ""
	}
	fields := strings.Fields(auth)
	if len(fields) == 2 && strings.EqualFold(fields[0], "Bearer") {
		return fields[1]
	}
	return ""
}
