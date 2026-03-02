package aibridge

import (
	"bytes"
	"io"
	"net/http"
	"regexp"
	"strings"

	"github.com/tidwall/gjson"
)

// guessSessionID attempts to retrieve a session ID which may have been sent by
// the client. We only attempt to retrieve sessions using methods recognized for
// the given client.
func guessSessionID(client Client, r *http.Request) string {
	headers := r.Header.Clone()
	payload, err := io.ReadAll(r.Body)
	if err != nil {
		// Failing silently is suitable here; if the body cannot be read, we won't be able to do much more.
		return ""
	}
	_ = r.Body.Close()

	// Restore the request body.
	r.Body = io.NopCloser(bytes.NewReader(payload))

	switch client {
	case ClientClaudeCode:
		/* Claude Code adds the session ID into the `metadata.user_id` field in the JSON body.
		{
			...
			"metadata": {
				"user_id": "user_{sha256}_account_{account_id}_session_{uuid_v4}"
			},
			...
		} */
		userID := gjson.GetBytes(payload, "metadata.user_id")
		if !userID.Exists() {
			return ""
		}

		matches := regexp.MustCompile(`_session_(.+)$`).FindStringSubmatch(userID.String())
		if len(matches) < 2 {
			return ""
		}
		return matches[1]
	case ClientCodex:
		return strings.TrimSpace(headers.Get("session_id"))
	case ClientMux:
		return strings.TrimSpace(headers.Get("X-Mux-Workspace-Id"))
	default:
		return ""
	}
}
