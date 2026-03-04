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
	case ClientZed:
		return "" // Zed does not send a session ID from Zed Agent or Text Thread.
	case ClientCopilotVSC:
		// This does not map precisely to what we consider a session, but it's close enough.
		// Most other providers' equivalent of this would persist for the duration of a
		// conversation; it does seem to persist across an agentic loop though, which is
		// all we really need.
		//
		// There's also `vscode-sessionid` but that's persistent for the duration of the
		// VS Code window.
		return strings.TrimSpace(headers.Get("x-interaction-id"))
	case ClientCopilotCLI:
		return strings.TrimSpace(headers.Get("X-Client-Session-Id"))
	case ClientKilo:
		return strings.TrimSpace(headers.Get("X-KILOCODE-TASKID"))
	case ClientRoo:
		return "" // RooCode doesn't send a session ID.
	case ClientCursor:
		return "" // Cursor is not currently supported.
	default:
		return ""
	}
}
