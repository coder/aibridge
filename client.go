package aibridge

import (
	"net/http"
	"strings"
)

const (
	// Possible values for the "client" field in interception records.
	// Must be kept in sync with documentation: https://github.com/coder/coder/blob/90c11f3386578da053ec5cd9f1475835b980e7c7/docs/ai-coder/ai-bridge/monitoring.md?plain=1#L36-L44
	ClientClaude     = "Claude Code"
	ClientCodex      = "Codex"
	ClientZed        = "Zed"
	ClientCopilotVSC = "GitHub Copilot (VS Code)"
	ClientCopilotCLI = "GitHub Copilot (CLI)"
	ClientKilo       = "Kilo Code"
	ClientRoo        = "Roo Code"
	ClientCursor     = "Cursor"
	ClientUnknown    = "Unknown"
)

// guessClient attempts to guess the client application from the request headers.
// Not all clients set proper user agent headers, so this is a best-effort approach.
// Based on https://github.com/coder/aibridge/issues/20#issuecomment-3769444101.
func guessClient(r *http.Request) string {
	userAgent := strings.ToLower(r.UserAgent())
	originator := r.Header.Get("originator")

	// Must be kept in sync with documentation: https://github.com/coder/coder/blob/90c11f3386578da053ec5cd9f1475835b980e7c7/docs/ai-coder/ai-bridge/monitoring.md?plain=1#L36-L44
	switch {
	case strings.HasPrefix(userAgent, "claude"):
		return ClientClaude
	case strings.HasPrefix(userAgent, "codex"):
		return ClientCodex
	case strings.HasPrefix(userAgent, "zed/"):
		return ClientZed
	case strings.HasPrefix(userAgent, "githubcopilotchat/"):
		return ClientCopilotVSC
	case strings.HasPrefix(userAgent, "copilot/"):
		return ClientCopilotCLI
	case strings.HasPrefix(userAgent, "kilo-code/") || originator == "kilo-code":
		return ClientKilo
	case strings.HasPrefix(userAgent, "roo-code/") || originator == "roo-code":
		return ClientRoo
	case strings.HasPrefix(userAgent, "copilot"):
		return ClientCursor
	case r.Header.Get("x-cursor-client-version") != "":
		return ClientCursor
	}
	return ClientUnknown
}
