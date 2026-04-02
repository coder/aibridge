package intercept

// Credential kind constants for interception recording.
const (
	CredentialKindCentralized    = "centralized"
	CredentialKindPersonalAPIKey = "byok_api_key"
	CredentialKindSubscription   = "byok_subscription"
)

// CredentialInfo holds credential metadata for an interception.
type CredentialInfo struct {
	Kind string
	Hint string
}
