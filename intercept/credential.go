package intercept

// Credential kind constants for interception recording.
const (
	CredentialKindCentralized    = "centralized"
	CredentialKindPersonalAPIKey = "byok_api_key"
	CredentialKindSubscription   = "byok_subscription"
)

// CredentialFields holds credential metadata for an interception.
type CredentialFields struct {
	Kind string
	Hint string
}
