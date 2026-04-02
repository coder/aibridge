package intercept

// Credential kind constants for interception recording.
const (
	CredentialKindCentralized    = "centralized"
	CredentialKindPersonalAPIKey = "personal_api_key"
	CredentialKindSubscription   = "subscription"
)

// CredentialFields holds credential metadata for an interception.
type CredentialFields struct {
	Kind string
	Hint string
}
