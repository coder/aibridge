package intercept

import "github.com/coder/aibridge/utils"

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

// NewCredentialInfo creates a CredentialInfo from a raw credential.
// The credential is automatically masked before storage so that the
// original secret is never retained.
func NewCredentialInfo(kind, credential string) CredentialInfo {
	return CredentialInfo{
		Kind: kind,
		Hint: utils.MaskSecret(credential),
	}
}
