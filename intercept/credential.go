package intercept

import "github.com/coder/aibridge/utils"

// CredentialKind identifies how a request was authenticated.
// Keep in sync with the credential_kind enum in coderd's database.
type CredentialKind string

// Credential kind constants for interception recording.
const (
	CredentialKindCentralized    CredentialKind = "centralized"
	CredentialKindPersonalAPIKey CredentialKind = "byok_api_key"
	CredentialKindSubscription   CredentialKind = "byok_subscription"
)

// CredentialInfo holds credential metadata for an interception.
type CredentialInfo struct {
	Kind CredentialKind
	Hint string
}

// NewCredentialInfo creates a CredentialInfo from a raw credential.
// The credential is automatically masked before storage so that the
// original secret is never retained.
func NewCredentialInfo(kind CredentialKind, credential string) CredentialInfo {
	return CredentialInfo{
		Kind: kind,
		Hint: utils.MaskSecret(credential),
	}
}
