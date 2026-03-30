package intercept

// Credential kind constants for interception recording.
const (
	CredentialKindCentralized    = "centralized"
	CredentialKindPersonalAPIKey = "personal_api_key"
	CredentialKindSubscription   = "subscription"
)

// CredentialFields is an embeddable helper that implements the
// SetCredential, CredentialKind, and CredentialHint methods of the
// Interceptor interface.
type CredentialFields struct {
	Kind string
	Hint string
}

func (c *CredentialFields) SetCredential(kind, hint string) {
	c.Kind = kind
	c.Hint = hint
}

func (c *CredentialFields) CredentialKind() string {
	return c.Kind
}

func (c *CredentialFields) CredentialHint() string {
	return c.Hint
}
