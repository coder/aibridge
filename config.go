package aibridge

import "go.uber.org/atomic"

type ProviderConfig struct {
	baseURL, key          atomic.String
	upstreamLoggingDir    atomic.String
	enableUpstreamLogging atomic.Bool
}

// NewProviderConfig creates a new ProviderConfig with the given values.
func NewProviderConfig(baseURL, key, upstreamLoggingDir string) *ProviderConfig {
	cfg := &ProviderConfig{}
	cfg.baseURL.Store(baseURL)
	cfg.key.Store(key)
	cfg.upstreamLoggingDir.Store(upstreamLoggingDir)
	return cfg
}

// BaseURL returns the base URL for the provider.
func (c *ProviderConfig) BaseURL() string {
	return c.baseURL.Load()
}

// SetBaseURL sets the base URL for the provider.
func (c *ProviderConfig) SetBaseURL(baseURL string) {
	c.baseURL.Store(baseURL)
}

// Key returns the API key for the provider.
func (c *ProviderConfig) Key() string {
	return c.key.Load()
}

// SetKey sets the API key for the provider.
func (c *ProviderConfig) SetKey(key string) {
	c.key.Store(key)
}

// UpstreamLoggingDir returns the base directory for upstream logging.
// If empty, the OS's tempdir will be used.
// Logs are written to $UpstreamLoggingDir/$provider/$model/$interceptionID.{req,res}.log
func (c *ProviderConfig) UpstreamLoggingDir() string {
	return c.upstreamLoggingDir.Load()
}

// SetUpstreamLoggingDir sets the base directory for upstream logging.
func (c *ProviderConfig) SetUpstreamLoggingDir(dir string) {
	c.upstreamLoggingDir.Store(dir)
}

// SetEnableUpstreamLogging enables or disables upstream logging at runtime.
func (c *ProviderConfig) SetEnableUpstreamLogging(enabled bool) {
	c.enableUpstreamLogging.Store(enabled)
}

// IsUpstreamLoggingEnabled returns whether upstream logging is currently enabled.
func (c *ProviderConfig) IsUpstreamLoggingEnabled() bool {
	return c.enableUpstreamLogging.Load()
}
