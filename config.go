package aibridge

import "sync/atomic"

type ProviderConfig struct {
	BaseURL, Key string
	// EnableUpstreamLogging enables logging of upstream API requests and responses to /tmp/$provider.log
	enableUpstreamLogging atomic.Bool
}

// SetEnableUpstreamLogging enables or disables upstream logging at runtime.
func (c *ProviderConfig) SetEnableUpstreamLogging(enabled bool) {
	c.enableUpstreamLogging.Store(enabled)
}

// EnableUpstreamLogging returns whether upstream logging is currently enabled.
func (c *ProviderConfig) EnableUpstreamLogging() bool {
	return c.enableUpstreamLogging.Load()
}
