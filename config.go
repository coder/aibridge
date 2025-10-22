package aibridge

import "sync/atomic"

type ProviderConfig struct {
	BaseURL, Key string
	// UpstreamLoggingDir specifies the base directory for upstream logging.
	// If empty, os.TempDir() will be used.
	// Logs are written to $UpstreamLoggingDir/$provider/$model/$id.{req,res}.log
	UpstreamLoggingDir string
	// enableUpstreamLogging enables logging of upstream API requests and responses.
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
