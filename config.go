package aibridge

type ProviderConfig struct {
	BaseURL, Key string
	// EnableUpstreamLogging enables logging of upstream API requests and responses to /tmp/$provider.log
	EnableUpstreamLogging bool
}

type Config struct {
	OpenAI    ProviderConfig
	Anthropic ProviderConfig
}
