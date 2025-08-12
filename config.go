package aibridge

type ProviderConfig struct {
	BaseURL, Key string
}

type Config struct {
	OpenAI    ProviderConfig
	Anthropic ProviderConfig
}
