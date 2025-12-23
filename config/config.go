package config

const (
	ProviderAnthropic = "anthropic"
	ProviderOpenAI    = "openai"
)

type AnthropicConfig struct {
	BaseURL string
	Key     string
}

type AWSBedrockConfig struct {
	Region                     string
	AccessKey, AccessKeySecret string
	Model, SmallFastModel      string
	// EndpointOverride allows overriding the Bedrock endpoint URL for testing.
	// If set, requests will be sent to this URL instead of the default AWS Bedrock endpoint.
	EndpointOverride string
}

type OpenAIConfig struct {
	BaseURL string
	Key     string
}
