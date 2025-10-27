package aibridge

type ProviderConfig struct {
	BaseURL, Key string
}

type (
	OpenAIConfig    ProviderConfig
	AnthropicConfig ProviderConfig
)

type AWSBedrockConfig struct {
	Region                     string
	AccessKey, AccessKeySecret string
	Model, SmallFastModel      string
	// EndpointOverride allows overriding the Bedrock endpoint URL for testing.
	// If set, requests will be sent to this URL instead of the default AWS Bedrock endpoint.
	EndpointOverride string
}

type Config struct {
	OpenAI    ProviderConfig
	Anthropic ProviderConfig
	Bedrock   AWSBedrockConfig
}
