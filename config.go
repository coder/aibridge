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
}

type Config struct {
	OpenAI    ProviderConfig
	Anthropic ProviderConfig
	Bedrock   AWSBedrockConfig
}
