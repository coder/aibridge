package config

const (
	ProviderAnthropic = "anthropic"
	ProviderOpenAI    = "openai"
	ProviderAmp       = "amp"
)

type Anthropic struct {
	BaseURL string
	Key     string
}

type AWSBedrock struct {
	Region                     string
	AccessKey, AccessKeySecret string
	Model, SmallFastModel      string
	// EndpointOverride allows overriding the Bedrock endpoint URL for testing.
	// If set, requests will be sent to this URL instead of the default AWS Bedrock endpoint.
	EndpointOverride string
}

type OpenAI struct {
	BaseURL string
	Key     string
}

type Amp struct {
	BaseURL string
	Key     string
}
