package config

import "github.com/coder/aibridge/circuitbreaker"

const (
	ProviderAnthropic = "anthropic"
	ProviderOpenAI    = "openai"
)

type Anthropic struct {
	BaseURL        string
	Key            string
	CircuitBreaker *circuitbreaker.Config
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
	BaseURL        string
	Key            string
	CircuitBreaker *circuitbreaker.Config
}
