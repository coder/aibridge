package config

import "time"

const (
	ProviderAnthropic = "anthropic"
	ProviderOpenAI    = "openai"
)

// CircuitBreaker holds configuration for circuit breakers.
type CircuitBreaker struct {
	// MaxRequests is the maximum number of requests allowed in half-open state.
	MaxRequests uint32
	// Interval is the cyclic period of the closed state for clearing internal counts.
	Interval time.Duration
	// Timeout is how long the circuit stays open before transitioning to half-open.
	Timeout time.Duration
	// FailureThreshold is the number of consecutive failures that triggers the circuit to open.
	FailureThreshold uint32
	// IsFailure determines if a status code should count as a failure.
	// If nil, defaults to DefaultIsFailure.
	IsFailure func(statusCode int) bool
	// OpenErrorResponse returns the response body when the circuit is open.
	// This should match the provider's error format.
	OpenErrorResponse func() []byte
}

// DefaultCircuitBreaker returns sensible defaults for circuit breaker configuration.
func DefaultCircuitBreaker() CircuitBreaker {
	return CircuitBreaker{
		FailureThreshold: 5,
		Interval:         10 * time.Second,
		Timeout:          30 * time.Second,
		MaxRequests:      3,
	}
}

type Anthropic struct {
	BaseURL        string
	Key            string
	APIDumpDir     string
	CircuitBreaker *CircuitBreaker
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
	APIDumpDir     string
	CircuitBreaker *CircuitBreaker
}
