package integrationtest

import (
	"testing"

	"cdr.dev/slog/v3"
	"cdr.dev/slog/v3/sloggers/slogtest"
	"github.com/coder/aibridge/config"
)

// anthropicCfg creates a minimal Anthropic config for testing.
func anthropicCfg(url, key string) config.Anthropic {
	return config.Anthropic{
		BaseURL: url,
		Key:     key,
	}
}

func anthropicCfgWithAPIDump(url, key, dumpDir string) config.Anthropic {
	cfg := anthropicCfg(url, key)
	cfg.APIDumpDir = dumpDir
	return cfg
}

// testBedrockCfg returns a test AWS Bedrock config pointing at the given URL.
func testBedrockCfg(url string) *config.AWSBedrock {
	return &config.AWSBedrock{
		Region:          "us-west-2",
		AccessKey:       "test-access-key",
		AccessKeySecret: "test-secret-key",
		Model:           "beddel",  // This model should override the request's given one.
		SmallFastModel:  "modrock", // Unused but needed for validation.
		BaseURL:         url,
	}
}

// openAICfg creates a minimal OpenAI config for testing.
func openAICfg(url, key string) config.OpenAI {
	return config.OpenAI{
		BaseURL: url,
		Key:     key,
	}
}

func openaiCfgWithAPIDump(url, key, dumpDir string) config.OpenAI {
	cfg := openAICfg(url, key)
	cfg.APIDumpDir = dumpDir
	return cfg
}

// newLogger creates a test logger at Debug level.
func newLogger(t *testing.T) slog.Logger {
	t.Helper()
	return slogtest.Make(t, &slogtest.Options{}).Leveled(slog.LevelDebug)
}
