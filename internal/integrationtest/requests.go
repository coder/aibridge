package integrationtest

import (
	"github.com/coder/aibridge/config"
)

// apiKey is the default API key used across integration tests.
const apiKey = "api-key"

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
