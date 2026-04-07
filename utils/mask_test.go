package utils_test

import (
	"testing"

	"github.com/coder/aibridge/utils"
	"github.com/stretchr/testify/assert"
)

func TestMaskSecret(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{"empty", "", ""},
		{"short", "short", "***"},
		{"short_9_chars", "veryshort", "***"},
		{"medium_15_chars", "thisisquitelong", "th***ng"},
		{"long_api_key", "sk-ant-api03-abcdefgh", "sk-a***efgh"},
		{"unicode", "hélloworld🌍!", "hé***🌍!"},
		{"github_token", "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefgh", "ghp_***efgh"},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			assert.Equal(t, tc.expected, utils.MaskSecret(tc.input))
		})
	}
}
