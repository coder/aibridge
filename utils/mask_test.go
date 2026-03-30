package utils_test

import (
	"strings"
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
		{"short", "short", "...(5)..."},
		{"short_9_chars", "veryshort", "...(9)..."},
		{"medium_15_chars", "thisisquitelong", "th...(11)...ng"},
		{"long_api_key", "sk-ant-api03-abcdefgh", "sk-a...(13)...efgh"},
		{"unicode", "hélloworld🌍!", "hé...(8)...🌍!"},
		{"github_token", "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefgh", "ghp_...(30)...efgh"},
		{"jwt_300_chars", "eyJh" + strings.Repeat("a", 292) + "Xk8s", "eyJh...(292)...Xk8s"},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			assert.Equal(t, tc.expected, utils.MaskSecret(tc.input))
		})
	}
}
