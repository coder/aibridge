package utils_test

import (
	"testing"

	"github.com/coder/aibridge/utils"
	"github.com/stretchr/testify/assert"
)

func TestMaskSecret(t *testing.T) {
	t.Parallel()

	tests := []struct {
		input    string
		expected string
	}{
		{"", ""},
		{"short", "short"},
		{"exactly8", "exactly8"},
		{"sk-ant-api03-abcdefgh", "sk-a...efgh"},
		{"sk-ant-oat01-abcdefghijklmnop", "sk-a...mnop"},
	}

	for _, tc := range tests {
		t.Run(tc.input, func(t *testing.T) {
			t.Parallel()
			assert.Equal(t, tc.expected, utils.MaskSecret(tc.input))
		})
	}
}
