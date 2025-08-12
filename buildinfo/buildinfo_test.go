package buildinfo_test

import (
	"testing"

	"github.com/coder/aibridge/buildinfo"
	"github.com/stretchr/testify/assert"
)

func TestBuildInfo(t *testing.T) {
	t.Run("Version", func(t *testing.T) {
		// Should return a non-empty version
		version := buildinfo.Version()
		assert.NotEmpty(t, version)
	})
}
