package buildinfo_test

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/coder/aibridge/buildinfo"
)

func TestBuildInfo(t *testing.T) {
	t.Parallel()

	t.Run("Version", func(t *testing.T) {
		t.Parallel()
		// Should return a non-empty version
		version := buildinfo.Version()
		assert.NotEmpty(t, version)
	})
}
