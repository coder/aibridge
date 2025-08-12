package mcp_test

import (
	"testing"

	"github.com/coder/aibridge/mcp"
	"github.com/stretchr/testify/assert"
)

func TestGetClientInfo(t *testing.T) {
	info := mcp.GetClientInfo()

	assert.Equal(t, "coder/aibridge", info.Name)
	assert.NotEmpty(t, info.Version)
	// Version will either be a git revision, a semantic version, or a combination
	assert.NotEqual(t, "", info.Version)
}