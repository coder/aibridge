package provider

import (
	"net/http"
	"testing"

	"github.com/stretchr/testify/assert"
)

func Test_anthropicIsFailure(t *testing.T) {
	t.Parallel()

	tests := []struct {
		statusCode int
		isFailure  bool
	}{
		{http.StatusOK, false},
		{http.StatusBadRequest, false},
		{http.StatusUnauthorized, false},
		{http.StatusTooManyRequests, true}, // 429
		{http.StatusInternalServerError, false},
		{http.StatusBadGateway, false},
		{http.StatusServiceUnavailable, true}, // 503
		{http.StatusGatewayTimeout, true},     // 504
		{529, true},                           // Anthropic Overloaded
	}

	for _, tt := range tests {
		assert.Equal(t, tt.isFailure, anthropicIsFailure(tt.statusCode), "status code %d", tt.statusCode)
	}
}
