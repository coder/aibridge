package utils_test

import (
	"testing"

	"github.com/coder/aibridge/utils"
	"github.com/stretchr/testify/require"
)

func TestExtractJSONField(t *testing.T) {
	t.Parallel()

	// Generics can't be neatly specified in table tests, so opting for imperative approach.

	require.Equal(t,
		utils.ExtractJSONField[bool]([]byte(`{"stream": true}`), "stream"),
		true,
	)
	require.Equal(t,
		utils.ExtractJSONField[bool]([]byte(`{}`), "stream"),
		false, // Zero value.
	)
	require.Equal(t,
		utils.ExtractJSONField[bool](nil, "stream"),
		false, // Zero value.
	)
	require.Equal(t,
		utils.ExtractJSONField[string](nil, "stream"),
		"", // Zero value.
	)
	require.Equal(t,
		utils.ExtractJSONField[map[string]any]([]byte(`{"owner":{"username": "admin"}}`), "owner"),
		map[string]any{"username": "admin"},
	)
	require.Equal(t,
		utils.ExtractJSONField[string]([]byte(`{"owner":{"username": "admin"}}`), "owner"),
		"", // Zero value since result could not be coerced to generic type.
	)
}
