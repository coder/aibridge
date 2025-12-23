package testutil

import (
	"fmt"
	"testing"

	"github.com/tidwall/sjson"
)

// SetJSON sets a JSON value at the given key/path (tidwall/sjson syntax) and
// returns the updated JSON bytes.
func SetJSON(in []byte, key string, val any) ([]byte, error) {
	if len(in) == 0 {
		return nil, fmt.Errorf("empty JSON input")
	}
	if key == "" {
		return nil, fmt.Errorf("empty JSON key")
	}

	out, err := sjson.SetBytes(in, key, val)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func MustSetJSON(t testing.TB, in []byte, key string, val any) []byte {
	t.Helper()
	out, err := SetJSON(in, key, val)
	mustNoError(t, err, "set JSON")
	return out
}
