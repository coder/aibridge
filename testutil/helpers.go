package testutil

import (
	"fmt"
	"sort"
	"testing"
)

func sortedKeys[M ~map[string]V, V any](m M) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

func mustNoError(t testing.TB, err error, format string, args ...any) {
	t.Helper()
	if err == nil {
		return
	}
	prefix := ""
	if format != "" {
		prefix = fmt.Sprintf(format, args...) + ": "
	}
	t.Fatalf("%s%v", prefix, err)
}
