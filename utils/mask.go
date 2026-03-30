package utils

import "fmt"

// MaskSecret masks the middle of a secret string, revealing a small
// prefix and suffix for identification. The number of hidden characters
// is embedded in the masked portion (e.g. "sk-a...(21)...efgh").
// The number of characters revealed scales with string length.
func MaskSecret(s string) string {
	if s == "" {
		return ""
	}

	runes := []rune(s)
	reveal := revealLength(len(runes))

	// If we'd reveal everything or more, mask it all.
	if reveal*2 >= len(runes) {
		return fmt.Sprintf("...(%d)...", len(runes))
	}

	prefix := string(runes[:reveal])
	suffix := string(runes[len(runes)-reveal:])
	masked := len(runes) - reveal*2
	return prefix + fmt.Sprintf("...(%d)...", masked) + suffix
}

// revealLength returns the number of runes to show at each end.
func revealLength(n int) int {
	switch {
	case n >= 20:
		return 4
	case n >= 10:
		return 2
	default:
		return 0
	}
}
