package utils

// MaskSecret returns the first 4 and last 4 characters of s
// separated by "...", or the full string if 8 characters or fewer.
func MaskSecret(s string) string {
	if len(s) <= 8 {
		return s
	}
	return s[:4] + "..." + s[len(s)-4:]
}
