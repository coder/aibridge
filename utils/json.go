package utils

import "encoding/json"

// ExtractJSONField extracts the stream flag from JSON
func ExtractJSONField[T any](raw []byte, field string) T {
	var zero T
	var data map[string]any
	if err := json.Unmarshal(raw, &data); err != nil {
		return zero
	}

	if stream, ok := data[field].(T); ok {
		return stream
	}
	return zero
}
