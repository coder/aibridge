package aibridge

import (
	"bufio"
	"io"
	"strconv"
	"strings"
	"sync"
)

const (
	sseEventTypeMessage = "message"
	sseEventTypeError   = "error"
	sseEventTypePing    = "ping"
)

type sseEvent struct {
	Type  string
	Data  string
	ID    string
	Retry int
}

type sseParser struct {
	events map[string][]sseEvent
	mu     sync.RWMutex
}

func NewSSEParser() *sseParser {
	return &sseParser{
		events: make(map[string][]sseEvent),
	}
}

func (p *sseParser) Parse(reader io.Reader) error {
	scanner := bufio.NewScanner(reader)

	var currentEvent sseEvent
	var dataLines []string

	for scanner.Scan() {
		line := scanner.Text()

		// Empty line indicates end of event
		if line == "" {
			if len(dataLines) > 0 {
				currentEvent.Data = strings.Join(dataLines, "\n")
			}

			// Default to message type if no event type specified
			if currentEvent.Type == "" {
				currentEvent.Type = sseEventTypeMessage
			}

			// Store the event
			p.mu.Lock()
			p.events[currentEvent.Type] = append(p.events[currentEvent.Type], currentEvent)
			p.mu.Unlock()

			// Reset for next event
			currentEvent = sseEvent{}
			dataLines = nil
			continue
		}

		// Skip comments
		if strings.HasPrefix(line, ":") {
			continue
		}

		// Parse field:value format
		if colonIndex := strings.Index(line, ":"); colonIndex != -1 {
			field := line[:colonIndex]
			value := line[colonIndex+1:]

			// Remove leading space from value if present
			if len(value) > 0 && value[0] == ' ' {
				value = value[1:]
			}

			switch field {
			case "event":
				currentEvent.Type = value
			case "data":
				dataLines = append(dataLines, value)
			case "id":
				currentEvent.ID = value
			case "retry":
				if retryMs, err := strconv.Atoi(value); err == nil {
					currentEvent.Retry = retryMs
				}
			}
		}
	}

	return scanner.Err()
}

func (p *sseParser) EventsByType(eventType string) []sseEvent {
	p.mu.RLock()
	defer p.mu.RUnlock()

	events := p.events[eventType]
	result := make([]sseEvent, len(events))
	copy(result, events)
	return result
}

func (p *sseParser) MessageEvents() []sseEvent {
	return p.EventsByType(sseEventTypeMessage)
}

func (p *sseParser) AllEvents() map[string][]sseEvent {
	p.mu.RLock()
	defer p.mu.RUnlock()

	result := make(map[string][]sseEvent)
	for eventType, events := range p.events {
		eventsCopy := make([]sseEvent, len(events))
		copy(eventsCopy, events)
		result[eventType] = eventsCopy
	}
	return result
}
