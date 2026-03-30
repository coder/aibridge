package context

import (
	"context"

	"github.com/coder/aibridge/recorder"
)

type (
	actorContextKey        struct{}
	originalHostContextKey struct{}
)

type Actor struct {
	ID       string
	Metadata recorder.Metadata
}

func AsActor(ctx context.Context, actorID string, metadata recorder.Metadata) context.Context {
	return context.WithValue(ctx, actorContextKey{}, &Actor{ID: actorID, Metadata: metadata})
}

func ActorFromContext(ctx context.Context) *Actor {
	a, ok := ctx.Value(actorContextKey{}).(*Actor)
	if !ok {
		return nil
	}

	return a
}

// WithOriginalHost stores the original destination host in the context.
func WithOriginalHost(ctx context.Context, host string) context.Context {
	return context.WithValue(ctx, originalHostContextKey{}, host)
}

// OriginalHostFromContext retrieves the original destination host from the context.
// Returns an empty string if not set.
func OriginalHostFromContext(ctx context.Context) string {
	h, _ := ctx.Value(originalHostContextKey{}).(string)
	return h
}

// ActorIDFromContext safely extracts the actor ID from the context.
// Returns an empty string if no actor is found.
func ActorIDFromContext(ctx context.Context) string {
	if actor := ActorFromContext(ctx); actor != nil {
		return actor.ID
	}
	return ""
}
