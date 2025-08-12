package aibridge

import "context"

type actorContextKey struct{}

type actor struct {
	id       string
	metadata Metadata
}

func AsActor(ctx context.Context, actorID string, metadata Metadata) context.Context {
	return context.WithValue(ctx, actorContextKey{}, &actor{id: actorID, metadata: metadata})
}

func actorFromContext(ctx context.Context) *actor {
	a, ok := ctx.Value(actorContextKey{}).(*actor)
	if !ok {
		return nil
	}

	return a
}
