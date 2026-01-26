package intercept

import (
	"testing"

	"github.com/coder/aibridge/context"
	"github.com/coder/aibridge/recorder"
	"github.com/google/uuid"
	"github.com/stretchr/testify/require"
)

func TestNilActor(t *testing.T) {
	t.Parallel()

	require.Nil(t, ActorHeadersAsOpenAIOpts(nil))
	require.Nil(t, ActorHeadersAsAnthropicOpts(nil))
}

func TestBasic(t *testing.T) {
	t.Parallel()

	actorID := uuid.NewString()
	actor := &context.Actor{
		ID: actorID,
	}

	// We can't peek inside since these opts require an internal type to apply onto.
	// All we can do is check the length.
	// See TestActorHeaders for an integration test.
	oaiOpts := ActorHeadersAsOpenAIOpts(actor)
	require.Len(t, oaiOpts, 1)
	antOpts := ActorHeadersAsAnthropicOpts(actor)
	require.Len(t, antOpts, 1)
}

func TestBasicAndMetadata(t *testing.T) {
	t.Parallel()

	actorID := uuid.NewString()
	actor := &context.Actor{
		ID: actorID,
		Metadata: recorder.Metadata{
			"This": "That",
			"And":  "The other",
		},
	}

	// We can't peek inside since these opts require an internal type to apply onto.
	// All we can do is check the length.
	// See TestActorHeaders for an integration test.
	oaiOpts := ActorHeadersAsOpenAIOpts(actor)
	require.Len(t, oaiOpts, 1+len(actor.Metadata))
	antOpts := ActorHeadersAsAnthropicOpts(actor)
	require.Len(t, antOpts, 1+len(actor.Metadata))
}
