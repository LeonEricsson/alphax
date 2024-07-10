from typing import Optional
from jaxtyping import Array

import jax
import jax.numpy as jnp
from pgx.core import State

KeyArray = Array

def auto_chance(step_fn):
    """
    This automatically skips all chance nodes in environment and samples the chances
    according to their probabilities directly.
    """

    def wrapped_step_fn(
        state: State, action: Array,
        key: Optional[KeyArray] = None,
    ):
        assert key is not None, (
            'Please specify PRNGKey at the third argument.'
        )

        key1, key2 = jax.random.split(key)
        state = step_fn(state, action, key1)

        def cond_fn(val):
            state, _ = val
            return (state.is_chance & ~state.terminated)

        def body_fn(val):
            state, key = val
            key, subkey = jax.random.split(key)

            # Mask away the non active chance actions
            chance_probs = jnp.where(
                state.legal_action_mask,
                state._chance_probs,
                jnp.finfo(state._chance_probs.dtype).min,
            )
            chance_outcome = jax.random.categorical(subkey, chance_probs)
            key, subkey = jax.random.split(key)
            state = step_fn(state, chance_outcome, subkey)
            return state, key

        # Skip past all potential chance nodes
        state, _ = jax.lax.while_loop(
            cond_fn,
            body_fn,
            (state, key2),
        )
        return state

    return wrapped_step_fn
