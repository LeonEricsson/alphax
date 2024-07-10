import pickle
from typing import NamedTuple, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import pgx
from jax_tqdm import scan_tqdm
from omegaconf import OmegaConf
from pgx.experimental import auto_reset
from pydantic import BaseModel
from jaxtyping import Array

from src.mcts.base import RecurrentFnOutput
from src.mcts.base import RootFnOutput
from src.mcts.policies import muzero_policy
from src.network import AZNet

KeyArray = Array
Model = Tuple[hk.Params, hk.State]
State = pgx.State

class Sample(NamedTuple):
    obs: jnp.ndarray
    policy_tgt: jnp.ndarray
    value_tgt: jnp.ndarray
    mask: jnp.ndarray


class Config(BaseModel):
    ckpt: str = 'checkpoints/...'
    seed: int = 0
    eval_simulations: int = 800

    class Config:
        extra = 'forbid'


conf_dict = OmegaConf.from_cli()
config: Config = Config(**conf_dict)
print('Running eval with parameters:', config)

ckpt_path = config.ckpt

with open(ckpt_path, 'rb') as f:
    ckpt = pickle.load(f)

model = ckpt['model']

ckpt_config = ckpt['config']

env = pgx.make(ckpt_config.env_id)
step = jax.jit(auto_reset(env.step, env.init))


def forward_fn(x: Array, is_eval=False):
    net = AZNet(
        num_actions=env.num_actions,
        num_channels=ckpt_config.num_channels,
        num_blocks=ckpt_config.num_layers,
        resnet_v2=ckpt_config.resnet_v2,
    )
    policy_out, value_out = net(
        x, is_training=not is_eval, test_local_stats=False,
    )
    return policy_out, value_out


forward = hk.without_apply_rng(hk.transform_with_state(forward_fn))


def recurrent_fn(
    model: Model, rng_key: KeyArray,
    action: Array, state: State,
):
    # model: params
    # state: embedding
    del rng_key
    model_params, model_state = model

    current_player = state.current_player
    state = jax.vmap(env.step)(state, action)

    (logits, value), _ = forward.apply(
        model_params, model_state, state.observation, is_eval=True,
    )
    # mask invalid actions
    logits = logits - jnp.max(logits, axis=-1, keepdims=True)

    logits = jnp.where(
        jnp.expand_dims(state.is_chance, axis=-1),
        state._chance_probs,
        logits,
    )

    logits = jnp.where(
        state.legal_action_mask,
        logits,
        jnp.finfo(
            logits.dtype,
        ).min,
    )

    reward = state.rewards[jnp.arange(state.rewards.shape[0]), current_player]
    value = jnp.where(state.terminated, 0.0, value)
    discount = -1.0 * jnp.ones_like(value)
    discount = jnp.where(state.current_player == current_player, 1.0, discount)
    discount = jnp.where(state.terminated, 0.0, discount)

    recurrent_fn_output = RecurrentFnOutput(
        reward=reward,
        discount=discount,
        prior_logits=logits,
        value=value,
        is_chance=state.is_chance,
        is_terminal=state.terminated,
    )
    return recurrent_fn_output, state


@jax.jit
def step_fn(state: State, key: KeyArray) -> State:
    key1, key2 = jax.random.split(key)

    state = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, 0), state)

    (logits, value), _ = forward.apply(
        model_params, model_state, state.observation, is_eval=True,
    )
    root = RootFnOutput(prior_logits=logits, value=value, embedding=state)

    policy_output = muzero_policy(
        params=model,
        rng_key=key1,
        root=root,
        recurrent_fn=recurrent_fn,
        num_simulations=config.eval_simulations,
        invalid_actions=~state.legal_action_mask,
        dirichlet_fraction=0.05,
    )

    state = jax.tree_util.tree_map(lambda x: jnp.squeeze(x, 0), state)
    state = step(state, policy_output.action[0], key2)

    return state, policy_output.action[0]


if __name__ == '__main__':
    import time
    key = jax.random.key(config.seed)

    model_params, model_state = model

    key, subkey = jax.random.split(key)

    state = env.init(subkey)

    episode_len = 256

    print('Running episode...')

    @scan_tqdm(episode_len)
    def body_fn(val, _):
        key, state = val
        key, subkey = jax.random.split(key)

        new_state, action = step_fn(state, subkey)

        return (key, new_state), (state, action)

    start = time.perf_counter()
    _, sequence = jax.lax.scan(body_fn, (key, state), jnp.arange(episode_len))
    end = time.perf_counter()
    print(f'Elapsed time: {end - start:.2f} s')
