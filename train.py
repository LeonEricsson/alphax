import os
import pickle
import time
from functools import partial
from typing import NamedTuple

import haiku as hk
import jax
import jax.numpy as jnp
import optax
import pgx
from omegaconf import OmegaConf
from pgx.experimental import act_randomly
from pgx.experimental import auto_reset
from pydantic import BaseModel
from torch.utils.tensorboard import SummaryWriter

from env.strike import Strike
from env.wrappers import auto_chance
from src.mcts.base import RecurrentFnOutput
from src.mcts.base import RootFnOutput
from src.mcts.policies import muzero_policy
from src.mcts.qtransforms import qtransform_by_min_max
from src.network import AZNet


os.environ['XLA_FLAGS'] = (
    '--xla_gpu_enable_triton_softmax_fusion=true '
    '--xla_gpu_triton_gemm_any=True '
    '--xla_gpu_enable_async_collectives=true '
    '--xla_gpu_enable_latency_hiding_scheduler=true '
    '--xla_gpu_enable_highest_priority_async_stream=true '
)

os.environ.update(
    {
        'NCCL_LL128_BUFFSIZE': '-2',
        'NCCL_LL_BUFFSIZE': '-2',
        'NCCL_PROTO': 'SIMPLE,LL,LL128',
    }
)

devices = jax.local_devices()
num_devices = len(devices)


class Sample(NamedTuple):
    obs: jnp.ndarray
    policy_tgt: jnp.ndarray
    value_tgt: jnp.ndarray
    mask: jnp.ndarray


class Config(BaseModel):
    env_id: pgx.EnvId = 'strike'
    scenario: str = 'mediumv2'
    seed: int = 42
    max_num_iters: int = 200
    num_channels: int = 64
    num_layers: int = 3
    resnet_v2: bool = True
    selfplay_batch_size: int = 1024     # num of parallel env
    num_simulations: int = 64
    max_num_steps: int = 64  # max env step when self playing
    training_batch_size: int = 4096
    # training_num_batches: int = 128
    lr_init: float = 0.001
    # lr_end: float = 0.0001
    # lr_steps: int = 100
    weight_decay: float = 1e-5
    # ['basic', 'standard', 'intermediate', 'advanced']
    eval_complexity: str = 'basic'
    eval_batch_size: int = 128
    eval_interval: int = 10
    ckpt_interval: int = 10
    num_eval_opponents: int = 2
    value_loss_scale: float = 1.0
    dirichlet_alpha: float = 0.5
    custom_value_target: bool = False
    # buffer will contain (num_devices x buffer_size x selfplay_batch_size x
    # max_num_steps) samples
    buffer_size: int = 8
    num_eval_simulations: int = 256
    name: str = None

    class Config:
        extra = 'forbid'


conf_dict = OmegaConf.from_cli()

print(conf_dict)

resume_run = conf_dict.get('resume_from')

if resume_run is not None:
    with open(resume_run, 'rb') as f:
        ckpt = pickle.load(f)

    resume = True
    config = ckpt['config']
    key = ckpt['key']
    model = ckpt['model']
    opt_state = ckpt['opt_state']
    iteration = ckpt['iteration']

    # Load the buffers and put them on their corresponding gpu
    buffer = ckpt['buffer']
    buffer = jax.tree_util.tree_map(
        lambda x: jax.device_put_sharded(
            [x[i] for i in range(num_devices)],
            devices,
        ),
        buffer,
    )

    print('Resuming from run with config:\n', config)

else:
    config: Config = Config(**conf_dict)

    resume = False

    buffer = None

    if config.name is None:
        raise ValueError(
            'The run must have a name (set argument name=[NAME] when running train.py)',
        )

    if config.eval_complexity not in [
            'basic', 'standard', 'intermediate', 'advanced']:
        raise ValueError(
            'The evaluation function must be one of: basic, standard, intermediate or advanced',
        )

    print(config)

if config.env_id == 'strike':
    env = Strike(config.scenario)
else:
    env = pgx.make(config.env_id)


SAMPLES_PER_ITERATION: int = config.selfplay_batch_size * config.max_num_steps
SAMPLES_PER_DEVICE: int = (
    config.selfplay_batch_size // num_devices
) * config.max_num_steps


def forward_fn(x, is_eval=False):
    net = AZNet(
        num_actions=env.num_actions,
        num_channels=config.num_channels,
        num_blocks=config.num_layers,
        resnet_v2=config.resnet_v2,
    )
    policy_out, value_out = net(
        x,
        is_training=not is_eval,
        test_local_stats=False,
    )
    return policy_out, value_out


forward = hk.without_apply_rng(hk.transform_with_state(forward_fn))


# transition_steps = config.lr_steps * \
#     ((config.buffer_size * SAMPLES_PER_DEVICE) // config.training_batch_size)

# lr_schedule = optax.schedules.linear_schedule(
#     init_value=config.lr_init,
#     end_value=config.lr_end,
#     transition_steps=transition_steps,
# )

# lr_schedule = optax.schedules.linear_onecycle_schedule(
#     transition_steps=transition_steps,
#     peak_value=config.lr_init,
#     div_factor=10.0,
#     final_div_factor=20.0
# )

lr_schedule = config.lr_init


optimizer = optax.adamw(
    learning_rate=lr_schedule,
    weight_decay=config.weight_decay,
)


def recurrent_fn(
    model,
    rng_key: jnp.ndarray,
    action: jnp.ndarray,
    state: pgx.State,
):
    # model: params
    # state: embedding
    del rng_key
    model_params, model_state = model

    current_player = state.current_player
    state = jax.vmap(env.step)(state, action)

    (logits, value), _ = forward.apply(
        model_params,
        model_state,
        state.observation,
        is_eval=True,
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
        reward=jax.lax.stop_gradient(reward),
        discount=jax.lax.stop_gradient(discount),
        prior_logits=jax.lax.stop_gradient(logits),
        value=jax.lax.stop_gradient(value),
        is_chance=jax.lax.stop_gradient(state.is_chance),
        is_terminal=jax.lax.stop_gradient(state.terminated),
    )
    return recurrent_fn_output, state


@jax.pmap
def init_env_states(rng_key, warmup_iterations=10):
    batch_size = config.selfplay_batch_size // num_devices
    rng_key, sub_key = jax.random.split(rng_key)
    keys = jax.random.split(sub_key, batch_size)
    states = jax.vmap(env.init)(keys)

    def body_fn(states, key):
        logits = jnp.where(states.legal_action_mask, 1.0,
                           jnp.finfo(jnp.float32).min)

        key, subkey = jax.random.split(key)
        action = jax.random.categorical(subkey, logits, axis=-1)

        key, subkey = jax.random.split(key)
        keys = jax.random.split(subkey, batch_size)
        states = jax.vmap(auto_reset(auto_chance(env.step), env.init))(
            states,
            action,
            keys,
        )

        return states, None

    rng_key, sub_key = jax.random.split(rng_key)
    keys = jax.random.split(subkey, warmup_iterations)

    states, _ = jax.lax.scan(body_fn, states, keys)

    return states


@jax.pmap
def selfplay(
    model,
    rng_key: jnp.ndarray,
    state,
) -> Sample:
    model_params, model_state = model
    batch_size = config.selfplay_batch_size // num_devices

    class StepFnOutput(NamedTuple):
        obs: jnp.ndarray
        reward: jnp.ndarray
        terminated: jnp.ndarray
        action_weights: jnp.ndarray
        discount: jnp.ndarray
        value_est: jnp.ndarray

    def step_fn(state, key) -> StepFnOutput:
        key1, key2 = jax.random.split(key)
        observation = state.observation

        (logits, value), _ = forward.apply(
            model_params,
            model_state,
            state.observation,
            is_eval=True,
        )
        root = RootFnOutput(prior_logits=logits, value=value, embedding=state)

        policy_output = muzero_policy(
            params=model,
            rng_key=key1,
            root=root,
            recurrent_fn=recurrent_fn,
            num_simulations=config.num_simulations,
            invalid_actions=~state.legal_action_mask,
            dirichlet_alpha=config.dirichlet_alpha,
        )

        actor = state.current_player
        keys = jax.random.split(key2, batch_size)
        state = jax.vmap(auto_reset(auto_chance(env.step), env.init))(
            state,
            policy_output.action,
            keys,
        )
        discount = -1.0 * jnp.ones_like(value)
        discount = jnp.where(state.current_player == actor, 1.0, discount)
        discount = jnp.where(state.terminated, 0.0, discount)
        return state, StepFnOutput(
            obs=jax.lax.stop_gradient(observation),
            action_weights=jax.lax.stop_gradient(policy_output.action_weights),
            reward=jax.lax.stop_gradient(
                state.rewards[jnp.arange(state.rewards.shape[0]), actor]
            ),
            terminated=jax.lax.stop_gradient(state.terminated),
            discount=jax.lax.stop_gradient(discount),
            value_est=jax.lax.stop_gradient(policy_output.value),
        )

    # Run selfplay for max_num_steps by batch
    key_seq = jax.random.split(rng_key, config.max_num_steps)
    state, data = jax.lax.scan(step_fn, state, key_seq)

    # If episode is truncated, there is no value target
    # So when we compute value loss, we need to mask it
    value_mask = jnp.cumsum(data.terminated[::-1, :], axis=0)[::-1, :] >= 1

    # Compute value target
    def body_fn(carry, i):
        ix = config.max_num_steps - i - 1
        v = data.reward[ix] + data.discount[ix] * carry
        return v, v

    _, z = jax.lax.scan(
        body_fn,
        jnp.zeros(batch_size),
        jnp.arange(config.max_num_steps),
    )
    z = z[::-1, :]

    if config.custom_value_target:
        q = data.value_est
        value_tgt = jnp.where(
            value_mask,
            (z + q) / 2,
            q,
        )
        value_mask = jnp.ones_like(value_tgt)
    else:
        value_tgt = z

    return state, Sample(
        obs=jnp.swapaxes(data.obs, 0, 1),
        policy_tgt=jnp.swapaxes(data.action_weights, 0, 1),
        value_tgt=jnp.swapaxes(value_tgt, 0, 1),
        mask=jnp.swapaxes(value_mask, 0, 1),
    )


def loss_fn(model_params, model_state, data: Sample):
    (logits, value), model_state = forward.apply(
        model_params,
        model_state,
        data.obs,
        is_eval=False,
    )

    logits = jax.nn.log_softmax(logits, axis=-1)
    min_val = jnp.finfo(logits.dtype).min
    policy_loss = jnp.sum(
        data.policy_tgt *
        (jnp.maximum(jnp.log(data.policy_tgt), min_val) - logits),
        axis=-1,
    )
    policy_loss = jnp.mean(policy_loss)

    value_loss = config.value_loss_scale * optax.l2_loss(value, data.value_tgt)
    # mask if the episode is truncated
    value_loss = jnp.mean(value_loss * data.mask)

    return policy_loss + value_loss, (model_state, policy_loss, value_loss)


@partial(jax.pmap, axis_name='i')
def train(model, opt_state, samples: Sample):

    # convert into minibatches of certain length
    num_updates = samples.obs.shape[0] // config.training_batch_size
    samples = jax.tree_util.tree_map(
        lambda x: x.reshape(
            (num_updates, config.training_batch_size, *x.shape[1:]),
        ),
        samples,
    )

    # Make batches
    def train_net(carry, data: Sample):
        model, opt_state = carry
        model_params, model_state = model
        grads, (model_state, policy_loss, value_loss) = jax.grad(loss_fn, has_aux=True)(
            model_params,
            model_state,
            data,
        )
        grads = jax.lax.pmean(grads, axis_name='i')
        updates, opt_state = optimizer.update(grads, opt_state, model_params)
        model_params = optax.apply_updates(model_params, updates)
        model = (model_params, model_state)
        return (model, opt_state), (policy_loss, value_loss)

    (model, opt_state), (policy_loss, value_loss) = jax.lax.scan(
        train_net,
        (model, opt_state),
        samples,
        unroll=4,
    )

    return model, opt_state, policy_loss.mean(), value_loss.mean()


def az_action(state, model, key) -> pgx.State:
    model_params, model_state = model
    (logits, value), _ = forward.apply(
        model_params,
        model_state,
        state.observation,
        is_eval=True,
    )
    root = RootFnOutput(prior_logits=logits, value=value, embedding=state)

    policy_output = muzero_policy(
        params=model,
        rng_key=key,
        root=root,
        recurrent_fn=recurrent_fn,
        num_simulations=config.num_eval_simulations,
        invalid_actions=~state.legal_action_mask,
        dirichlet_fraction=0.05,
    )

    return policy_output.action


def random_action(state, key) -> pgx.State:
    action = act_randomly(key, state.legal_action_mask)
    return action


def mcts_action(rng_key, state, batch_size):
    def policy(state):
        """Random policy."""
        logits = jnp.ones_like(state.legal_action_mask, dtype=jnp.float32)

        return jnp.where(
            state.legal_action_mask,
            logits,
            jnp.finfo(
                logits.dtype,
            ).min,
        )

    def value_function(state, rng_key):
        def rollout(state, rng_key):
            '''
            Simulate a game and returns the reward from the perspective of the initial player.
            '''
            def cond(a):
                state, _ = a
                return ~state.terminated

            def loop_fn(a):
                state, key = a
                key, subkey = jax.random.split(key)
                action = jax.random.categorical(subkey, policy(state))
                key, subkey = jax.random.split(key)
                state = auto_chance(env.step)(state, action, subkey)
                return state, key

            leaf, _ = jax.lax.while_loop(cond, loop_fn, (state, rng_key))

            return leaf.rewards[state.current_player]

        return rollout(state, rng_key).astype(jnp.float32)

    def root_fn(state, rng_key):
        return RootFnOutput(
            prior_logits=policy(state),
            value=value_function(state, rng_key),
            embedding=state,
        )

    def mcts_recurrent(
        _, rng_key, action, state,
    ):
        del _

        current_player = state.current_player
        state = jax.vmap(env.step)(state, action)

        logits = jax.vmap(policy)(state)

        reward = state.rewards[jnp.arange(
            state.rewards.shape[0]), current_player]

        keys = jax.random.split(rng_key, state.current_player.shape[0])
        value = jax.vmap(value_function)(state, keys)
        discount = -1.0 * jnp.ones_like(value)
        discount = jnp.where(state.current_player ==
                             current_player, 1.0, discount)
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

    key1, key2 = jax.random.split(rng_key)

    policy_output = muzero_policy(
        params=None,
        rng_key=key1,
        root=jax.vmap(root_fn)(
            state, jax.random.split(key2, batch_size)),
        recurrent_fn=mcts_recurrent,
        num_simulations=config.num_eval_simulations,
        max_depth=None,
        qtransform=partial(
            qtransform_by_min_max, min_value=-1, max_value=1),
        dirichlet_fraction=0.0,
    )

    return policy_output.action


@jax.pmap
def eval_advanced(current_model, opponents, key):
    """Evaluation using full AZ vs previous checkpoints."""
    agent = 0

    key, subkey = jax.random.split(key)
    batch_size = config.eval_batch_size // num_devices
    keys = jax.random.split(subkey, batch_size)

    state = jax.vmap(env.init)(keys)

    agent_is_blue = state.current_player == 0

    def step_fn(val):
        key, state, R = val

        key, subkey = jax.random.split(key)
        # current model plays in every environment
        p0_actions = az_action(state, current_model, subkey)

        # reshape state to enable vmap over the different opponent models
        s = jax.tree_util.tree_map(
            lambda x: x.reshape(
                config.num_eval_opponents,
                -1,
                *x.shape[1:],
            ),
            state,
        )
        key, subkey = jax.random.split(key)
        keys = jax.random.split(subkey, config.num_eval_opponents)
        p1_actions = jax.vmap(az_action)(s, opponents, keys).flatten()

        actions = jax.lax.select(
            state.current_player == agent,
            p0_actions,
            p1_actions,
        )

        key, subkey = jax.random.split(key)
        keys = jax.random.split(subkey, batch_size)
        state = jax.vmap(auto_chance(env.step))(state, actions, keys)

        R = R + state.rewards[jnp.arange(batch_size), agent]
        return (key, state, R)

    _, _, R = jax.lax.while_loop(
        lambda x: ~(x[1].terminated.all()),
        step_fn,
        (
            key,
            state,
            jnp.zeros(batch_size),
        ),
    )

    return R, agent_is_blue


@jax.pmap
def eval_intermediate(current_model, key):
    """Evaluation using full AZ vs MCTS."""
    agent = 0

    key, subkey = jax.random.split(key)
    batch_size = config.eval_batch_size // num_devices
    keys = jax.random.split(subkey, batch_size)

    state = jax.vmap(env.init)(keys)

    agent_is_blue = state.current_player == 0

    def step_fn(val):
        key, state, R = val

        key, subkey = jax.random.split(key)
        p0_actions = az_action(state, current_model, subkey)

        key, subkey = jax.random.split(key)
        p1_actions = mcts_action(subkey, state, batch_size)

        actions = jax.lax.select(
            state.current_player == agent,
            p0_actions,
            p1_actions,
        )

        key, subkey = jax.random.split(key)
        keys = jax.random.split(subkey, batch_size)
        state = jax.vmap(auto_chance(env.step))(state, actions, keys)

        R = R + state.rewards[jnp.arange(batch_size), agent]
        return (key, state, R)

    _, _, R = jax.lax.while_loop(
        lambda x: ~(x[1].terminated.all()),
        step_fn,
        (
            key,
            state,
            jnp.zeros(batch_size),
        ),
    )

    return R, agent_is_blue


@jax.pmap
def eval_standard(current_model, key):
    """Evaluation using full AZ vs random agent."""
    agent = 0

    key, subkey = jax.random.split(key)
    batch_size = config.eval_batch_size // num_devices
    keys = jax.random.split(subkey, batch_size)

    state = jax.vmap(env.init)(keys)

    agent_is_blue = state.current_player == 0

    def step_fn(val):
        key, state, R = val

        key, subkey = jax.random.split(key)
        p0_actions = az_action(state, current_model, subkey)

        key, subkey = jax.random.split(key)
        keys = jax.random.split(subkey, batch_size)
        p1_actions = jax.vmap(random_action)(state, keys).flatten()

        actions = jax.lax.select(
            state.current_player == agent,
            p0_actions,
            p1_actions,
        )

        key, subkey = jax.random.split(key)
        keys = jax.random.split(subkey, batch_size)
        state = jax.vmap(auto_chance(env.step))(state, actions, keys)

        R = R + state.rewards[jnp.arange(batch_size), agent]
        return (key, state, R)

    _, _, R = jax.lax.while_loop(
        lambda x: ~(x[1].terminated.all()),
        step_fn,
        (
            key,
            state,
            jnp.zeros(batch_size),
        ),
    )

    return R, agent_is_blue


@jax.pmap
def eval_basic(current_model, rng_key):
    """Evaluation using AZ network vs random agent."""
    agent = 0
    my_model_params, my_model_state = current_model

    key, subkey = jax.random.split(rng_key)
    batch_size = config.selfplay_batch_size // num_devices
    keys = jax.random.split(subkey, batch_size)
    state = jax.vmap(env.init)(keys)
    agent_is_blue = state.current_player == 0

    def body_fn(val):
        key, state, R = val
        (my_logits, _), _ = forward.apply(
            my_model_params,
            my_model_state,
            state.observation,
            is_eval=True,
        )

        opp_logits = jnp.ones_like(state.legal_action_mask)
        is_my_turn = (state.current_player == agent).reshape((-1, 1))
        logits = jnp.where(is_my_turn, my_logits, opp_logits)

        # mask illegal actions
        logits = jnp.where(
            state.legal_action_mask,
            logits,
            jnp.finfo(
                logits.dtype,
            ).min,
        )

        key, subkey = jax.random.split(key)
        action = jax.random.categorical(subkey, logits, axis=-1)
        key, subkey = jax.random.split(key)
        keys = jax.random.split(subkey, batch_size)
        state = jax.vmap(auto_chance(env.step))(state, action, keys)
        R = R + state.rewards[jnp.arange(batch_size), agent]
        return (key, state, R)

    _, _, R = jax.lax.while_loop(
        lambda x: ~(x[1].terminated.all()),
        body_fn,
        (
            key,
            state,
            jnp.zeros(batch_size),
        ),
    )
    return R, agent_is_blue


def evaluate(model, opponents, keys):

    eval_functions = {
        'basic': eval_basic,
        'standard': eval_standard,
        'intermediate': eval_intermediate,
        'advanced': partial(eval_advanced, opponents=opponents)
    }

    keys = jax.random.split(subkey, num_devices)
    R, agent_is_blue = eval_functions[config.eval_complexity](model, keys)

    return R, agent_is_blue


def opponent_tree(models):
    # combine model list into a pytree
    tree = jax.tree_util.tree_map(lambda x: x[None], models[0])
    for i in range(1, len(models)):
        tree = jax.tree_util.tree_map(
            lambda x, y: jnp.concatenate([x, y[None]], 0),
            tree,
            models[i],
        )


@jax.jit
def add_opponent(opponents, new_model):
    def roll_and_insert(opponents_param, new_param):
        return jnp.roll(opponents_param, shift=1, axis=0).at[0].set(new_param)

    return jax.tree_util.tree_map(roll_and_insert, opponents, new_model)


@jax.pmap
def init_buffer(samples):
    return jax.tree_util.tree_map(
        lambda x: jnp.zeros(
            (x.shape[0], x.shape[1] * config.buffer_size, *x.shape[2:]),
            dtype=x.dtype,
        ),
        samples,
    )


@jax.vmap
def update_value_targets(input_tgt, input_mask):
    """
    Steps through the targets and value masks to update value targets of previously
    unfinished environments
    """

    def body_fn(carry, input):
        prev_tgt, prev_mask = carry
        tgt, mask = input
        tgt = jax.lax.select(
            mask,
            tgt,
            prev_tgt,
        )
        mask = mask | prev_mask
        return (tgt, mask), (tgt, mask)

    return jax.lax.scan(
        body_fn,
        (0.0, False),
        (
            input_tgt,
            input_mask,
        ),
        reverse=True,
        unroll=4,
    )[1]


@partial(jax.pmap, donate_argnums=(0,))
def update_buffer(buffer, samples):

    # Insert new sample at the end of the buffer
    buffer = jax.tree_util.tree_map(
        lambda b, s: jnp.roll(
            b,
            axis=1,
            shift=-s.shape[1],
        )
        .at[:, -s.shape[1]:]
        .set(s),
        buffer,
        samples,
    )

    # Update the last value target and masks for unfinished environments
    updated_tgt, updated_mask = update_value_targets(
        buffer.value_tgt[-128:, :],
        buffer.mask[-128:, :],
    )

    return Sample(
        buffer.obs,
        buffer.policy_tgt,
        buffer.value_tgt.at[-128:, :].set(updated_tgt),
        buffer.mask.at[-128:, :].set(updated_mask),
    )


@partial(jax.pmap, in_axes=(0, None))
def fetch_from_buffer(buffer, idx):
    return jax.tree_util.tree_map(
        lambda x: x.reshape(-1, *x.shape[2:])[idx],
        buffer,
    )


if __name__ == '__main__':
    # Initialize model and opt_state

    if not resume:
        dummy_state = jax.vmap(
            env.init,
        )(
            jax.random.split(
                jax.random.PRNGKey(0),
                2,
            ),
        )
        dummy_input = dummy_state.observation
        model = forward.init(
            jax.random.PRNGKey(0),
            dummy_input,
        )  # (params, state)
        opt_state = optimizer.init(params=model[0])
        key = jax.random.PRNGKey(config.seed)
        iteration: int = 0

    opponents = None
    if config.eval_complexity == 'advanced':
        opponents = jax.tree_util.tree_map(
            lambda x: jnp.stack(
                [x] * config.num_eval_opponents,
                axis=0,
            ),
            model,
        )

    # replicates to all devices
    model, opt_state = jax.device_put_replicated((model, opt_state), devices)

    writer = SummaryWriter(log_dir=f'./runs/{config.name}')

    # Prepare checkpoint dir
    ckpt_dir = os.path.join('checkpoints', f'{config.env_id}_{config.name}')
    os.makedirs(ckpt_dir, exist_ok=True)

    with open(os.path.join(ckpt_dir, 'log.txt'), 'a') as f:
        f.write(str(config))
        f.write('\n')

    # Initialize logging dict

    log = {}

    # Evaluation
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, num_devices)
    env_state = init_env_states(keys)

    while True:
        if iteration % config.eval_interval == 0:
            time_pre_eval = time.perf_counter()

            # Evaluation
            key, subkey = jax.random.split(key)
            R, agent_is_blue = evaluate(model, opponents, keys)

            nrb = agent_is_blue.sum()
            nrr = R.size - nrb

            time_eval = time.perf_counter() - time_pre_eval

            log.update(
                {
                    f'eval/{config.eval_complexity}/avg_R': R.mean().item(),
                    f'eval/{config.eval_complexity}/avg_R_as_blue': R[agent_is_blue].mean().item(),
                    f'eval/{config.eval_complexity}/avg_R_as_red': R[~agent_is_blue].mean().item(),
                    f'eval/{config.eval_complexity}/win_rate': ((R > 0).sum() / R.size).item(),
                    f'eval/{config.eval_complexity}/draw_rate': ((R == 0).sum() / R.size).item(),
                    f'eval/{config.eval_complexity}/lose_rate': ((R < 0).sum() / R.size).item(),
                    'stats/time_eval_sec': time_eval,
                    f'eval/{config.eval_complexity}/win_rate_as_blue': (
                        (R > 0)[agent_is_blue].sum() / nrb
                    ).item(),
                    f'eval/{config.eval_complexity}/win_rate_as_red': (
                        (R > 0)[~agent_is_blue].sum() / nrr
                    ).item(),
                },
            )

        # Store checkpoints
        if iteration % config.ckpt_interval == 0:
            model_0, opt_state_0 = jax.tree_util.tree_map(
                lambda x: x[0],
                (model, opt_state),
            )
            with open(os.path.join(ckpt_dir, f'{iteration:06d}.ckpt'), 'wb') as f:
                dic = {
                    'config': config,
                    'key': key,
                    'model': jax.device_get(model_0),
                    'opt_state': jax.device_get(opt_state_0),
                    'iteration': iteration,
                    'buffer': jax.device_get(buffer),
                    'pgx.__version__': pgx.__version__,
                    'env_id': env.id,
                    'env_version': env.version,
                }
                pickle.dump(dic, f)

            # add to opponent buffer
            if config.eval_complexity == 'advanced':
                opponents = add_opponent(opponents, model_0)

        # Log to tensorboard
        for k, v in log.items():
            writer.add_scalar(f'{k}', v, iteration)
        writer.flush()

        if iteration >= config.max_num_iters:
            break

        log = {}
        st = time.perf_counter()

        # Selfplay
        key, subkey = jax.random.split(key)
        keys = jax.random.split(subkey, num_devices)

        env_state, samples = selfplay(
            model,
            keys,
            env_state,
        )  # (device, total_batch, ...)

        time_sample_generation = time.perf_counter() - st
        # Update buffer and then get new samples
        if buffer is None:
            buffer = init_buffer(samples)

        buffer = update_buffer(buffer, samples)

        # Sample indexes on CPU side
        key, subkey = jax.random.split(key)

        # max_index = min(floor((iteration + 1) / 2) + 1, config.buffer_size) * SAMPLES_PER_DEVICE

        max_index = min(iteration + 1, config.buffer_size) * SAMPLES_PER_DEVICE

        # Use replacement if we have not filled buffer yet
        with_replacement = max_index < config.buffer_size * SAMPLES_PER_DEVICE

        policy_loss = 0.0
        value_loss = 0.0

        # if config.training_num_batches * config.training_batch_size <=
        # max_index:
        idx = jax.random.choice(
            subkey,
            max_index,
            # shape=(config.training_num_batches * config.training_batch_size,),
            shape=(config.buffer_size * SAMPLES_PER_DEVICE,),
            # shape=(max_index * 2,),
            replace=True,
        )
        batch = fetch_from_buffer(buffer, -(idx + 1))

        # Train the model
        model, opt_state, policy_losses, value_losses = train(
            model,
            opt_state,
            batch,
        )
        policy_loss = policy_losses.mean().item()
        value_loss = value_losses.mean().item()

        et = time.perf_counter()
        time_iteration = et - st

        log.update(
            {
                'train/policy_loss': policy_loss,
                'train/value_loss': value_loss,
                'stats/fps': SAMPLES_PER_ITERATION / time_iteration,
                'stats/samples_per_sec': SAMPLES_PER_ITERATION / time_sample_generation,
                'stats/time_total_sec': time_iteration,
            },
        )

        iteration += 1


writer.close()
