## alphax
a jax-native training and inference framework for alphazero, built around [mctx](https://github.com/google-deepmind/mctx).

features:

- SPMD, enabling parallel data-collection (self-play) and training across multiple XLA devices.
- fully JIT-composable. 
- circular buffer to hold memory samples (replay buffer).
- a lineup of evaluation functions 
    - AZ Net v. Random
    - AZ v. Random
    - AZ v. MCTS
    - AZ v. AZ (League based)
- stochastic alphazero with chance nodes (on branch `stochastic_alpha_zero`).
- resume training from checkpoint.

## install

1. install [poetry](https://python-poetry.org/docs/)
2. inside repo `poetry install`
3. `poetry shell`
4. install [JAX](https://jax.readthedocs.io/en/latest/installation.html) 0.4.30 with `poetry run pip3 install ...` (system dependent)

## usage

training

`python3 src/train.py name=test env_id=connect_four`

inference

`python3 src/inference.py ckpt="checkpoints/connect_four_test/000000.ckpt" eval_simulations=1600`


## caveats
**stochastic alphazero** 

the `recurrent_fn` expects the State to hold two extra attributes:

```
class State:
    
    # everything else ...

    - _chance_probs: the chance node probabilities across all actions
    - is_chance: if the current node is a chance node
    
    _chance_probs: jnp.ndarray
    is_chance: bool 
```

**jax** 

jax is volatile, get's updated frequently. things will most likely crash if you don't use the intended version

**environment**

this is built for [PGX](https://github.com/sotetsuk/pgx) environments. that being said swapping out for your own
completely custom environment should be straight forward as long as you implement the following:

```
class State(ABC):
    """
    Base state class for Pgx game environments.
    
    Key attributes:
    - current_player: ID of agent to play
    - observation: Current state observation
    - rewards: Intermediate rewards for each agent
    - terminated: Whether the state is terminal
    - truncated: Whether the episode was truncated
    - legal_action_mask: Boolean array of legal actions
    """
    
    current_player: jnp.ndarray
    observation: jnp.ndarray
    rewards: jnp.ndarray
    terminated: jnp.ndarray
    truncated: jnp.ndarray
    legal_action_mask: jnp.ndarray

class Env(ABC):
    """
    Base environment class for Pgx games.
    
    Key properties:
    - id: Environment identifier
    - num_actions: Size of action space
    - num_players: Number of players
    - observation_shape: Shape of observation
    - version: Environment version
    
    Key methods:
    - init: Initialize the environment state
    - step: Perform an action and get the next state
    - observe: Get observation for a specific player
    """

    @property
    @abstractmethod
    def id(self) -> str:
        pass

    @property
    def num_actions(self) -> int:
        pass

    @property
    @abstractmethod
    def num_players(self) -> int:
        pass

    @property
    def observation_shape(self) -> Tuple[int, ...]:
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        pass

    @abstractmethod
    def init(self, key: jnp.ndarray) -> 'State':
        pass

    @abstractmethod
    def step(self, state: 'State', action: jnp.ndarray, key: jnp.ndarray = None) -> 'State':
        pass

    @abstractmethod
    def observe(self, state: 'State', player_id: int) -> jnp.ndarray:
        pass
```

See [docs](https://sotets.uk/pgx/) for more info, or the [Tic-Tac-Toe](https://github.com/sotetsuk/pgx/blob/main/pgx/tic_tac_toe.py) implementation.
