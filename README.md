## alphax
a training and inference wrapper around [mctx](https://github.com/google-deepmind/mctx).

extended features:

- SPMD allowing parallel data-collection (self-play) and training across multiple XLA devices. 
- circular buffer to hold memory samples (replay buffer)
- an array of evaluation functions 
    - AZ Net v. Random
    - AZ v. Random
    - AZ v. MCTS
    - AZ v. AZ
- resume training from checkpoint
- stochastic AlphaZero with chance nodes (branch `stochastic_alpha_zero`)



## install

1. install [poetry](https://python-poetry.org/docs/)
2. inside repo `poetry install`
3. `poetry shell`
4. install [JAX](https://jax.readthedocs.io/en/latest/installation.html) 0.4.30 with `poetry run pip3 install ...` (system dependent)
