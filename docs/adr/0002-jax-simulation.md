# ADR 0002: Vectorised JAX Environment for Core Rules Engine

## Status
Accepted

## Context
Kilithattu-RL requires simulating tens of thousands of games for multi-agent reinforcement learning (MAPPO and self-play league). Traditional CPU-bound step loops in standard Python / PettingZoo would constrain training throughput to ~10³–10⁴ steps/sec, making league self-play prohibitively slow on modest hardware.

## Decision
We implement `kilithattu-core` as a **purely functional, vectorised JAX simulation engine**:
- States and configurations are structured JAX Pytrees.
- Step transitions and observation functions are stateless and compilable with `jax.jit`.
- Batch simulation across thousands of environments runs via `jax.vmap` / `jax.lax.scan`.
- Determinism is achieved through explicit PRNG key management (`jax.random.PRNGKey`).
- A slow, pure-Python reference implementation is maintained solely for differential testing.

## Consequences
- **Pros**:
  - Millions of env steps per second on modern hardware; fast CPU execution for interactive game play.
  - Zero Python GIL bottlenecks during training rollouts.
  - Bit-for-bit determinism across re-simulations for exact replay verification.
- **Cons**:
  - Dynamic branching and variable-sized arrays require fixed max-dimension padding with existence masks.
