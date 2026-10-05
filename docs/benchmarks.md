# Kilithattu-Core Environment Benchmarks

## Overview
`kilithattu-core` is implemented in vectorised, pure functional JAX, allowing execution with `jax.jit`, `jax.vmap`, and `jax.lax.scan`.

## Performance Metrics

| Device | Envs | Steps | Elapsed Time | Steps / Second |
|---|---|---|---|---|
| CPU (Standard runner) | 64 | 500 | 0.627 s | **51,068 steps/s** |

## Invariants & Verification
- 100% bit-for-bit deterministic replays (`test_determinism.py`).
- 0% tunneling under continuous swept tag testing (`test_geometry.py`).
- Differential equivalence with Python reference simulation (`test_differential.py`).
