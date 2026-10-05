# ADR 0001: Monorepo Architecture with uv Workspace

## Status
Accepted

## Context
Kilithattu-RL comprises multiple layers:
1. Core formal rules engine and physics simulation
2. Scripted bots and neural agent architectures
3. Training pipelines (MAPPO, self-play league, FCP)
4. Evaluation and balancing suites
5. Real-time interactive game client with GUI and audio

We evaluated three repository structures:
- Single package
- Multi-repository (independent git repos for env, training, and game)
- Monorepo organized as a multi-package workspace (`uv workspace`)

## Decision
We adopt a **monorepo organized as a `uv` workspace with five packages**:
- `packages/kilithattu-core`
- `packages/kilithattu-agents`
- `packages/kilithattu-train`
- `packages/kilithattu-eval`
- `packages/kilithattu-game`

## Consequences
- **Pros**:
  - Single atomic commit / PR can update rules, agents, and game simultaneously without version mismatch.
  - One unified lockfile (`uv.lock`) guarantees reproducible dependency resolution across all packages.
  - Clean separation of concerns with enforced import boundaries via `import-linter`.
  - Game users and players do not need to install heavy training dependencies (Optax, W&B, Optuna).
- **Cons**:
  - Slightly more complex directory layout and CI configuration.
