# Contributing to Kilithattu-RL

Thank you for contributing to Kilithattu-RL!

## Architecture Boundaries

The codebase is organized as a `uv` workspace of five distinct packages:
- `kilithattu-core`: Pure rules and simulation. Must **never** depend on agents, train, eval, or game.
- `kilithattu-agents`: Scripted bots and model inference. Depends only on `kilithattu-core`.
- `kilithattu-train`: MAPPO, curriculum, and league training. Depends on `kilithattu-core` and `kilithattu-agents`.
- `kilithattu-eval`: Evaluation, Elo rating, and tournament runners. Depends on `kilithattu-core`, `kilithattu-agents`, and `kilithattu-train`.
- `kilithattu-game`: Interactive pygame client. Depends on `kilithattu-core` and `kilithattu-agents`.

These contracts are enforced in CI via `import-linter`.

## Development Workflow

1. Install dependencies:
   ```bash
   uv sync
   ```

2. Run test suites:
   ```bash
   uv run pytest
   ```

3. Code formatting and linting:
   ```bash
   uv run ruff check --fix
   uv run ruff format
   ```

4. Conventional Commits:
   Prefix commit messages with package scope:
   - `feat(core): ...`
   - `test(core): ...`
   - `feat(agents): ...`
   - `feat(train): ...`
   - `feat(game): ...`
   - `docs: ...`

## Adding a Rule Toggle

When introducing a new rule toggle or modifying rule behaviour:
1. Update `RULES.md` documenting rationale and parameter defaults.
2. Update `kilithattu_core.config.EnvParams`.
3. Add implementation in `kilithattu_core.rules` and differential tests in `kilithattu_core.tests`.
4. Bump `ENV_VERSION` appropriately.
