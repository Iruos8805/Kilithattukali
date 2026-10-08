# Experiment E6: Human Teaming and Fictitious Co-Play

## Objective
Answer **RQ6**: Does training with a diverse partner distribution (FCP) yield policies that coordinate better with unseen partners and human players than naive self-play?

## Key Metrics
- **Cross-Play Team Score**: Points achieved when paired with held-out, non-standard teammate behaviors.
- **Hypothesis**: FCP policies exhibit significantly higher zero-shot coordination resilience than pure self-play agents.

## Running
```bash
uv run python experiments/e6_human_teaming/run_e6.py --innings 10 --seed 42
```
