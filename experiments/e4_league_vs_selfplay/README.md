# Experiment E4: Self-Play League vs Naive Self-Play

## Objective
Answer **RQ4**: Does an AlphaStar-style league with prioritized fictitious self-play (PFSP) and exploiter agents reduce exploitability and cyclic strategy forgetting compared to naive self-play?

## Key Metrics
- **Exploitability Proxy**: maximum payoff gain of an exploiter against the target policy mixture.
- **Nash Equilibrium Support**: number of distinct active strategies in the Nash distribution.

## Running
```bash
uv run python experiments/e4_league_vs_selfplay/run_e4.py --iterations 25 --seed 42
```
