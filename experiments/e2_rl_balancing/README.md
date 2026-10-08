# Experiment E2: RL and Empirical Balancing

## Objective
Answer **RQ2**: Can automated balance search determine physical parameters (speed ratios, tag reach, row depth) that ensure robust, two-sided viability?

## Competitive Criteria
- **Points per Innings (PPI)**: in [7.0, 14.0] (allowing 1–3 round trips per innings).
- **All-Out Rate**: in [20%, 80%] (defenders regularly claim tags without total denial).

## Running
```bash
uv run python experiments/e2_rl_balancing/run_e2.py --trials 6 --seed 42
```
