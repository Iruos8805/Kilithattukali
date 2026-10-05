# Experiment E1: Rule Depth and Ablations

## Objective
Answer **RQ1**: Does adding tactical features (stamina, facing cones, fog of war, dynamic claps) enhance game-theoretic depth and competitiveness without breaking balance?

## Methodology
We evaluate 6 rule configurations:
1. **Classic**: Base rules R1–R14.
2. **Classic+P1**: Acceleration limits and continuous stamina depletion/regeneration.
3. **Classic+P2**: Facing cones and discrete turning delays.
4. **Classic+P3**: Fog-of-war partial observability.
5. **Classic+P4**: Dynamic Kili clapping window.
6. **Plus**: Full Kilithattu+ suite (P1–P4 enabled).

## Running
```bash
uv run python experiments/e1_rule_ablation/run_e1.py --innings 20 --seed 42
```
