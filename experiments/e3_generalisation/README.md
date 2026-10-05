# Experiment E3: Generalisation Across Procedural Fields

## Objective
Answer **RQ3**: Can policies trained on specific field geometries ({2×5, 2×7}) transfer zero-shot to unseen interpolated ({2×6}) and extrapolated ({3×7, 2×9}) dimensions?

## Metrics
- **Normalised Held-Out PPI**: ratio of held-out score to baseline in-distribution score.
- **Hypothesis**: $\ge 80\%$ performance retention on interpolated sizes, declining gracefully on extrapolated sizes.

## Running
```bash
uv run python experiments/e3_generalisation/run_e3.py --innings 10 --seed 42
```
