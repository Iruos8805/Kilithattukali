# Experiment E5: Team Communication Calls (P6)

## Objective
Answer **RQ5**: Do discrete intra-team calls improve coordination, prevent box-sharing fouls, and increase delivery rates compared to uncoordinated play?

## Treatments
- **Calls Active**: Communication tokens shared with teammates.
- **Calls Disabled**: Communication head masked to silence.
- **Calls Scrambled**: Communication channels injected with uniform noise to test semantic reliance.

## Running
```bash
uv run python experiments/e5_team_calls/run_e5.py --innings 10 --seed 42
```
