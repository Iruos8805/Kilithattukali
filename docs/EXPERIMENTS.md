# Experiments: status and how to run

| ID | Question | Status | Run |
|---|---|---|---|
| E1 | Do the Plus rules add strategic depth? | run, 3 seeds: Plus (all on together) is strongly defence-favoured; no depth metric computed; see RESULTS.md | `uv run python tools/pipeline.py --seed 1 --rules plus` then `aggregate_tables.py --prefix q` |
| E2 | Re-fit the balance with learned agents | balance v1 fitted on scripted bots; Optuna search tested at tiny size | `uv run kilithattu-eval balance --trials 40` (scripted); learned-agent version not written |
| E3 | Do policies transfer to other field sizes and team sizes? | run: partial transfer (see RESULTS.md section 5) | `uv run python tools/generalisation.py --seeds 1 2 3 -n 32` |
| E4 | League vs naive self-play at equal compute | league run (3 seeds, defence-heavy and equal ratios; strong-fine-tune pipeline: see RESULTS.md section 0); plain self-play baseline not run | `kilithattu-train selfplay --config configs/train/selfplay.yaml` |
| E5 | Do team calls help? | not done: calls cannot be learned by imitation | needs a new recipe (league fine-tune with `p6_calls`) |
| E6 | Does FCP give a better teammate than self-play? | run twice; matched-budget rerun: FCP learner slightly ahead as a teammate (not significant), one seed | see its docstring |
| E7 | Which tactics emerge? | run (3 innings per agent): learned attacks rush, no decoys; detectors are crude | `uv run python tools/tactics.py --seeds 1 2 3` |
| - | Elo and Nash ratings of the population | run: ratings are non-transitive; scripted pair carries most of the Nash weight | `uv run python tools/tournament.py --seeds 1 2 3 -n 32` |

Order I would run them in: tournament, tactics, generalisation, FCP, Plus-rules pipeline.
