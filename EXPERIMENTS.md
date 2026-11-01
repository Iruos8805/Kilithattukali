# Experiments: status and how to run

| ID | Question | Status | Run |
|---|---|---|---|
| E1 | Do the Plus rules add strategic depth? | run, 3 seeds: Plus (all on together) is strongly defence-favoured; no depth metric computed; see RESULTS.md | `uv run python tools/pipeline.py --seed 1 --rules plus` then `aggregate_tables.py --prefix q` |
| E2 | Re-fit the balance with learned agents | balance v1 fitted on scripted bots; Optuna search tested at tiny size | `uv run kilithattu-eval balance --trials 40` (scripted); learned-agent version not written |
| E3 | Do policies transfer to other field sizes and team sizes? | run with strong agents: learned attacks beat the scripted crosser on most other fields (RESULTS.md section 0d) | `uv run python tools/generalisation.py --seeds 1 2 3 -n 32` |
| E4 | League vs naive self-play at equal compute | league run (3 seeds, defence-heavy and equal ratios; strong-fine-tune pipeline: see RESULTS.md section 0); plain self-play baseline not run | `kilithattu-train selfplay --config configs/train/selfplay.yaml` |
| E5 | Do team calls help? | not done: calls cannot be learned by imitation | needs a new recipe (league fine-tune with `p6_calls`) |
| E6 | Does FCP give a better teammate than self-play? | run with strong partners and matched controls (3 learners): FCP was a worse teammate in 7 of 9 cells; see RESULTS.md sections 0f-0g | see its docstring |
| E7 | Which tactics emerge? | run with strong agents (5 innings): fine-tuned attacks rush; league attacks spread out; detectors are crude (section 0e) | `uv run python tools/tactics.py --seeds 1 2 3` |
| - | Elo and Nash ratings of the population | run with strong agents: league pairs outrank the scripted pair; a single league pair is the Nash equilibrium (section 0c) | `uv run python tools/tournament.py --seeds 1 2 3 -n 32` |
| - | Does the reward setup matter? (gamma 0.999, no shaping) | run, 3 seeds each: no detectable difference at this power (RESULTS.md section 0h) | `finetune_attack.py ... -o ppo.gamma=0.999` or `ppo.shaping_beta=0.0` |

Order I would run them in: tournament, tactics, generalisation, FCP, Plus-rules pipeline.
