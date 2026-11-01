# Kilithattu-RL

A formal rules engine, reinforcement-learning stack and playable game for **Kilithattu Kali**, the Kerala Onam
line-tag game. See `PLAN.md` / `PROPOSAL.md` for the research plan and `RULES.md` for the implemented rules.

## Packages
| Package | Contents |
|---|---|
| `kilithattu-core` | JAX environment (Classic + P1-P4, P6), rules, replays, plain-Python reference for differential tests |
| `kilithattu-agents` | entity-attention actor/critic, scripted bots, checkpoints, difficulty ladder, human-fair wrapper (P7) |
| `kilithattu-train` | MAPPO, curriculum, self-play league with PFSP and exploiters, FCP partners, behaviour cloning |
| `kilithattu-eval` | matches, Elo and Nash ratings, tournaments, exploitability, behaviour detectors, Optuna balance search, plots |
| `kilithattu-game` | pygame client: play as crosser, guard or Kili against scripted or trained bots; replays |

`web/kilithattu.html` is a small single-file browser version of the game (scripted bots, no install).

## Install
```bash
pip install -r requirements.txt
pip install -e packages/kilithattu-core -e packages/kilithattu-agents -e packages/kilithattu-game
pip install -e packages/kilithattu-train -e packages/kilithattu-eval --no-deps
```
Or `uv sync`. GPU: `pip install "jax[cuda12]"`.

## Play
```bash
kilithattu --side attack --difficulty medium         # run as a crosser (WASD/arrows, Shift dash, Tab switch)
kilithattu --side kili --difficulty easy             # run the Kili (hold a direction, you turn at junctions)
kilithattu --side watch --rules plus                 # watch the bots, Kilithattu+ rules
kilithattu --headless-ticks 400 --side watch         # no window, prints the result
```

## Train and evaluate
```bash
kilithattu-train curriculum --config configs/train/smoke.yaml      # tiny run
kilithattu-train curriculum --config configs/train/curriculum.yaml # real run (use a GPU)
kilithattu-train league --config configs/train/league.yaml
kilithattu-train fcp --config configs/train/fcp.yaml
kilithattu-eval match scripted:crosser scripted:guard
kilithattu-eval verify-replay replays/<game>.kreplay
```
Experiments E1-E7 live in `experiments/` with configs in `configs/eval/`.

## Reproducing the results
```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
for s in 1 2 3; do uv run python tools/pipeline.py --seed $s --ratio1 25:25 --ratio2 25:25 --tag _eq; done   # about 1 h per seed on a small GPU
uv run python tools/aggregate_tables.py --tag _eq        # mean +- std over seeds
uv run python tools/make_figures.py                      # heatmaps and a markdown results table in docs/figures
uv run python tools/make_ladder.py --seed 1              # point the game's easy/medium/hard levels at trained checkpoints
uv run kilithattu                                        # menu: pick role, attack/defence policy, rules
```
The pipeline per seed: DAgger imitation of the scripted attack -> PPO fine-tune -> DAgger imitation of the scripted guards ->
two league iterations -> cross-table. Other experiments: `tools/fcp_experiment.py` (E6), `tools/tournament.py` (Elo and Nash),
`tools/tactics.py` (E7 detectors), `tools/generalisation.py` (E3), `pipeline.py --rules plus` (E1). See `docs/EXPERIMENTS.md`.

## Status: what has and has not been verified
**Verified by tests:** the rules engine (37 tests, including differential tests against a second implementation), agents,
game (menu, replays, trained and scripted opponents), and the training package's unit and smoke tests.

**Run at full size on a GPU (3 seeds, 128 innings per cell; details in `docs/RESULTS.md`):** the imitation -> PPO -> league pipeline, first with an
early-stopped fine-tune stage and then with a proper 1,200-update stage. With the proper stage: the fine-tuned attack beats the scripted crosser against the
scripted guards in every seed (39.6 to 112.5 points against 20.3, very large spread); the league defence holds fine-tuned-type attackers to 0.6-7 points, also
across seeds and against an attack that was in no training pool; the league attack is more robust than the attack it started from (worst case 29 against 4)
and beats the scripted crosser in 3 of 3 seeds, at the cost of peak strength; league defences are on par with or worse than the scripted guards against league
attackers. An earlier conclusion that the league does not improve the attack came from the early-stopped start and was reversed. FCP with strong partners and a matched
self-play control was a worse teammate in 7 of 9 cells (mean -6.5 points; three learners, one of them started from the self-play policy): an honest negative result.
Changing the discount or removing reward shaping in the fine-tune stage made no detectable difference (3 seeds each, low power). The Plus rules (all on together) were strongly defence-favoured; that run used the
early-stopped pipeline.

**Rerun on the strong agents:** tournament (league pairs outrank the scripted pair; one league pair is the Nash equilibrium of the population), generalisation (learned attacks beat the scripted crosser on most other fields), behaviour detectors (fine-tuned attacks rush; league attacks spread out). **Not yet rerun with strong agents:** FCP, Plus rules. **Tested at tiny size only:** the Optuna balance search (E2).
**Not done:** team calls (E5: calls cannot be imitated, so it needs a different recipe), human study, a rules check with people who play the game.

**Known limits:** one training recipe and hyperparameter set; scripted bots are the only fixed references; scripted bots assume full
observability (do not use with P3); round trips are rare in learned play, so points mostly come from outbound lines; Classic balance v1 was
tuned on scripted bots only; checkpoints use flax msgpack rather than Orbax; the earlier single-chain results (48.7-point attack, 595-point
exploit) were single-seed pilots and were not reproduced across seeds.
