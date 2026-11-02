# Kilithattu-RL: Multi-Agent Reinforcement Learning for a Traditional Kerala Tag Game

*Project report · \[Team members and registration IDs\] · \[Guide\] · \[Institution, course, date\]*

## Abstract

Kilithattu Kali is a traditional line-tag game played in Kerala at Onam, with no formal or digital model. We (i) formalise its rules as a zero-sum two-team Markov game with heterogeneous agents, implemented as a deterministic, vectorised JAX environment that is tested against an independent reference implementation; (ii) train attack and defence policies with MAPPO, imitation of hand-written bots (DAgger) and a self-play league; and (iii) provide a playable game. Over three training seeds, an imitation-then-PPO attack beats the hand-written attack (39.6 to 112.5 points per innings against 20.3), a league defence holds rushing attackers to 0.6–7 points, including attackers from other seeds, and league training makes the attack more robust (worst case 29.2 ± 11.6 against 3.8 ± 3.1 for its starting point). Negative or null findings are reported as such: Fictitious Co-Play did not give a better teammate (worse in 7 of 9 cells), an extended rule set was strongly defence-favoured, and reward ablations made no detectable difference. All conclusions rest on three seeds with large seed-to-seed variance.

## 1. Introduction

Kilithattu Kali is played on a rectangle divided by chalk lines. Attackers try to cross every line, reach the far end, and come back; defenders guard the lines and tag. It is a sequential decision problem with delayed reward, two adversarial teams, and teammates who must coordinate. There is no dataset of expert play, so supervised learning has nothing to copy, and any fixed strategy can be exploited by an opponent that adapts. Reinforcement learning fits naturally.

Contributions:

1. A formal, executable model of the game (`kilithattu-core`), deterministic and vectorised on GPU, with a replay format and 37 passing tests including differential tests against a second implementation.
2. A training stack (`kilithattu-agents`, `kilithattu-train`, `kilithattu-eval`): entity-attention actor and centralised critic, MAPPO, imitation of scripted bots, and a self-play league with prioritised opponent sampling.
3. An empirical study over three seeds, including held-out evaluation, a tournament, generalisation to other fields, teaming (FCP) and reward ablations, with negative results reported.
4. A playable game (`kilithattu-game`) in which a person can play any role against scripted or trained bots.

## 2. Related work

We build on PPO \[1\] with generalised advantage estimation \[2\] and the centralised-critic variant for cooperative games (MAPPO) \[3\]. League training with a pool of past opponents follows AlphaStar \[4\]; emergent strategy from competition is studied in \[5\]. Fictitious Co-Play (FCP) \[6\] trains an agent against a diverse pool of partners to obtain a better teammate for unseen partners. DAgger \[7\] corrects the distribution shift of behaviour cloning. Potential-based reward shaping \[8\] changes the learning signal without changing optimal policies. To our knowledge there is no published RL model of this game; we did not run a systematic literature search, so that statement is limited to what we know.

## 3. The game

| Item | Value (Classic rules, balance v1) |
| --- | --- |
| Field | 2 lanes × 5 rows of 3 m × 3 m boxes: 6 m wide, 15 m long; 2 m safe home and far zones |
| Attack | 4 crossers, free 2D movement in 8 directions, 5.0 m/s |
| Defence | 4 guards (one per inner line, slide along it, 3.0 m/s) and the Kili (moves along the chalk-line graph, turns only at junctions, 3.6 m/s) |
| Time | 20 ticks per second; clap at 2 s; innings of 120 s of live play |
| Tag | a defender within 0.5 m (swept test over the tick) puts a crosser out, except in the safe zones |

Rules implemented:

- **Trip progress.** A crosser starts as *pacha* (going out). Progress κ advances only when the *next expected* line is crossed in the expected direction: the six lines L0 to L5 outward, then, as *uppu*, the same six lines back. Re-crossing a counted line earns nothing, so points cannot be farmed.
- **Foul.** If an uppu and a pacha are in the same box at the end of a tick, the one that entered the box that tick is out (the uppu if neither or both entered).
- **Side exit.** Leaving the field sideways between L0 and L5 puts the crosser out.
- **Scoring.** One point per correct line (12 per round trip) plus 6 on delivering the salt home: 18 points per trip. The game is zero-sum.
- **Optional rules ("Plus").** Momentum and stamina, facing and turning, a strategic clap, limited vision and team calls are implemented as toggles; only momentum, facing and the strategic clap were trained.

**Balance.** With the speeds we first assumed (guard 4.2, Kili 4.8, tag radius 0.6) the scripted attack made no round trips and rarely lost all four crossers: the defence mirrored every crosser. A sweep against the scripted bots chose guard 3.0 m/s, Kili 3.6 m/s and tag radius 0.5 m ("balance v1"), giving about 20 points per innings and 0.2 trips for the scripted pair. This was tuned only for scripted bots.

## 4. Problem formulation

The game is a two-team zero-sum Markov game. All agents of a team share one reward, and execution is decentralised: each agent acts from its own observation.

**State.** Positions, velocities, status (pacha, uppu, out) and trip progress κ of each crosser; a waiting counter (ticks since the crosser last made progress); positions of guards and the Kili; clock, clap flag, score. The transition is a pure deterministic function of state and actions (randomness enters only at reset).

**Observation (per agent).** Entity tokens: 9 player tokens × 20 features (self, teammate and opponent flags, role, status, relative and absolute position, velocity, facing, stamina, trip progress, patience or turn timer, visibility), 9 line tokens × 5 features, and one global token of 13 values (time left, pre-clap flag, score, rule flags, last team call). The critic sees the same tokens without fog from the team's viewpoint. The waiting counter was added to the observation as a patience cue after we found that the scripted crosser's decisions depend on how long it has waited.

**Actions.** Five categorical heads per agent: move, dash, turn, clap, call. Under Classic rules only `move` is active; the other heads are masked to a no-op. Crossers choose among 9 moves (stay and 8 directions), guards among 3 (stay, left, right), the Kili among 5 (stay, N, E, S, W) with moves restricted by the line graph. Illegal actions are masked out of the policy distribution.

**Transition.** One 0.05 s tick: (1) mask illegal actions; (2) clap; (3) move all agents; (4) mark side exits; (5) swept tag test; (6) update κ for surviving crossers; (7) fouls; (8) reward and termination.

**Reward.** With n\_rows = 5 there are K = n\_rows + 1 = 6 lines per direction and a full trip is worth 2K + 6 = 18 points. Let N\_lines(t) be the number of correct line crossings in tick t summed over crossers and N\_del(t) the number of salt deliveries.

```
r_A(t) = ( N_lines(t) + 6 · N_del(t) ) / 18          (attack, shared by all crossers)
r_D(t) = − r_A(t)                                    (defence, shared by all defenders; zero-sum)
```

For training only, a potential-based shaping term is added with weight β = 0.1 and discount γ = 0.995:

```
r'_A(t) = r_A(t) + β · ( γ·Φ(s_{t+1}) − Φ(s_t) ),      r'_D(t) = − r'_A(t)
Φ(s) = (1/4) · Σ_i φ_i(s)          (0 if the episode has ended)
φ_i  = 0                                       if crosser i is out
φ_i  = y_i / (2L)                              if κ_i < K   (outbound, y_i clipped to [0, L], L = 15 m)
φ_i  = (2L − y_i) / (2L)                       if κ_i ≥ K   (returning)
```

Evaluation reports unshaped points per innings, which equal 18 × Σ\_t r\_A(t). **Termination:** all crossers out, or 2,400 live ticks (120 s).

## 5. Method

**Networks.** Tokens are embedded to d = 64 with a token-type embedding and passed through two pre-layer-norm transformer blocks (4 heads, MLP width 128). The actor concatenates the agent's own token, the mean of the player tokens and the global token, applies a 128-unit ReLU layer plus a role embedding, an optional GRU, and five masked categorical heads. The critic uses the same encoder without an ego token and two 128-unit layers to output a scalar; there is one centralised critic per team, used only in training. Parameters do not depend on field size or team size, which allows transfer to other fields. Most experiments use memoryless actors (no GRU), because imitation worked better that way.

**MAPPO.** Per update, 64 environments run 32 steps; advantages use GAE (γ = 0.995, λ = 0.95), normalised over valid agent samples; crossers that are out are masked from the loss. The loss is

```
L = L_pg + c_v · L_v − c_e · H
L_pg = − E[ min( ρ·Â , clip(ρ, 1−ε, 1+ε)·Â ) ],     ρ = exp( log π_θ(a|o) − log π_old(a|o) ),  ε = 0.2
L_v  = 0.5 · E[ max( (V − R)², (V_clip − R)² ) ],      V_clip = V_old + clip(V − V_old, −0.2, 0.2),  c_v = 0.5
H    = entropy of the masked, factorised policy; coefficient c_e decays (0.005 → 0.002 in fine-tuning, 0.01 → 0.004 in the league)
```

The policy is factorised over heads, log π(a|o) = Σ\_h log softmax(masked logits\_h)\[a\_h\], with illegal logits set to −10⁹. Optimiser: Adam, learning rate 1e-4 with linear decay to 10%, global gradient-norm clip 0.5, 2 epochs and 8 minibatches per update.

**Imitation (DAgger).** Two hand-written bots serve as experts and baselines: a crosser that stages near the next line, pairs with teammates on opposite sides, and dashes when the gap is large enough (patience decays with waiting), and guards that slide toward the nearest crosser. Round 0 is behaviour cloning; later rounds let the learner act (the expert takes over with probability β = 0.5^r) while the expert labels the visited states. The loss is the negative log-likelihood of the expert's action under the masked policy, with labels the mask forbids replaced by the no-op the environment would execute. 6 rounds, 800 steps per round, windows of 128 ticks.

**League.** A pool of frozen snapshots per side. A main agent trains against the latest opposing main agent with probability 0.35 and otherwise against an opponent sampled by prioritised fictitious self-play, with weight (1 − p\_win)² so that hard opponents are chosen more often. Each phase trains the attack and the defence for 25 updates each; a run has 20 phases and the pipeline runs two league iterations. Exploiter agents were implemented but not used.

**Pipeline per seed.** (1) DAgger imitation of the scripted attack; (2) PPO fine-tuning of the attack against the scripted guards for 1,200 updates; (3) DAgger imitation of the scripted guards; (4) league iteration 1; (5) league iteration 2; (6) cross-table. About 1 hour per seed on one RTX 3050 (4 GB), at roughly 4,000 environment steps per second with 64 × 32 batches.

## 6. Experimental protocol

- **References.** The scripted crosser and the scripted guards (reaction 0.4). "Imitation defence" is the clone of the guards; "fine-tuned attack" is stage 2; "league attack/defence" is the stage 5 output.
- **Evaluation.** 128 innings per cell with sampled actions; mean points per innings and mean round trips; the three seeds are separate training runs. Worst case of an attack is its minimum over the defences in a table, and of a defence its maximum over attacks. Zero-sum game values use linear programming or grid search on the table; Elo is fitted to pairwise scores.
- **Caution on noise.** Two runs with the same seed and recipe scored 62.2 and 39.6 points against the scripted guards, and three seeds of the same recipe spread from 40 to 121. The same pair evaluated with different evaluation seeds differs by up to about 10 percent.

## 7. Results

### 7.1 Main result (3 seeds, Classic rules, equal 25:25 league updates)

Mean attack points per innings, ± std over seeds (round trips in brackets):

| Attack \\ Defence | Scripted guards | Imitation defence | League defence |
| --- | --- | --- | --- |
| Scripted crosser | 20.3 | 22.4 ± 1.2 (0.2) | 20.5 ± 2.7 (0.2) |
| Fine-tuned attack | 67.2 ± 39.5 (2.4) | 80.1 ± 13.3 (3.3) | 3.8 ± 3.1 (0.0) |
| League attack | 31.3 ± 8.8 (0.5) | 49.7 ± 18.7 (1.6) | 33.5 ± 14.3 (0.7) |

Per seed against the scripted guards the fine-tuned attack scored 39.6, 49.5 and 112.5, always above the scripted crosser's 20.3. Worst cases: fine-tuned attack 3.8 ± 3.1, league attack 29.2 ± 11.6, scripted crosser 19.4 ± 1.6; for defences, scripted guards 67.2 ± 39.5, imitation defence 80.1 ± 13.3, league defence 33.5 ± 14.3. Solving the mean table as a zero-sum game, the scripted crosser and imitation defence are dominated; the attack mixes about 3.5 percent fine-tuned and 96.5 percent league attack, the defence about 45 percent scripted guards and 55 percent league defence, with value about 32.5 points. This is one 3 × 3 snapshot, not an equilibrium of the full game.

### 7.2 Held-out evaluation across seeds

Each league defence was also played against the other seeds' attacks and against an attack (`p1_sp_ctrl`, 62 points against the scripted guards) that appeared in no training pool. Cells are mean attack points; SG = scripted guards, D1–D3 = league defences of seeds 1–3.

| Attack | SG | D1 | D2 | D3 |
| --- | --- | --- | --- | --- |
| fine-tuned s1 | 43.9 | 6.2 | 4.8 | 4.6 |
| fine-tuned s2 | 49.5 | 7.1 | 4.5 | 4.6 |
| fine-tuned s3 | 121.4 | 5.2 | 4.4 | 0.6 |
| control (no pool) | 60.8 | 5.1 | 4.2 | 3.5 |
| league s1 | 28.6 | 37.5 | 31.2 | 47.7 |
| league s2 | 41.2 | 36.3 | 44.5 | 57.5 |
| league s3 | 26.0 | 16.7 | 9.0 | 20.4 |

League defences hold every fine-tuned-type attack to 0.6–7.1 points; held-out cells average 5.1 against 3.8 in-seed, and the never-pooled control gets 3.5–5.1. League attacks show no in-seed advantage (34.1 in-seed, 33.1 across seeds); seed quality matters more (league s2 is strongest, s3 weakest). Against league attackers the league defences are on par with or worse than the scripted guards (30.2, 28.2, 41.9 against 31.9), so their advantage is specific to rushing attackers. The value of this restricted game lies between 36.3 and 37.5 points.

### 7.3 Tournament

An agent is an (attack, defence) pair; 64 innings per direction.

| Agent | Elo | Nash weight |
| --- | --- | --- |
| league s2 | 1219.6 | 1.00 |
| league s1 | 1119.3 | 0.00 |
| league s3 | 1037.3 | 0.00 |
| fine-tuned s3 | 1004.6 | 0.00 |
| fine-tuned s2 | 902.0 | 0.00 |
| scripted | 873.7 | 0.00 |
| fine-tuned s1 | 843.4 | 0.00 |

All league pairs rank above the scripted pair, and the population's equilibrium is one agent (league s2, the best of three seeds). Elo gaps of a few tens are within the noise of this sample size.

### 7.4 Generalisation to other fields

Attacks trained on the 2 × 5 field with four crossers were played, without retraining, on a field with 3 lanes, one with 7 rows, with five crossers, and with faster guards, against the scripted guards of each field. Relative to the scripted crosser on the same field (mean ratio over the four non-classic fields): fine-tuned 1.93, 2.18, 2.86 and league 1.45, 1.96, 1.20 for seeds 1 to 3. Learned attacks beat the scripted crosser on five crossers and on faster guards in every case, and on the other two fields in 17 of 24 cells; the exceptions are fine-tuned s2 and league s3 on 3 lanes and fine-tuned s3 on 7 rows. Only a few field changes were tested, with 64 innings per cell.

### 7.5 Behaviour

With five innings per agent, a crude detector counted pincers (two crossers passing a line within 1 s at least 2.5 m apart) and decoys (a sideways dash near a line followed by a teammate crossing). Per point scored, fine-tuned attacks cross in simultaneous groups far more than the scripted crosser (0.40 to 0.74 against 0.26), consistent with a rush; league attacks are at or below the scripted rate (0.10 to 0.22) and show decoy-like dashes about as often as the scripted crosser (2.0 to 3.8 against 2.6). The scripted crosser switches sides by design, so similar counts do not show intent. Learned defences mark crossers more loosely than the scripted guards (1.3 to 1.7 m of lateral error against 0.29 m) and win against rushers without tight mirroring; how they win is not established.

### 7.6 Teaming (FCP)

One crosser played by an FCP learner or by a self-play control, plus three crossers played by a held-out partner, against the scripted guards. Each FCP learner trained 1,200 updates with partners from a pool; the control had the same start and budget. Across nine cells (three learners; replicates A, A2 started from the imitation policy, from the self-play policy, B), FCP was worse than the control in 7 and better in 2 (both with weak partners); the mean difference was −6.5 points. FCP learners were weak when played alone (4 to 8.5 points against 42 and 109 for the controls). Mixing styles hurts strong attackers even for the control: one control crosser with three `r3_ft` partners scored 44.7 against 109.2 for `r3_ft` alone. The partners are AI agents of one family, so nothing here speaks to human teammates.

### 7.7 Extended rules (Plus)

With momentum, facing and the strategic clap switched on together, scores were lower and round trips almost absent (0.0 to 0.1 per innings); in one seed the league attack fell to about 0 points for several phases. The league defence dominated. The rules were not switched on one at a time and not balance-tuned, and this run used an early-stopped fine-tune stage, so it shows that this configuration is defence-favoured, not that the rules lack depth.

### 7.8 Reward ablation

| Variant | Points per seed | Median | Mean round trips |
| --- | --- | --- | --- |
| Baseline (γ 0.995, shaping 0.1) | 43.9, 49.5, 121.4 | 49.5 | 2.64 |
| γ 0.999 | 35.5, 50.3, 57.9 | 50.3 | 1.29 |
| No shaping | 55.3, 52.6, 39.2 | 52.6 | 1.50 |

Neither change helped. The medians agree to within 3 points; the baseline's higher mean comes from one exceptional seed. With three seeds per variant, differences of 10 to 20 points cannot be detected.

### 7.9 A confound we found and corrected

Our first pipeline stopped PPO fine-tuning as soon as the evaluation reached 24 points. Every attack from that version was therefore stopped early. A control trained for 1,200 updates reached about 62 points, and the pipeline rerun without early stopping (Section 7.1) reversed one conclusion: with the early-stopped start, the league seemed not to improve the attack; with a proper start, it does. The earlier early-stopped results are kept in the repository for the record.
