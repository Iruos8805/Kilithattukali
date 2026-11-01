# Results

> **Revision note.** Stage 2 of the pipeline stopped PPO fine-tuning at 24 points, so every "fine-tuned attack" in sections 1-5 is an early-stopped
> policy. A control trained for 1,200 updates without early stopping reaches 62 points against the scripted guards (section 7). Conclusions about
> the league's effect on the attack (section 1) and the attack-based numbers elsewhere should be re-checked with a strong starting attack.

All numbers: mean attack points per innings, 128 innings per cell unless stated, three training seeds (std over seeds), Classic rules
unless stated. "Scripted" = the hand-written bots. One hyperparameter set, one machine (RTX 3050). Trips are rare in every learned setting.

## 0. Main result: pipeline with a proper fine-tune stage (3 seeds, Classic rules, equal 25:25 league updates)
Stage 2 ran 1,200 PPO updates at learning rate 1e-4 with no early stop (prefix `r` in the repo). 128 innings per cell; mean +- std over seeds (trips in brackets).

| attack \ defence | scripted guards | imitation defence | league defence |
|---|---|---|---|
| scripted crosser | 20.3 | 22.4 +- 1.2 (0.2) | 20.5 +- 2.7 (0.2) |
| fine-tuned attack | 67.2 +- 39.5 (2.4) | 80.1 +- 13.3 (3.3) | 3.8 +- 3.1 (0.0) |
| league attack | 31.3 +- 8.8 (0.5) | 49.7 +- 18.7 (1.6) | 33.5 +- 14.3 (0.7) |
Worst case over the three defences (attacks): scripted 19.4 +- 1.6, fine-tuned 3.8 +- 3.1, league 29.2 +- 11.6. Worst case over the three attacks (defences):
scripted guards 67.2 +- 39.5, imitation 80.1 +- 13.3, league 33.5 +- 14.3.

Per seed (fine-tuned / league attack against the scripted guards): seed 1 39.6 / 28.6, seed 2 49.5 / 41.2, seed 3 112.5 / 24.2.
Per seed worst case, league defence vs scripted guards: 36.6 vs 39.6, 46.1 vs 49.5, 17.9 vs 112.5.

What this supports:
- The strong fine-tuned attack beats the scripted crosser against the scripted guards in every seed (39.6 to 112.5 against 20.3), but with enormous seed-to-seed spread.
- It is a specialist: the league defence holds it to 0.5-6.5 points in every seed. Caveat: that fine-tuned attack was in the league defence's training pool, so this cell is not held out.
- With a strong start the league makes the attack robust: worst case 29.2 +- 11.6 against 3.8 for its own starting point, and it beats the scripted crosser against the scripted guards in 3 of 3 seeds. It pays for this with peak strength against weak defences (31 against 67 versus the scripted guards).
- The league defence is the best pure defence in the worst-case sense in all three seeds, but by only about 3 points in seeds 1 and 2; the large gap comes from seed 3, where the fine-tuned attack scored 112.5 against the scripted guards.
- Solving the mean table as a zero-sum game: the scripted crosser and the imitation defence are dominated; the attack mixes about 3.5% fine-tuned and 96.5% league; the defence mixes about 45% scripted guards and 55% league defence; value about 32.5 points per innings. One 3x3 snapshot, not an equilibrium of the full game.
- This reverses the earlier conclusion (section 1) that the league does not improve the attack: that came from an early-stopped starting attack.
- Held-out evidence for the league defence: a separately trained attack (`p1_sp_ctrl`, 62 points against the scripted guards) was held to 8.8 and 7.3 points by two league defences (section 7).
Run-to-run noise is large: two runs with the same seed and recipe scored 62.2 (`p1_sp_ctrl`) and 39.6 (`r1_ft`) against the scripted guards.

## 0b. Held-out cross-seed table (7 attacks x 4 defences, 128 innings; strong-fine-tune pipeline plus the earlier control attack `p1_sp_ctrl`)
Columns: scripted guards (SG) and the three league defences D1, D2, D3 (seeds 1-3). Cells: mean attack points.
| attack | SG | D1 | D2 | D3 |
|---|---|---|---|---|
| fine-tuned s1 | 43.9 | 6.2 | 4.8 | 4.6 |
| fine-tuned s2 | 49.5 | 7.1 | 4.5 | 4.6 |
| fine-tuned s3 | 121.4 | 5.2 | 4.4 | 0.6 |
| control (never in any pool) | 60.8 | 5.1 | 4.2 | 3.5 |
| league s1 | 28.6 | 37.5 | 31.2 | 47.7 |
| league s2 | 41.2 | 36.3 | 44.5 | 57.5 |
| league s3 | 26.0 | 16.7 | 9.0 | 20.4 |

- League defences hold every fine-tuned-type attack to 0.6-7.1 points, including other seeds' attacks (held-out mean 5.1 against 3.8 in-seed) and an attack that was in no training pool (3.5-5.1). The counter to a rushing attack is not seed-specific.
- League attacks score 9.0-57.5 against league defences, with no in-seed advantage (in-seed mean 34.1, other seeds 33.1). Seed quality matters more than pairing: league s2 is the strongest attack (worst case 36.3), league s3 the weakest (9.0).
- League defences are far better than the scripted guards against fine-tuned-type attackers (mean conceded 3.3-5.9 against 68.9) but on par or worse against league attackers (30.2, 28.2, 41.9 against 31.9). Averaged over all seven attackers they concede 14.7-19.8 against 53.1 for the scripted guards, but that average depends on the attacker mix: four of the seven are fine-tuned-type.
- Worst case over the seven attackers: scripted guards 121.4; D1 37.5; D2 44.5; D3 57.5. Pure maximin attack: league s2 (36.3); pure minimax defence: D1 (37.5), so no pure saddle. The value of this restricted game lies between 36.3 and 37.5 (about 36.8 numerically; the defence mixes about 92% D1 and 8% scripted guards). This describes a chosen population, not the full game.
- Evaluation noise: the same pair measured with different evaluation seeds differs by up to about 10 percent (fine-tuned s3 against the scripted guards: 112.5 and 121.4).

> Sections 1-6 use the earlier pipeline whose fine-tune stage stopped early (24 points). They are kept for the record; section 0 supersedes them for attack/defence conclusions.

## 1. Classic rules, league update ratio (E4 support)
| attack \ defence | scripted guards | imitation defence | league defence (equal 25:25) | league defence (10:25, 8:30) |
|---|---|---|---|---|
| scripted crosser | 20.3 | 21.4 +- 1.6 | 24.2 +- 4.5 | 23.1 +- 3.0 |
| fine-tuned attack (imitation + PPO) | 26.2 +- 2.2 | 26.1 +- 2.6 | 3.3 +- 1.7 | 2.8 +- 0.6 |
| league attack (equal 25:25) | 18.8 +- 1.6 | 35.6 +- 1.4 | 21.2 +- 9.7 | |
| league attack (10:25, 8:30) | 17.9 +- 6.2 | 37.7 +- 17.0 | 8.8 +- 5.3 | |

Supported: the imitation + PPO attack beats the scripted crosser against the scripted guards; imitating the guards reproduces their
strength; the league defence holds learned attackers to a few points. Not supported: that the league improves the attack in general
(it scores below the pre-league attack against the scripted guards). The equal update ratio gave a more consistent, more competitive league attack.

## 2. Plus rules (E1: momentum, facing, strategic clap, all on together)
| attack \ defence | scripted guards | imitation defence | league defence |
|---|---|---|---|
| scripted crosser | 12.8 | 17.4 +- 1.9 | 19.6 +- 3.9 |
| fine-tuned attack | 20.5 +- 7.8 | 21.1 +- 8.3 | 1.0 +- 0.6 |
| league attack | 11.5 +- 3.0 | 17.1 +- 4.1 | 3.9 +- 4.1 |

Scores are lower than under Classic and round trips are essentially absent (0.0-0.1 per innings). In seed 3 the league attack collapsed to
about 0 points for several phases. The league defence dominates. Caveats: the three rules were switched on together (no per-rule ablation),
the combination was never balance-tuned, and no depth metric (exploitability, outcome entropy, non-transitivity) was computed.
So this does not show the rules add depth; it shows this configuration is strongly defence-favoured.

## 3. Tournament (Elo and Nash, 7 agents, 32 innings per direction)
Elo: s3_league 1063, s2_league 1060, s1_league 1040, scripted 1003, s3_ft 964, s2_ft 937, s1_ft 933.
Nash support: scripted 0.81, s3_league 0.16, s2_ft 0.03. League pairs rank highest (their defence is strong), but the Nash mixture puts most
weight on the scripted pair, so the ratings are non-transitive and Elo overstates robustness. Differences of about 50 Elo are within the noise of this sample size.

## 4. Behaviour detectors (E7; 3 innings per agent, so counts are noisy)
| agent | pincers | decoys | guard error (m) | tags |
|---|---|---|---|---|
| scripted | 6.3 | 4.0 | 0.40 | 2.3 |
| fine-tuned (s1, s2, s3) | 7.3, 9.0, 15.3 | 0, 0, 0 | 1.7, 2.1, 2.1 | 2.3, 2.7, 2.0 |
| league (s1, s2, s3) | 4.7, 3.0, 1.3 | 1.3, 3.0, 0.7 | 2.3, 2.5, 2.7 | 2.3, 3.3, 3.7 |

The pincer detector cannot tell a coordinated pincer from a plain simultaneous rush, so the high counts for the fine-tuned attacks mostly reflect a
rush. No decoy behaviour appears in the fine-tuned attacks. The learned defences mark much more loosely than the scripted guards (guard error 1.7-2.7 m
against 0.40 m) yet score as well or better, so their strength does not come from tighter mirroring. How they win is not established.

## 5. Generalisation to other fields (E3; 32 innings, normalised points vs the scripted guards of each field)
| attack | classic | 3 lanes | 7 rows | 5 crossers | fast guards | retention |
|---|---|---|---|---|---|---|
| scripted | 0.310 | 0.538 | 0.214 | 0.245 | 0.178 | 0.95 |
| fine-tuned s1 / s2 / s3 | .418 / .350 / .334 | .288 / .134 / .248 | .223 / .130 / .190 | .423 / .319 / .284 | .375 / .344 / .255 | 0.78 / 0.66 / 0.73 |
| league s1 / s2 / s3 | .247 / .247 / .256 | .238 / .464 / .307 | .115 / .103 / .144 | .220 / .261 / .290 | .306 / .237 / .241 | 0.89 / 1.08 / 0.96 |

The fine-tuned attacks keep their advantage over the scripted crosser with five crossers and with faster guards, but fall behind it on the 3-lane field
and are no better on the 7-row field. League attacks retain more of their (lower) classic score. With 32 innings and a field change, single cells
swing a lot (s2 league: 0.46 on 3 lanes, 0.25 on classic), so treat retention values as indicative.

## 6. FCP (E6; 128 innings vs the scripted guards; team = one learner + three held-out partners)
First run (300 updates for the FCP learner, a longer run for the baseline): FCP 19.6 / 16.7 vs baseline 22.4 / 24.3. That comparison was confounded by
training budget and is superseded by the matched run below.

Matched budgets: both the FCP learner and the self-play control (`p1_sp_ctrl`) trained 1,200 PPO updates from the same imitation start at learning rate 1e-4.
| held-out partner | FCP learner + partners | self-play control + partners | partner alone (x4) |
|---|---|---|---|
| p3 fine-tuned | 22.6 +- 0.7 | 20.9 +- 0.7 | 23.0 +- 0.7 |
| p3 league | 25.2 +- 1.1 | 23.3 +- 1.1 | 19.6 +- 1.0 |
Played alone (x4): FCP learner 6.9 +- 0.3, self-play control 69.1 +- 3.6.

The FCP learner is slightly ahead as a teammate (+1.7 and +1.9 points; each under two standard errors of the difference, one learner seed, one control seed).
The control is a much stronger solo attacker but drops to partner level with unfamiliar teammates, while the FCP learner cannot play alone yet lifts mixed teams.
Heuristic check (crossers' points add): expected mixed-team score if slots acted independently is 0.25 x solo(X) + 0.75 x solo(Y); observed minus expected is
-13.6 and -8.7 for the control and +3.6 and +8.8 for the FCP learner. Caveats: the held-out partners are AI policies from the same recipe (similar conventions),
not human-like; crossers interact through defenders and the foul rule; no human teammates were tested.

## 7. A stronger control attack against the defences (128 innings)
`p1_sp_ctrl` (imitation start, 1,200 PPO updates, no early stop; about 62-72 points against the scripted guards in training evaluations):
| defence | points | trips |
|---|---|---|
| scripted guards | 62.2 | 2.11 |
| imitation defence (seed 1) | 99.7 | 4.23 |
| league defence (seed 1, equal ratio) | 8.8 | 0.00 |
| league defence (seed 2, equal ratio) | 7.3 | 0.00 |
The league defences were never trained against this attack, and they limit it to 7-9 points where the scripted guards concede 62. This is a held-out-attacker
result for two defence seeds and one attacker. It also shows the early-stopped attacks of sections 1-5 understated what the fine-tuning recipe can reach.

## What remains untested
Team calls (E5); the RL-driven balance search with learned agents (E2); plain self-play at equal compute (E4); the league and the cross-tables rerun from a strong (not early-stopped) attack; FCP with more seeds and with human-like partners; any human check of the rules or human study.
