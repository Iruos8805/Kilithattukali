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

## 0c. Tournament with the strong agents (Elo and Nash; 7 agents, 64 innings per direction)
An agent is an (attack, defence) pair: scripted; for each seed, "ft" (fine-tuned attack with the imitation defence) and "league" (league attack with league defence).
| agent | Elo | Nash weight | score vs the Nash mix |
|---|---|---|---|
| league s2 | 1219.6 | 1.00 | 0.000 |
| league s1 | 1119.3 | 0.00 | -0.037 |
| league s3 | 1037.3 | 0.00 | -0.593 |
| ft s3 | 1004.6 | 0.00 | -0.943 |
| ft s2 | 902.0 | 0.00 | -0.737 |
| scripted | 873.7 | 0.00 | -0.284 |
| ft s1 | 843.4 | 0.00 | -1.087 |
All three league pairs rank above the scripted pair, and the scripted pair now has zero Nash weight (with the earlier early-stopped agents it held 0.81). The equilibrium
of this population is a single agent, league s2, which also had the strongest attack and a robust defence in the cross-seed table; league s1 is almost tied with it.
The fine-tuned pairs rank lowest or middle because their defence is the imitation defence. Caveats: Elo gaps of a few tens are within the noise of 64 innings; scores are
differences in normalised points; three seeds; the population is small and chosen.

## 0d. Generalisation to other fields with the strong agents (E3; 64 innings; normalised points vs the scripted guards of each field)
| attack | classic | 3 lanes | 7 rows | 5 crossers | fast guards |
|---|---|---|---|---|---|
| scripted | 0.277 | 0.492 | 0.221 | 0.251 | 0.195 |
| fine-tuned s1 / s2 / s3 | .553 / .826 / 1.492 | .692 / .422 / .512 | .331 / .279 / .141 | .575 / .818 / 1.157 | .489 / .655 / 1.005 |
| league s1 / s2 / s3 | .441 / .594 / .325 | .520 / .666 / .439 | .275 / .298 / .265 | .433 / .593 / .337 | .349 / .541 / .266 |
Ratio to the scripted crosser on the non-classic fields (mean): fine-tuned 1.93 / 2.18 / 2.86, league 1.45 / 1.96 / 1.20. Learned attacks beat the scripted crosser on five crossers
and fast guards in every case, and on the other two fields in 7 of 12 and 10 of 12 cells; the exceptions are fine-tuned s2 and league s3 on 3 lanes, and fine-tuned s3 on 7 rows.
The "retention" ratio (non-classic over classic) is misleading for attacks with a very high classic score (fine-tuned s3: 0.47, yet still above the scripted crosser on three of four other fields).
Transfer is therefore much better than the earlier early-stopped agents showed. No uncertainty is available beyond the 64 innings per cell, and the field changes are only a few of many possible ones.

## 0e. Behaviour detectors with the strong agents (E7; 5 innings per agent, so noisy)
| agent | pincers | decoys | guard error (m) | tags | pincers per point (rough) |
|---|---|---|---|---|---|
| scripted | 5.2 | 2.6 | 0.29 | 1.6 | 0.26 |
| fine-tuned s1 / s2 / s3 | 24.8 / 20.0 / 90.0 | 0 / 0 / 2.6 | 2.03 / 1.90 / 2.01 | 3.2 / 3.4 / 3.4 | 0.56 / 0.40 / 0.74 |
| league s1 / s2 / s3 | 6.2 / 6.4 / 2.6 | 3.6 / 3.8 / 2.0 | 1.65 / 1.27 / 1.36 | 4.0 / 3.2 / 4.0 | 0.22 / 0.16 / 0.10 |
The raw pincer counts scale with how many lines an agent crosses (fine-tuned s3 scores 100+ points), so I divided by the points from the cross-table as a rough normalisation (different
evaluation runs, so indicative only). Fine-tuned attacks cross lines in simultaneous groups far more often than the scripted crosser (0.4-0.74 per point against 0.26), consistent with a rush;
league attacks are at or below the scripted rate (0.10-0.22) and show decoy-like sideways dashes about as often as the scripted crosser (2.0-3.8 against 2.6), whereas fine-tuned s1 and s2 show none.
The scripted crosser is built to switch sides, so a similar decoy count is not evidence of intent. The league defences mark more loosely than the scripted guards (1.27-1.65 m against 0.29 m)
but tighter than the imitation defence (1.9-2.0 m); their strength against rushers does not come from tight mirroring. Learned matches have more tags per innings (3.2-4.0 against 1.6).
The detectors are crude and the sample is five innings, so these are descriptions, not demonstrations of tactics.

## 0f. FCP with strong partners and a matched control (E6; 128 innings vs the scripted guards; two replicates)
Each FCP learner trained 1,200 updates (lr 1e-4) from its seed's imitation start, with partners drawn from a pool; the control is the same start and budget trained in self-play (the pipeline's fine-tuned attack).
Replicate A: learner from `r1_dagger_attack`, control `r1_ft`, pool {r2_ft, r2_lg2, r1_lg2, p1_sp_ctrl}, held out {r3_ft, r3_lg2, p3_ft}. Replicate B: learner from `r3_dagger_attack`, control `r3_ft`, pool {r1_ft, r1_lg2, r3_lg2, p1_sp_ctrl}, held out {r2_ft, r2_lg2, p3_ft}.
Team = one learner (or control) crosser plus three held-out partner crossers.
| replicate / held-out partner | partner alone (x4) | FCP team | control team | difference (z) | FCP synergy | control synergy |
|---|---|---|---|---|---|---|
| A / r3_ft | 109.2 | 25.1 +- 1.5 | 44.7 +- 2.0 | -19.6 (-7.8) | -58.9 | -47.8 |
| A / r3_lg2 | 22.6 | 29.7 +- 0.7 | 25.3 +- 0.9 | +4.5 (+3.8) | +10.6 | -2.2 |
| A / p3_ft | 23.0 | 21.0 +- 0.6 | 19.8 +- 0.7 | +1.1 (+1.2) | +1.6 | -7.9 |
| B / r2_ft | 49.6 | 33.2 +- 1.6 | 45.3 +- 2.8 | -12.1 (-3.7) | -5.1 | -19.2 |
| B / r2_lg2 | 43.8 | 19.4 +- 1.2 | 38.4 +- 1.7 | -19.0 (-9.0) | -14.5 | -21.7 |
| B / p3_ft | 23.0 | 18.4 +- 0.6 | 24.1 +- 0.8 | -5.7 (-5.9) | +0.1 | -20.4 |
Played alone (x4): FCP learners 8.5 +- 0.4 (A) and 4.4 +- 0.1 (B); controls 42.2 +- 1.3 (A) and 109.2 +- 6.2 (B). Synergy = observed team score minus 0.25 x solo(X) + 0.75 x solo(partner).

- FCP did not produce a better teammate. The FCP team scored below the control team in 4 of 6 cells (by 5.7 to 19.6 points) and above it in 2 (+1.1, +4.5), both with weak partners (about 23 points alone). Mean difference over the six cells: -8.5 (cells are not independent: two learners, two controls).
- The FCP learner is a weak crosser on its own (4.4-8.5 points), and mixed-team scores cluster at 18-33 whatever the partner's strength (partners alone score 22.6 to 109.2): the learner caps the team near its own level.
- Mixing styles hurts strong attackers a lot: the control plus three r3_ft partners scores 44.7 against 109.2 for r3_ft alone and 92.5 expected under independence (synergy -47.8). The control's synergy is negative in all six cells; FCP's is positive in three. The strong rushing attacks rely on all four crossers sharing conventions, which is the convention problem FCP is meant to address, though this implementation did not solve it.
- The earlier small FCP advantage (+1.7 and +1.9 with weaker partners, within noise) did not survive stronger partners.
- Untested possible remedies: start the FCP learner from the self-play control instead of the imitation policy; train longer; add per-agent credit assignment (only one crosser's data is used and the reward is shared by four); include the learner's own checkpoints in the pool.
- Caveats: two replicates; the held-out partners are AI agents of one family, so this says nothing about human teammates; each pool contains a descendant of its control; one evaluation seed per cell.

## 0g. FCP, third learner: started from the self-play policy (replicate A2)
Same pool, held-out partners and budget as replicate A, but the learner starts from `r1_ft` (the control) instead of the imitation policy.
| held-out partner | partner alone | FCP team | control team | difference (z) | FCP synergy | control synergy |
|---|---|---|---|---|---|---|
| r3_ft | 109.2 | 39.4 +- 1.7 | 44.7 +- 2.0 | -5.3 (-2.0) | -44.6 | -47.8 |
| r3_lg2 | 22.6 | 23.6 +- 0.8 | 25.3 +- 0.9 | -1.7 (-1.4) | +4.6 | -2.2 |
| p3_ft | 23.0 | 19.0 +- 0.6 | 19.8 +- 0.7 | -0.8 (-0.9) | -0.3 | -7.9 |
Starting from a strong policy narrows the gap (mean -2.6 over these three cells, against -4.7 for replicate A and -12.3 for replicate B in section 0f), but the FCP learner still collapsed as a solo
attacker (8.1 +- 0.2 against the control's 42.2 +- 1.3) and its teams are no better than the control's. Over all nine cells (three FCP learners) the difference is negative in 7 and positive in 2 (the two weak-partner cells of
replicate A), mean -6.5 points. Conclusion: no evidence that FCP training, as implemented here, yields a better teammate; the effect is null with weak partners and negative with strong ones.

## 0h. Reward ablation (fine-tune stage, 3 seeds each, 1,200 updates, lr 1e-4, 128 innings vs the scripted guards)
| variant | points per seed | mean | median | std | mean trips |
|---|---|---|---|---|---|
| baseline (gamma 0.995, shaping 0.1) | 43.9, 49.5, 121.4 | 71.6 | 49.5 | 43.2 | 2.64 |
| gamma 0.999 | 35.5, 50.3, 57.9 | 47.9 | 50.3 | 11.4 | 1.29 |
| no shaping (shaping 0) | 55.3, 52.6, 39.2 | 49.0 | 52.6 | 8.6 | 1.50 |
Neither change improved on the baseline: the medians are within 3 points of each other, and the baseline's higher mean comes from one exceptional seed (121.4; without it the baseline mean is 46.7). Shaping was not needed to reach about 50 points from the
imitation start. Three seeds per variant cannot detect differences of 10-20 points (identical-recipe runs have scored 40 to 62 and 40 to 121). So the current reward was kept; this is a null ablation at low power, and it does not test training from scratch,
different scoring values, or per-agent credit assignment.

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

## 6. FCP with weak partners (E6; superseded by section 0f; 128 innings vs the scripted guards; team = one learner + three held-out partners)
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
Team calls (E5); the RL-driven balance search with learned agents (E2); plain self-play at equal compute (E4); the league and the cross-tables rerun from a strong (not early-stopped) attack; FCP variants (see section 0f) and FCP with human teammates; any human check of the rules or human study.
