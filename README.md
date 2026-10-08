# Kilithattu-RL (കിളിത്തട്ടുകളി)

> Deep Multi-Agent Reinforcement Learning and Playable Interactive Game for **Kilithattu Kali (Uppukali)** — the traditional Onam tag game of Kerala.

[![CI](https://github.com/iruos8805/Kilithattukali/actions/workflows/ci.yml/badge.svg)](https://github.com/iruos8805/Kilithattukali/actions/workflows/ci.yml)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](LICENSE)
[![Python: 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](.python-version)

---

## 🌟 Overview

**Kilithattu Kali** is a strategic line-tag team game where an attacking team of **Crossers** attempts to cross a grid of lines, reach the far safe zone to become **Uppu** (salt), and return home to deliver salt while avoiding line-bound **Guards** and a roaming, line-jumping **Kili**.

This project provides:
1. **`kilithattu-core`**: A vectorised, deterministic simulation engine built with JAX, implementing both **Classic** rules and strategic **Kilithattu+** toggles (momentum, facing angles, fog-of-war, tactical clap, and procedural fields). Over **50,000 steps/sec** on CPU and millions on GPU.
2. **`kilithattu-agents`**: Scripted bots, human-fair latency wrappers, and neural policies (entity-attention Transformer actors and centralised critics).
3. **`kilithattu-train`**: Fully jitted MAPPO (Multi-Agent PPO), curriculum stages (S0–S6), AlphaStar-style self-play league training with Prioritised Fictitious Self-Play (PFSP), and Fictitious Co-Play (FCP) for human-compatible teammates.
4. **`kilithattu-eval`**: Tournaments, Elo and Nash averaging, exploitability proxy measurement, and game-balancing suites.
5. **`kilithattu-game`**: A pygame-ce interactive game supporting human vs AI, AI vs AI spectator mode, difficulty ladders, and deterministic `.kreplay` playback.

---

## 🎮 Controls

| Action | Keyboard (Crosser) | Keyboard (Kili / Guard) | Gamepad |
|---|---|---|---|
| **Move** | `W` `A` `S` `D` (diagonals supported) | `W` `A` `S` `D` (Guards: `A` / `D` only) | Left Stick (8 directions) |
| **Sprint / Dash (P1)** | `Left Shift` | `Left Shift` | `RT` |
| **Turn (P2)** | — | `Q` | `LB` |
| **Clap (P4)** | — | `Space` | `A` button |
| **Switch Player** | `Tab` | `Tab` | `RB` |
| **Pause** | `Esc` | `Esc` | `Start` |

---

## 🚀 Quick Start

### Installation

Ensure [uv](https://github.com/astral-sh/uv) is installed:
```bash
# Sync all workspace packages and dependencies
uv sync
```

### Play the Game

Launch the interactive game client:
```bash
# Play as Crosser against AI defenders
uv run kilithattu-game --mode play_crosser --difficulty medium

# Play as Kili defender
uv run kilithattu-game --mode play_kili --difficulty easy

# Watch AI vs AI match
uv run kilithattu-game --mode watch
```

### Training & Curriculum

Run MAPPO curriculum stages:
```bash
# Train on curriculum stage S0 (Run Home)
uv run kilithattu-train stage --stage S0

# Train on stage S1 (One Line)
uv run kilithattu-train stage --stage S1
```

### Evaluation & Tournaments

Run round-robin tournament across policies:
```bash
uv run kilithattu-eval tournament --innings 20
```

### Running Research Experiments (E1–E7)

Run any of the empirical experiment suites:
```bash
# E1: Rule depth & ablation study (Classic vs Plus toggles)
uv run python experiments/e1_rule_ablation/run_e1.py --innings 20

# E2: RL & empirical balancing search
uv run python experiments/e2_rl_balancing/run_e2.py --trials 6

# E3: Generalisation across field sizes
uv run python experiments/e3_generalisation/run_e3.py --innings 10

# E4: Self-play league vs naive self-play exploitability
uv run python experiments/e4_league_vs_selfplay/run_e4.py --iterations 20

# E5: Team communication calls analysis
uv run python experiments/e5_team_calls/run_e5.py --innings 10

# E6: Fictitious Co-Play with held-out teammates
uv run python experiments/e6_human_teaming/run_e6.py --innings 10

# E7: Emergent tactical maneuver detectors
uv run python experiments/e7_emergent_tactics/run_e7.py --innings 10
```

---

## 🧪 Testing & Verification

```bash
# Run complete test suite (29 tests across all 5 packages)
uv run pytest

# Check architectural import contracts
uv run lint-imports

# Check code formatting & linting
uv run ruff check .
uv run ruff format --check .

# Benchmark simulation throughput
uv run python tools/bench_env.py
```

---

## 📐 Project Structure

```
Kilithattukali/
├── RULES.md              # Canonical game rules and parameter specifications
├── PLAN.md               # Detailed technical design and project roadmap
├── PROPOSAL.md           # Research background and literature review
├── packages/
│   ├── kilithattu-core/   # JAX rules engine & replay system
│   ├── kilithattu-agents/ # Policies, scripted bots, neural models
│   ├── kilithattu-train/  # MAPPO, curriculum, self-play league, FCP
│   ├── kilithattu-eval/   # Tournaments, ratings, exploitability, tactics
│   └── kilithattu-game/   # Interactive pygame client & HUD
├── configs/              # YAML presets for rules, training, evaluation, ladder
├── experiments/          # Empirical experiments E1–E7 and result records
├── tools/                # Benchmarking, balance sweep, and ladder export tools
└── docs/                 # Architecture decision records, benchmarks, reports
```

---

## 📜 Rules & Specifications

See [RULES.md](RULES.md) for the formal definition of R1–R14 (Classic) and P1–P7 (Kilithattu+ improvements).

---

## ⚖️ License

Licensed under the [GNU Affero General Public License v3.0](LICENSE).
