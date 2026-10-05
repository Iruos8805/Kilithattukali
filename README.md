# Kilithattu-RL (കിളിത്തട്ടുകളി)

> Deep Multi-Agent Reinforcement Learning and Playable Interactive Game for **Kilithattu Kali (Uppukali)** — the traditional Onam tag game of Kerala.

[![CI](https://github.com/iruos8805/Kilithattukali/actions/workflows/ci.yml/badge.svg)](https://github.com/iruos8805/Kilithattukali/actions/workflows/ci.yml)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](LICENSE)
[![Python: 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](.python-version)

---

## 🌟 Overview

**Kilithattu Kali** is a strategic line-tag team game where an attacking team of **Crossers** attempts to cross a grid of lines, reach the far safe zone to become **Uppu** (salt), and return home to deliver salt while avoiding line-bound **Guards** and a roaming, line-jumping **Kili**.

This project provides:
1. **`kilithattu-core`**: A vectorised, deterministic simulation engine built with JAX, implementing both **Classic** rules and strategic **Kilithattu+** toggles (momentum, facing angles, fog-of-war, tactical clap, and procedural fields).
2. **`kilithattu-agents`**: Scripted bots, human-fair latency wrappers, and neural policies (entity-attention Transformer actors and centralised critics).
3. **`kilithattu-train`**: Fully jitted MAPPO (Multi-Agent PPO), curriculum stages, self-play league training, and Fictitious Co-Play (FCP) for human-compatible teammates.
4. **`kilithattu-eval`**: Tournaments, Elo and Nash averaging, exploitability proxy measurement, and game-balancing suites.
5. **`kilithattu-game`**: A pygame-ce interactive game supporting human vs AI, AI vs AI spectator mode, difficulty ladders, and deterministic replay playback.

---

## 🚀 Quick Start

### Installation

Ensure [uv](https://github.com/astral-sh/uv) is installed:
```bash
# Sync all workspace packages and dependencies
uv sync
```

### Play the Game

Launch the interactive game:
```bash
uv run kilithattu-game
# or with specific mode
uv run kilithattu-game --mode play_crosser --difficulty medium
```

### Run Tests

```bash
uv run pytest
```

### Run Benchmarks

```bash
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
│   ├── kilithattu-train/  # MAPPO, curriculum, self-play league
│   ├── kilithattu-eval/   # Tournaments, ratings, exploitability
│   └── kilithattu-game/   # Interactive game & visualizer
├── configs/              # YAML presets for rules, training, evaluation, game
└── docs/                 # Architecture decision records, benchmarks, reports
```

---

## 📜 Rules & Specifications

See [RULES.md](RULES.md) for the formal definition of R1–R14 (Classic) and P1–P7 (Kilithattu+ improvements).

---

## ⚖️ License

Licensed under the [GNU Affero General Public License v3.0](LICENSE).
