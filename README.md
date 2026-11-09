# Kilithattu Kali: Multi-Agent Reinforcement Learning & League Training

An executable digital environment, multi-agent reinforcement learning (MARL) framework, and playable client for **Kilithattu Kali**, a traditional line-tag game played in Kerala during Onam.

This project formalizes Kilithattu Kali as a zero-sum, two-team Markov game with heterogeneous, asymmetric agents. Attack and defense policies are trained using **Multi-Agent Proximal Policy Optimization (MAPPO)** under an **Imitation-Initialized League Training** pipeline (DAgger pre-training followed by prioritized fictitious self-play).

---

## Table of Contents
- [Project Overview](#project-overview)
- [System Architecture & Training Pipeline](#system-architecture--training-pipeline)
- [Repository Structure](#repository-structure)
- [Prerequisites & Dependencies](#prerequisites--dependencies)
- [Pre-Flight Verification](#pre-flight-verification)
- [Quick Start: Interactive Menu](#quick-start-interactive-menu)
- [Demo Matches (Scientific Replications)](#demo-matches-scientific-replications)
- [Interactive Play (Human vs AI)](#interactive-play-human-vs-ai)
- [Difficulty Ladder Setup](#difficulty-ladder-setup)
- [Replay Inspection](#replay-inspection)
- [Troubleshooting & Performance Checklist](#troubleshooting--performance-checklist)

---

## Project Overview

Kilithattu Kali is played on a grid of $2 \times 5$ boxes ($6\text{ m} \times 15\text{ m}$) separated by chalk lines:
- **Attack Team ($\mathcal{A}$):** 4 Crossers moving freely in 8 directions (speed: 5.0 m/s). Their goal is to cross lines $L_0 \dots L_5$ sequentially (*pacha* / outbound), turn back (*uppu* / inbound), and deliver the "salt" home without being tagged or fouling. Full trip = 18 points.
- **Defense Team ($\mathcal{D}$):** 4 Guards constrained to lateral sliding along lines $L_1 \dots L_4$ (speed: 3.0 m/s) and 1 **Kili** roaming the chalk line perimeter and central grid line (speed: 3.6 m/s, turning at junctions). Defenders tag crossers with a swept continuous collision radius ($\rho = 0.5\text{ m}$).

### Why Imitation-Initialized League Training?
1. **Cold-Start Problem:** From random noise, agents struggle to explore the multi-stage crossing sequence. We warm-start policies via **DAgger** (Dataset Aggregation) imitating hand-coded heuristic bots.
2. **Fine-Tuning (MAPPO):** Pre-trained crossers are fine-tuned against scripted defense, learning coordinated rushing tactics.
3. **League Training (PFSP):** To prevent cyclic counter-strategies and brittle specialization, agents train against a historical pool of snapshots sampled via Prioritized Fictitious Self-Play:
   $$w_j \propto \left(1 - p_{\mathrm{win}}(j)\right)^2$$
   This elevates worst-case attack performance from **$3.8$ points to $29.2$ points**, while defense evolves to dismantle coordinated group rushes.

---

## System Architecture & Training Pipeline

```
[ Heuristic Expert Bots ]
          │
          ▼  (Stage 1 & 3: DAgger Imitation)
[ Warm-Started Actors (Attack & Defense) ]
          │
          ▼  (Stage 2: MAPPO Single-Role Fine-Tuning)
[ High-Scoring Rushing Attackers ]
          │
          ▼  (Stage 4: Two-Team Alternating League)
[ Prioritized League Pool: Historical Snapshots + Main Learners ]
```

- **Environment Engine:** Vectorized in **JAX** (64 parallel environments per batch, pure functional step transitions, swept continuous-time tag detection).
- **Policy Networks:** Entity-attention transformer encoders invariant to field dimensions (9 player tokens, 9 line tokens, 1 global token embedded into $d=64$, 2 transformer blocks, 4 heads) with masked categorical action heads.
- **Critic:** Centralized state-value network $V_\phi(s)$ utilized strictly during training (CTDE).

---

## Repository Structure

```text
Kilithattukali/
├── configs/               # Hyperparameter and game configurations
│   └── game/ladder.yaml   # Difficulty tiers definition
├── packages/
│   ├── kilithattu-core/   # Vectorized JAX rules engine & reference Python engine
│   ├── kilithattu-agents/ # Entity-attention models, Flax checkpoints, heuristic bots
│   ├── kilithattu-train/  # MAPPO loop, DAgger, League & PFSP trainer
│   ├── kilithattu-eval/   # Elo/Nash evaluation, tournaments, behavior detectors
│   └── kilithattu-game/   # Pygame graphical client & replay viewer
├── replays/               # Auto-saved match replays (.kreplay)
├── runs/                  # Policy checkpoints (imitation, fine-tuned, league)
│   ├── r1_dagger_defence/
│   ├── r1_ft/
│   └── r1_lg2/
└── tools/                 # Evaluation scripts, ladder generators, analysis utilities
```

---

## Prerequisites & Dependencies

- **Operating System:** Linux (Ubuntu 20.04/22.04 recommended)
- **Python Tooling:** Managed via [`uv`](https://github.com/astral-sh/uv) (fast Python package manager)
- **Hardware:** CUDA-capable GPU (tested on NVIDIA RTX 3050 4GB). JAX with CUDA support enabled.
- **Display:** Active X11/Wayland display server (do **not** use headless `SDL_VIDEODRIVER=dummy` when running the GUI).

---

## Pre-Flight Verification

Ensure your shell is inside the repository root, your CUDA environment is active, and ROS environment paths do not pollute dependencies:

```bash
cd ~/Desktop/Kilithattukali

# Verify JAX detects the GPU (Output MUST list CudaDevice, not CpuDevice)
PYTHONPATH= uv run python -c "import jax; print(jax.devices())"
```

> **Note on `PYTHONPATH=`:** Always prepend commands with `PYTHONPATH=` to avoid conflicts if your workstation environment sources ROS/ROS2 libraries.

---

## Quick Start: Interactive Menu

Launch the interactive graphical interface:

```bash
PYTHONPATH= uv run kilithattu
```

### Menu Controls & Configuration:
1. **You Play:** 
   - `Watch only` (observe AI vs AI)
   - `Crosser`, `Guard`, or `Kili` (human-controlled slot).
2. **Team Selection:** Use **Left/Right Arrows** to switch checkpoints:
   - `Trained: r1_ft/latest` (Fine-tuned aggressive attack)
   - `Trained: r1_lg2/latest` (League-trained robust policies)
   - `Trained: r1_dagger_defence/imitation` (Baseline imitation defense)
3. **Rules:** Keep set to **Classic** (the formal model used during training).
4. **Launch Match:** Navigate down to **Start** and press **Enter**.

> **JIT Warmup Notice:** On the first match between a new pairing of policies, the client will display a **"Loading…"** screen for up to 60 seconds while JAX compiles the computational graphs. Subsequent matches will start immediately.

### In-Game Keybinds:
- **`Esc` or `M`**: Return to the main menu.
- **`R`**: Restart a new innings immediately.
- Replays are automatically saved to `replays/<timestamp>.kreplay` upon innings completion.

---

## Demo Matches (Scientific Replications)

Run these direct commands to visually inspect the central empirical findings documented in the paper:

### 1. Competitive League Showdown (Balanced Match)
*League Attack vs. League Defense (~37 points, ~0.8 round trips):*
```bash
PYTHONPATH= uv run kilithattu --side watch \
  --attack path:runs/r1_lg2/attack/latest \
  --defence path:runs/r1_lg2/defence/latest
```

### 2. Attack Domination (The Coordinated Rush)
*Fine-Tuned Attack vs. Imitation Defense (~69–80 points, ~2.6–3.3 round trips):*
```bash
PYTHONPATH= uv run kilithattu --side watch \
  --attack path:runs/r1_ft/attack/latest \
  --defence path:runs/r1_dagger_defence
```

### 3. Total Defensive Lockdown (The League Counter)
*Fine-Tuned Attack vs. League Defense (Crushed to ~3.8–6.5 points, 0.0 round trips):*
```bash
PYTHONPATH= uv run kilithattu --side watch \
  --attack path:runs/r1_ft/attack/latest \
  --defence path:runs/r1_lg2/defence/latest
```

> **Key Visual Finding:** Watching **Demo 2** followed by **Demo 3** demonstrates policy exploitability: the same specialized attacker that easily overwhelms imitation defenders is systematically shut down by the league defense.

---

## Interactive Play (Human vs AI)

Take control of a player on the field against trained models:

### Play as an Attacking Crosser
Test your evasion skills against the robust League Defense:
```bash
PYTHONPATH= uv run kilithattu --side attack \
  --attack path:runs/r1_ft/attack/latest \
  --defence path:runs/r1_lg2/defence/latest
```

### Play as the Defending Kili
Coordinate with AI Guards to intercept the fine-tuned rushing attackers:
```bash
PYTHONPATH= uv run kilithattu --side kili \
  --attack path:runs/r1_ft/attack/latest \
  --defence path:runs/r1_lg2/defence/latest
```

### Controls:
- **Movement:** `WASD` or `Arrow Keys`.
- **Switch Active Crosser:** `Tab`.
- **Target Indicator:** A **gold ring** highlights your active entity.
- **Foul Warnings:** Red-tinted floor boxes warn of imminent out-of-bounds or pacha/uppu collision foul hazards.
- **Kili Navigation:** Hold a directional key before reaching an intersection to queue a turn at the next junction.

---

## Difficulty Ladder Setup

Generate and update the predefined game ladder (`configs/game/ladder.yaml`) using trained checkpoints:

```bash
# 1. Back up your existing ladder config (optional)
cp configs/game/ladder.yaml configs/game/ladder.yaml.bak

# 2. Compile the ladder using Seed 1 checkpoints
PYTHONPATH= uv run python tools/make_ladder.py --seed 1 --prefix r --tag ""
```

This maps:
- **Medium:** Fine-tuned attack vs. Imitation defense.
- **Hard:** Full League attack vs. League defense pair.

Launch directly using the ladder presets:
```bash
PYTHONPATH= uv run kilithattu --side watch --difficulty hard
```

---

## Replay Inspection

List and replay past recorded matches:

```bash
# List completed matches
ls replays/

# Play back a match
PYTHONPATH= uv run kilithattu --replay replays/<filename>.kreplay
```

---

## Troubleshooting & Performance Checklist

| Issue | Cause | Resolution |
| :--- | :--- | :--- |
| **Blank window / immediate crash** | Missing graphical backend or headless flag | Ensure `SDL_VIDEODRIVER=dummy` is **unset** in your terminal. |
| **ROS package conflicts / import errors** | System paths polluting virtual environment | Always execute commands prefixed with `PYTHONPATH=`. |
| **Window stuttering during gameplay** | Small policy network overhead on CUDA runtime | Force actor inference onto CPU: prepend command with `JAX_PLATFORMS=cpu`. |
| **Check point path error (`FileNotFound`)** | Mismatched run naming or missing training directory | Verify checkpoint folders in `runs/` match paths (`runs/r1_lg2/attack/latest`, etc.). |
| **JAX defaults to CPU (`CpuDevice`)** | CUDA libraries not found by JAX | Verify NVIDIA drivers and run `uv pip install -U "jax[cuda12]"` inside your environment. |
