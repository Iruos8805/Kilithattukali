"""Training curves from the metrics.jsonl files written by the trainers.

  uv run python tools/plot_training.py --runs r1_ft r2_ft r3_ft --league r1_lg1 r1_lg2 --out docs/figures/training_curves.png
Panels: (a) evaluation points per innings during fine-tuning, (b) policy entropy, (c) league phase scores.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def read(path):
    p = Path(path)
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def smooth(ys, w=3):
    return [sum(ys[max(0, i - w + 1): i + 1]) / len(ys[max(0, i - w + 1): i + 1]) for i in range(len(ys))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=[], help="fine-tune run ids, e.g. r1_ft r2_ft r3_ft")
    ap.add_argument("--league", nargs="*", default=[], help="league run ids, e.g. r1_lg1 r1_lg2")
    ap.add_argument("--runs-dir", default="runs"); ap.add_argument("--out", default="docs/figures/training_curves.png")
    a = ap.parse_args()
    panels = (1 if a.runs else 0) * 2 + (1 if a.league else 0)
    if not panels:
        raise SystemExit("give --runs and/or --league")
    fig, axes = plt.subplots(1, panels, figsize=(5.2 * panels, 3.8))
    axes = [axes] if panels == 1 else list(axes)
    i = 0
    if a.runs:
        ax1, ax2 = axes[0], axes[1]
        for r in a.runs:
            rows = read(f"{a.runs_dir}/{r}/metrics.jsonl")
            ev = [(x["step"], x["eval/points"]) for x in rows if x.get("tag") == "eval" and "eval/points" in x]
            tr = [(x["step"], x["a/ent"]) for x in rows if x.get("tag") == "train" and "a/ent" in x]
            if ev:
                ax1.plot(*zip(*ev), marker="o", ms=3, label=r)
            if tr:
                ax2.plot([s for s, _ in tr], smooth([v for _, v in tr]), label=r)
        ax1.set_xlabel("update"); ax1.set_ylabel("evaluation points per innings (vs scripted guards)"); ax1.set_title("(a) fine-tuning progress"); ax1.grid(alpha=0.3); ax1.legend(fontsize=8)
        ax2.set_xlabel("update"); ax2.set_ylabel("policy entropy (nats)"); ax2.set_title("(b) entropy falls as the policy sharpens"); ax2.grid(alpha=0.3); ax2.legend(fontsize=8)
        i = 2
    if a.league:
        ax = axes[i]
        for r in a.league:
            rows = [x for x in read(f"{a.runs_dir}/{r}/metrics.jsonl") if x.get("tag") == "league" and "attack_phase_points" in x]
            if rows:
                ax.plot([x["step"] for x in rows], [x["attack_phase_points"] for x in rows], label=f"{r}: attack trained")
                ax.plot([x["step"] for x in rows], [x["defence_phase_points"] for x in rows], "--", label=f"{r}: defence trained")
        ax.set_xlabel("league phase"); ax.set_ylabel("attack points per innings"); ax.set_title("(c) league: the sides trade the lead"); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(a.out), dpi=150); plt.close(fig)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
