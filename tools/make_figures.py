"""Figures and a markdown table from the per-seed cross-tables written by tools/pipeline.py.

  uv run python tools/make_figures.py                 # reads runs/p[0-9]_table.json (+ the _eq variant if present)
Writes docs/figures/cross_table*.png, docs/figures/ratio_ablation.png and docs/figures/results_table.md
"""
import argparse
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ATT = ["scripted crosser", "fine-tuned attack", "league attack"]
DEF = ["scripted guards", "imitation defence", "league defence"]


def load(runs, tag, prefix="p"):
    files = sorted(glob.glob(f"{runs}/{prefix}[0-9]{tag}_table.json"))
    if not files:
        return None
    t = [json.load(open(f)) for f in files]
    return np.stack([np.array(x["points"]) for x in t]), np.stack([np.array(x["trips"]) for x in t])


def sd(x, axis=0):
    return x.std(axis, ddof=1) if x.shape[axis] > 1 else np.zeros_like(x.mean(axis))


def heatmap(pts, title, out):
    m, s = pts.mean(0), sd(pts)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    im = ax.imshow(m, cmap="YlOrRd", vmin=0, vmax=max(40, float(m.max())))
    ax.set_xticks(range(3), [d.replace(" ", "\n") for d in DEF]); ax.set_yticks(range(3), ATT)
    ax.set_xlabel("defence"); ax.set_ylabel("attack")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{m[i, j]:.1f}\n± {s[i, j]:.1f}", ha="center", va="center", fontsize=9, color="black" if m[i, j] < 0.6 * im.norm.vmax else "white")
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, label="mean attack points per innings"); fig.tight_layout(); fig.savefig(str(out), dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="p"); ap.add_argument("--runs-dir", default="runs"); ap.add_argument("--out", default="docs/figures")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    variants = {"": "defence-heavy updates (10:25, 8:30)", "_eq": "equal updates (25:25)"}
    found = {}  # only the tables that exist, so no value below can be None
    for t in variants:
        got = load(a.runs_dir, t, a.prefix)
        if got is not None:
            found[t] = got
    md = []
    for tag, label in variants.items():
        if tag not in found:
            continue
        pts, trips = found[tag]
        heatmap(pts, f"Attack points per innings, {label}\n(mean ± std over {pts.shape[0]} seeds)", out / f"cross_table_{a.prefix}{tag}.png")
        m, s, tm = pts.mean(0), sd(pts), trips.mean(0)
        md += [f"### {label} ({pts.shape[0]} seeds)", "", "| attack \\ defence | " + " | ".join(DEF) + " |", "|---|" + "---|" * 3]
        md += [f"| {ATT[i]} | " + " | ".join(f"{m[i, j]:.1f} ± {s[i, j]:.1f} ({tm[i, j]:.2f} trips)" for j in range(3)) + " |" for i in range(3)]
        wa, wd = pts.min(2), pts.max(1)
        md += ["", "Worst case (mean ± std over seeds): attacks " + ", ".join(f"{ATT[i]} {wa.mean(0)[i]:.1f} ± {sd(wa)[i]:.1f}" for i in range(3))
               + "; defences " + ", ".join(f"{DEF[j]} {wd.mean(0)[j]:.1f} ± {sd(wd)[j]:.1f}" for j in range(3)), ""]
    if all(t in found for t in variants):
        cells = [("league attack\nvs scripted guards", 2, 0), ("league attack\nvs imitation def.", 2, 1), ("league attack\nvs league def.", 2, 2),
                 ("scripted crosser\nvs league def.", 0, 2), ("fine-tuned attack\nvs league def.", 1, 2)]
        fig, ax = plt.subplots(figsize=(9, 3.8))
        w = 0.38
        for k, (tag, label) in enumerate(variants.items()):
            pts = found[tag][0]
            mm = [pts[:, i, j].mean() for _, i, j in cells]; ss = [sd(pts[:, i, j]) for _, i, j in cells]
            ax.bar(np.arange(len(cells)) + (k - 0.5) * w, mm, w, yerr=ss, capsize=3, label=label)
        ax.set_xticks(range(len(cells)), [c[0] for c in cells], fontsize=8)
        ax.set_ylabel("attack points per innings"); ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
        ax.set_title("Effect of the league update ratio (mean ± std over seeds; error bars are large, n is small)", fontsize=9)
        fig.tight_layout(); fig.savefig(str(out / f"ratio_ablation_{a.prefix}.png"), dpi=150); plt.close(fig)
    (out / f"results_table_{a.prefix}.md").write_text("\n".join(md))
    print("wrote:", ", ".join(sorted(p.name for p in out.iterdir())))


if __name__ == "__main__":
    main()