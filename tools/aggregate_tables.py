"""Combine the per-seed cross-tables written by tools/pipeline.py into mean +- std over seeds.

  uv run python tools/aggregate_tables.py            # reads runs/p*_table.json
"""
import glob
import json

import numpy as np

ATT = ["scripted crosser", "fine-tuned attack", "league attack"]
DEF = ["scripted guards", "imitation defence", "league defence"]


def ms(x, axis=0):
    n = x.shape[axis]
    return x.mean(axis), (x.std(axis, ddof=1) if n > 1 else np.full(x.mean(axis).shape, np.nan))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="", help="aggregate the tagged variant, e.g. --tag _eq")
    ap.add_argument("--prefix", default="p", help="p = classic runs, q = plus-rules runs")
    args = ap.parse_args()
    tag, prefix = args.tag, args.prefix
    files = sorted(glob.glob(f"runs/{prefix}[0-9]{tag}_table.json"))
    if not files:
        raise SystemExit(f"no runs/{prefix}[0-9]{tag}_table.json found; run tools/pipeline.py first")
    tabs = [json.load(open(f)) for f in files]
    pts, trips = np.stack([np.array(t["points"]) for t in tabs]), np.stack([np.array(t["trips"]) for t in tabs])
    print(f"{len(files)} seed(s): {', '.join(f.split('/')[-1].replace('_table.json', '') for f in files)}   ({tabs[0]['n']} innings per cell)\n")
    m, s = ms(pts)
    tm, _ = ms(trips)
    print(f"{'mean attack points +- std over seeds (trips)':<46}" + "".join(f"{d:>26}" for d in DEF))
    for i, a in enumerate(ATT):
        print(f"{a:<46}" + "".join(f"{m[i, j]:>13.1f} +- {s[i, j]:<5.1f}({tm[i, j]:.1f})" for j in range(len(DEF))))
    wa, wd = pts.min(2), pts.max(1)           # per seed: worst case of each attack / defence
    wam, was = ms(wa); wdm, wds = ms(wd)
    print("\nworst case over the three opponents (per seed, then mean +- std)")
    print("  attacks :", "   ".join(f"{a}: {wam[i]:.1f} +- {was[i]:.1f}" for i, a in enumerate(ATT)))
    print("  defences:", "   ".join(f"{d}: {wdm[j]:.1f} +- {wds[j]:.1f}" for j, d in enumerate(DEF)))
    beat = (pts[:, 2, 0] > pts[:, 0, 0]).mean()
    print(f"\nlearned league attack beats the scripted crosser against the scripted guards in {int(beat * len(files))} of {len(files)} seeds")


if __name__ == "__main__":
    main()
