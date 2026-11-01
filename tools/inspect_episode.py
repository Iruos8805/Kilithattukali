"""Diagnose how points are scored: how close were the defenders when crossers crossed lines, and are they active?

Usage: uv run python tools/inspect_episode.py <attack ref> <defence ref> [--seeds 3] [--rules classic]
Refs: path:runs/.../latest | scripted:crosser | scripted:guard?reaction=0.4
"""
import argparse

import numpy as np

from kilithattu_agents import load_policy
from kilithattu_core import load_params
from kilithattu_eval.behaviours import trajectory


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("attack"); ap.add_argument("defence")
    ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--rules", default="classic")
    a = ap.parse_args()
    p = load_params(a.rules)
    att, dfn = load_policy(a.attack, "attack", p), load_policy(a.defence, "defence", p)
    gy = np.arange(1, p.n_guards + 1) * p.row_d
    dist_at_cross, trips, tags, steps = [], [], [], 0
    guard_travel, kili_travel, track_err = 0.0, 0.0, []
    for seed in range(a.seeds):
        tr = trajectory(p, att, dfn, seed)
        T = len(tr["crossed"]); steps += T
        trips.append(int((tr["kappa"][1:] < tr["kappa"][:-1]).sum()))        # kappa resets to 0 on delivery (or on being out)
        tags.append(int(tr["tagged"].sum()))
        guard_travel += float(np.abs(np.diff(tr["g_x"], axis=0)).sum())
        kili_travel += float(np.abs(np.diff(tr["k_pos"], axis=0)).sum())
        for t in range(T):
            for i in np.flatnonzero(tr["crossed"][t]):
                c = tr["c_pos"][t, i]
                dg = np.hypot(tr["g_x"][t] - c[0], gy - c[1]).min()
                dk = np.hypot(*(tr["k_pos"][t] - c))
                dist_at_cross.append(min(dg, dk))
        for t in range(0, T, 5):                                              # guard tracking error near lines
            alive = tr["status"][t] != 2
            for j in range(p.n_guards):
                d = np.abs(tr["c_pos"][t, :, 1] - gy[j]); d[~alive] = 99
                k = int(np.argmin(d))
                if d[k] < 2.0:
                    track_err.append(abs(tr["g_x"][t, j] - tr["c_pos"][t, k, 0]))
    d = np.array(dist_at_cross) if dist_at_cross else np.array([np.nan])
    secs = steps * p.dt
    print(f"innings: {a.seeds}   completed trips per innings (approx): {np.mean(trips):.1f}   tags per innings: {np.mean(tags):.1f}")
    print(f"line crossings: {len(dist_at_cross)}   distance to nearest defender at the moment of crossing:")
    for lo, hi in ((0, 1), (1, 2), (2, 4), (4, 99)):
        print(f"   {lo:>2}-{hi if hi < 99 else 'inf':<3} m : {100 * np.mean((d >= lo) & (d < hi)):5.1f}%")
    print(f"guard movement: {guard_travel / secs / p.n_guards:.2f} m/s per guard (max speed {p.v_guard})   Kili: {kili_travel / secs:.2f} m/s (max {p.v_kili})")
    print(f"guard lateral error to the nearest crosser when one is within 2 m of its line: {np.mean(track_err) if track_err else float('nan'):.2f} m")


if __name__ == "__main__":
    main()
