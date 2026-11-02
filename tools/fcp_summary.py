"""Summarise tools/fcp_experiment.py evaluation files: FCP learner vs the matched self-play control as a teammate.

  uv run python tools/fcp_summary.py "runs/fcp_r*/eval_*.json"
For every file (one held-out partner) it prints the mixed-team scores, the difference with a z-score, and the 'synergy':
observed team score minus what independent crossers would give, 0.25 * solo(X) + 0.75 * solo(partner).
"""
import glob
import json
import math
import sys


def main():
    files = sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else "runs/fcp_r*/eval_*.json"))
    if not files:
        raise SystemExit("no evaluation files found")
    print(f"{'replicate / held-out partner':<34}{'FCP team':>11}{'control team':>14}{'diff (z)':>16}{'FCP synergy':>13}{'control synergy':>17}")
    diffs = []
    for f in files:
        d = json.load(open(f))
        get = lambda s: next(v for k, v in d.items() if k.endswith(s))
        fcp, ctl = get("FCP learner + held-out partners"), get("self-play baseline + held-out partners")
        fcp_solo, ctl_solo, part = get("FCP learner alone (x4)")[0], get("self-play baseline alone (x4)")[0], get("held-out partner alone (x4)")[0]
        diff = fcp[0] - ctl[0]
        se = math.hypot(fcp[1], ctl[1])
        diffs.append((diff, se))
        syn_f = fcp[0] - (0.25 * fcp_solo + 0.75 * part)
        syn_c = ctl[0] - (0.25 * ctl_solo + 0.75 * part)
        label = f.split("/")[-2].replace("fcp_", "") + " / " + f.split("eval_")[-1].replace(".json", "")
        print(f"{label:<34}{fcp[0]:>8.1f}+-{fcp[1]:<3.1f}{ctl[0]:>9.1f}+-{ctl[1]:<3.1f}{diff:>+9.1f} ({diff / se:+.1f}){syn_f:>+13.1f}{syn_c:>+17.1f}")
    n = len(diffs)
    mean = sum(x for x, _ in diffs) / n
    se_mean = math.sqrt(sum(s * s for _, s in diffs)) / n          # cells treated as independent
    print(f"\nmean difference (FCP minus control) over {n} cells: {mean:+.1f} +- {se_mean:.1f}  (z = {mean / se_mean:+.1f})")
    print("cells share the same learner and control, so they are not independent; treat the z-score as optimistic.")


if __name__ == "__main__":
    main()