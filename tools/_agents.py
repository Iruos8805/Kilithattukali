"""Shared helper: the agents produced by tools/pipeline.py, as {name: (attack ref, defence ref)}."""
import os


def build_agents(seeds, tag="_eq", prefix="p", extra=None, scripted=True):
    ag = {"scripted": ("scripted:crosser", "scripted:guard?reaction=0.4")} if scripted else {}
    for s in seeds:
        pre = f"runs/{prefix}{s}"
        ag[f"s{s}_ft"] = (f"path:{pre}_ft/attack/latest", f"path:{pre}_dagger_defence")
        ag[f"s{s}_league"] = (f"path:{pre}_lg2{tag}/attack/latest", f"path:{pre}_lg2{tag}/defence/latest")
    for e in extra or []:
        name, refs = e.split("=", 1)
        a, d = refs.split(",")
        ag[name] = (a, d)
    missing = [r for a, d in ag.values() for r in (a, d) if r.startswith("path:") and not os.path.exists(os.path.join(r[5:], "metadata.json"))]
    if missing:
        print("warning: checkpoints not found:", sorted(set(missing)))
    return ag
