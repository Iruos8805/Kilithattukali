"""In-window menu: choose your role, the attack/defence policies (scripted levels or trained checkpoints), and the rules."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pygame

from kilithattu_core import ENV_VERSION

SCRIPTED = [("Scripted easy", "main@easy"), ("Scripted medium", "main@medium"), ("Scripted hard", "main@hard")]
ROLES = [("Crosser (attack)", "attack"), ("Guard (defence)", "guard"), ("Kili (defence)", "kili"), ("Watch only", "watch")]
RULES = [("Classic", "classic"), ("Plus (momentum, facing, clap)", "plus")]


def discover_models(runs_dir: str | Path = "runs") -> dict[str, list[tuple[str, str]]]:
    """Find checkpoints: runs/<run>/<side>/{latest,best,fcp}/ and flat runs/<run>/ (imitation outputs).
    League snapshots and per-stage folders are hidden to keep the menu short."""
    out: dict[str, list[tuple[str, str]]] = {"attack": [], "defence": []}
    root = Path(runs_dir)
    cur_minor = ENV_VERSION.split(".")[1]
    metas = list(root.glob("*/*/*/metadata.json")) + list(root.glob("*/metadata.json"))
    for meta in sorted(metas):
        d = meta.parent
        parts = d.relative_to(root).parts
        if len(parts) == 3 and (parts[1] == "league" or parts[2] not in ("latest", "best", "fcp")):
            continue
        try:
            m = json.loads(meta.read_text())
        except Exception:
            continue
        side = m.get("side")
        if side not in out or m.get("env_version", "0").split(".")[0] != ENV_VERSION.split(".")[0]:
            continue
        old = m.get("env_version", "").split(".")[1:2] != [cur_minor]
        label = "/".join([parts[0]] + ([parts[2]] if len(parts) == 3 else ["imitation"]))
        out[side].append((("[old] " if old else "") + label, f"path:{d}"))
    for side in out:
        out[side].sort(key=lambda t: (t[0].startswith("[old]"), t[0]))
    return out


def ladder_options(side: str) -> list[tuple[str, str]]:
    """Easy/medium/hard from configs/game/ladder.yaml, labelled by what they really are (scripted or trained)."""
    try:
        from kilithattu_agents import load_ladder

        lad = load_ladder()["levels"]
    except Exception:
        lad = {}
    out = []
    for level in ("easy", "medium", "hard"):
        e = lad.get(level, {}).get("main", {}).get(side, {})
        kind = "trained" if "path" in e else "scripted"
        out.append((f"Ladder: {level} ({kind})", f"main@{level}"))
    return out


def run_menu(screen: pygame.surface.Surface, models: dict, message: str = "") -> dict | None:
    """Returns the chosen configuration, or None if the player quits."""
    atk = ladder_options("attack") + [("Scripted reference crosser", "scripted:crosser")] + [(f"Trained: {l}", r) for l, r in models["attack"]]
    dfn = ladder_options("defence") + [("Scripted reference guards", "scripted:guard?reaction=0.4")] + [(f"Trained: {l}", r) for l, r in models["defence"]]
    rows = [("You play", ROLES), ("Attack team", atk), ("Defence team", dfn), ("Rules", RULES)]
    sel = [0, 1, 1, 0]
    row = 0
    if os.environ.get("KILITHATTU_MENU_AUTOSTART"):  # CI smoke test
        return _choice(rows, sel)
    big, font, small = pygame.font.Font(None, 56), pygame.font.Font(None, 32), pygame.font.Font(None, 22)
    clock = pygame.time.Clock()
    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                return None
            if ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_UP, pygame.K_w):
                    row = (row - 1) % (len(rows) + 1)
                elif ev.key in (pygame.K_DOWN, pygame.K_s):
                    row = (row + 1) % (len(rows) + 1)
                elif ev.key in (pygame.K_LEFT, pygame.K_a) and row < len(rows):
                    sel[row] = (sel[row] - 1) % len(rows[row][1])
                elif ev.key in (pygame.K_RIGHT, pygame.K_d) and row < len(rows):
                    sel[row] = (sel[row] + 1) % len(rows[row][1])
                elif ev.key in (pygame.K_RETURN, pygame.K_SPACE):
                    if row == len(rows):
                        return _choice(rows, sel)
                    row = len(rows)
        W, H = screen.get_size()
        screen.fill((14, 29, 21))
        t = big.render("Kilithattu Kali", True, (241, 228, 189)); screen.blit(t, (W // 2 - t.get_width() // 2, 36))
        t = small.render("Teammates use your side's team policy; opponents use the other.", True, (160, 180, 150)); screen.blit(t, (W // 2 - t.get_width() // 2, 92))
        y = 150
        for i, (name, opts) in enumerate(rows):
            active = i == row
            label = opts[sel[i]][0]
            screen.blit(font.render(name, True, (224, 173, 74) if active else (200, 200, 190)), (60, y))
            txt = font.render(("<  " + label + "  >") if active else label, True, (255, 255, 255) if active else (180, 190, 175))
            screen.blit(txt, (300, y))
            y += 62
        start = font.render("[ Start ]", True, (14, 29, 21) if row == len(rows) else (224, 173, 74))
        r = pygame.Rect(60, y + 10, start.get_width() + 30, 44)
        if row == len(rows):
            pygame.draw.rect(screen, (224, 173, 74), r, border_radius=8)
        else:
            pygame.draw.rect(screen, (224, 173, 74), r, 2, border_radius=8)
        screen.blit(start, (r.x + 15, r.y + 9))
        picked = [rows[1][1][sel[1]][1], rows[2][1][sel[2]][1]]
        if any(p.startswith("path:") for p in picked):
            screen.blit(small.render("Trained models were trained on Classic rules; choose Classic for them.", True, (226, 130, 100)), (60, y + 70))
        if rows[3][1][sel[3]][1] == "plus" and any(p.startswith("path:") for p in picked):
            screen.blit(small.render("Warning: Plus rules + a Classic-trained model will behave oddly.", True, (226, 87, 76)), (60, y + 96))
        if message:
            screen.blit(small.render(message, True, (226, 87, 76)), (60, H - 70))
        screen.blit(small.render("Up/Down: row   Left/Right: change   Enter: start   Esc: quit", True, (160, 180, 150)), (60, H - 36))
        pygame.display.flip()
        clock.tick(30)


def _choice(rows, sel) -> dict:
    return {"side": rows[0][1][sel[0]][1], "attack": rows[1][1][sel[1]][1], "defence": rows[2][1][sel[2]][1], "rules": rows[3][1][sel[3]][1]}
