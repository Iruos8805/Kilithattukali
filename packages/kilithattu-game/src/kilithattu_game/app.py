"""Kilithattu Kali: the pygame client. ``kilithattu --side attack --difficulty medium``."""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

from kilithattu_agents import load_policy
from kilithattu_core import load_params
from kilithattu_core.replay import read_replay

from .controllers import team_policies
from .input.keyboard import HumanInput
from .session import Seat, Session

SIDES = {"attack": "attack", "guard": "defence", "kili": "defence", "watch": None}


def build_session(args) -> tuple[Session, HumanInput | None]:
    p = load_params(args.rules)
    att, dfn = team_policies(p, args.difficulty)
    if args.attack:  # e.g. path:runs/finetune_v2/attack/latest  (teammates when you play attack, opponents otherwise)
        att = load_policy(args.attack, "attack", p)
    if args.defence:
        dfn = load_policy(args.defence, "defence", p)
    if args.side == "watch":
        return Session(p, att, dfn, None, args.seed), None
    if args.side == "attack":
        seat, hi = Seat("attack", args.index), HumanInput("attack", False)
    elif args.side == "guard":
        seat, hi = Seat("defence", args.index), HumanInput("defence", False)
    else:
        seat, hi = Seat("defence", p.n_defenders - 1), HumanInput("defence", True)
    return Session(p, att, dfn, seat, args.seed), hi


def run_headless(sess: Session, hi: HumanInput | None, frames: int) -> dict:
    """Simulation without a window (used by tests and CI): the human seat idles."""
    for _ in range(frames):
        sess.tick({"move": 0} if hi else None)
        if sess.done:
            break
    return {"ticks": sess.t, "points": float(sess.state.points), "done": sess.done}


def load_with_screen(screen, args):
    """Build the session and compile its jitted functions in a worker thread, while the main thread keeps
    pumping window events and drawing a loading screen (otherwise the desktop reports "not responding")."""
    import threading

    import pygame

    box: dict = {}

    def work():
        try:
            if args.replay:
                header, log = read_replay(args.replay)
                rp = load_params(header["params"])
                sess = Session(rp, *team_policies(rp, args.difficulty), None, header["seed"], replay_log=log)
                hi = None
            else:
                sess, hi = build_session(args)
            sess.tick({"move": 0} if hi else None)  # first tick triggers the compilation
            sess.reset()
            box["v"] = (sess, hi)
        except Exception as e:  # reported on the main thread
            box["e"] = e

    th = threading.Thread(target=work, daemon=True)
    th.start()
    font, small, clock, n = pygame.font.Font(None, 48), pygame.font.Font(None, 24), pygame.time.Clock(), 0
    while th.is_alive():
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return None
        screen.fill((14, 29, 21))
        W, H = screen.get_size()
        t = font.render("Loading" + "." * (1 + n // 10 % 3), True, (241, 228, 189))
        screen.blit(t, (W // 2 - t.get_width() // 2, H // 2 - 30))
        t = small.render("Compiling the policies. The first start can take up to a minute.", True, (160, 180, 150))
        screen.blit(t, (W // 2 - t.get_width() // 2, H // 2 + 20))
        pygame.display.flip()
        clock.tick(30)
        n += 1
    if "e" in box:
        raise box["e"]
    return box["v"]


def play(screen, args, use_menu: bool) -> str:
    """Run one game. Returns 'menu' or 'quit'."""
    import pygame

    from .render.field import View

    clock = pygame.time.Clock()
    loaded = load_with_screen(screen, args)
    if loaded is None:  # window closed while loading
        return "quit"
    sess, hi = loaded
    view = View(screen, sess.p, args.colorblind, args.reduce_motion)
    held: set[str] = set()
    acc = 0.0
    last = time.perf_counter()
    names = {pygame.K_w: "w", pygame.K_a: "a", pygame.K_s: "s", pygame.K_d: "d", pygame.K_UP: "up", pygame.K_DOWN: "down",
             pygame.K_LEFT: "left", pygame.K_RIGHT: "right", pygame.K_LSHIFT: "shift", pygame.K_RSHIFT: "shift", pygame.K_SPACE: "space", pygame.K_e: "e"}
    running, saved, frames, result = True, False, 0, "quit"
    while running:
        frames += 1
        if args.max_frames and frames > args.max_frames:
            break
        pressed: set[str] = set()
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                running, result = False, ("menu" if use_menu else "quit")
            elif ev.type == pygame.VIDEORESIZE:
                view.layout()
            elif ev.type == pygame.KEYDOWN:
                if ev.key in names:
                    held.add(names[ev.key]); pressed.add(names[ev.key])
                elif ev.key == pygame.K_TAB and sess.human and sess.human.side == "attack":
                    alive = [i for i in range(sess.p.n_crossers) if int(sess.state.c_status[i]) != 2]
                    if alive:
                        nxt = [i for i in alive if i > sess.human.index] or alive
                        sess.human = Seat("attack", nxt[0])
                elif ev.key == pygame.K_m and use_menu:
                    running, result = False, "menu"
                elif ev.key == pygame.K_r and sess.done:
                    sess.seed += 1; sess.reset(); saved = False
            elif ev.type == pygame.KEYUP and ev.key in names:
                held.discard(names[ev.key])
        now = time.perf_counter()
        acc += min(now - last, 0.1); last = now
        while acc >= sess.p.dt and not sess.done:
            act = hi.action(held, pressed, sess.human_mask()) if hi and sess.human else None
            pressed = set()
            for e in sess.tick(act):
                x, y = (e["pos"] if "pos" in e else (sess.p.field_w / 2, 0.0))
                msg = {"crossed": ("+1", view.pal["pacha"]), "delivered": ("Salt +6", view.pal["gold"]), "tagged": ("Tagged!", view.pal["foul"]),
                       "fouled": ("Foul!", view.pal["foul"]), "side_exit": ("Out!", view.pal["foul"]), "clap": ("Clap!", view.pal["gold"])}.get(e["type"])
                if msg:
                    view.add_float(msg[0], float(x), float(y), msg[1])
            acc -= sess.p.dt
        seat_agent = sess.human_agent
        view.draw(sess, min(1.0, acc / sess.p.dt), clock.get_time() / 1000, seat_agent)
        if sess.done:
            if not saved and not args.replay:
                out = Path(args.replay_dir); out.mkdir(parents=True, exist_ok=True)
                sess.save_replay(out / f"game_{int(time.time())}.kreplay", {"side": args.side, "difficulty": args.difficulty}); saved = True
            t = view.big.render(f"Innings over: {int(sess.state.points)} points   (R: again, M/Esc: menu)", True, view.pal["text"])
            screen.blit(t, (screen.get_width() // 2 - t.get_width() // 2, screen.get_height() // 2))
        pygame.display.flip()
        clock.tick(60)
    return result


def run_window(args) -> None:
    import pygame

    from .menu import discover_models, run_menu

    pygame.init()
    screen = pygame.display.set_mode((960, 780), pygame.RESIZABLE)
    pygame.display.set_caption("Kilithattu Kali")
    use_menu = args.side is None
    message = ""
    while True:
        if use_menu:
            ch = run_menu(screen, discover_models(args.runs_dir), message)
            if ch is None:
                break
            args.side, args.attack, args.defence, args.rules = ch["side"], ch["attack"], ch["defence"], ch["rules"]
        try:
            result = play(screen, args, use_menu)
        except Exception as e:  # e.g. a checkpoint that cannot be loaded: go back to the menu with the reason
            if not use_menu:
                raise
            message, result = f"Could not start: {type(e).__name__}: {str(e)[:90]}", "menu"
        else:
            message = ""
        if result == "quit" or not use_menu:
            break
    pygame.quit()


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="kilithattu", description="Play Kilithattu Kali against trained or scripted bots")
    ap.add_argument("--side", choices=list(SIDES), default=None, help="crosser (attack), guard, Kili, or watch the bots")
    ap.add_argument("--index", type=int, default=0, help="which crosser or guard you control")
    ap.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="medium")
    ap.add_argument("--rules", default="classic", help="classic | plus | path to a rules yaml")
    ap.add_argument("--attack", default=None, help="policy ref for the attack team, e.g. path:runs/finetune_v2/attack/latest")
    ap.add_argument("--defence", default=None, help="policy ref for the defence team, e.g. path:runs/defence_v3/defence/best")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--replay", default=None, help="watch a .kreplay folder")
    ap.add_argument("--replay-dir", default="replays")
    ap.add_argument("--runs-dir", default="runs", help="where the in-game menu looks for trained checkpoints")
    ap.add_argument("--colorblind", action="store_true")
    ap.add_argument("--reduce-motion", action="store_true")
    ap.add_argument("--max-frames", type=int, default=0, help="exit the window after N frames (CI smoke test)")
    ap.add_argument("--headless-ticks", type=int, default=0, help="run N ticks without a window and print the result (CI)")
    return ap


def enable_jit_cache() -> None:
    try:
        import jax

        cache = Path.home() / ".cache" / "kilithattu_jax"
        cache.mkdir(parents=True, exist_ok=True)
        jax.config.update("jax_compilation_cache_dir", str(cache))
    except Exception:  # an optional speed-up only
        pass


def main(argv=None) -> None:
    args = parser().parse_args(argv)
    enable_jit_cache()
    if args.headless_ticks:
        args.side = args.side or "attack"
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        sess, hi = build_session(args)
        print(run_headless(sess, hi, args.headless_ticks))
        return
    run_window(args)


if __name__ == "__main__":
    main()
