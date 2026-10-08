"""Balance evaluation tool sweeping speed ratios, r_tag, and assessing win/all-out rates."""

import os

os.environ["SDL_VIDEODRIVER"] = "dummy"

import jax
import numpy as np
from kilithattu_agents.scripted.crosser import ScriptedCrosserPolicy
from kilithattu_agents.scripted.guard import ScriptedGuardPolicy
from kilithattu_agents.scripted.kili import ScriptedKiliPolicy
from kilithattu_core.config import EnvParams
from kilithattu_core.env import KilithattuEnv
from kilithattu_game.controllers.scripted import ScriptedController
from kilithattu_game.slots import SlotManager


def evaluate_balance(
    num_innings: int = 5,
    r_tag: float = 0.6,
    crosser_speed: float = 5.0,
    guard_speed: float = 4.2,
    kili_speed: float = 4.8,
) -> dict[str, float]:
    params = EnvParams(
        r_tag=r_tag,
        classic_crosser_speed=crosser_speed,
        classic_guard_speed=guard_speed,
        classic_kili_speed=kili_speed,
        innings_duration=30.0,
    )
    env = KilithattuEnv(params)
    slot_manager = SlotManager(params)

    crosser_ai = ScriptedCrosserPolicy()
    guard_ai = ScriptedGuardPolicy()
    kili_ai = ScriptedKiliPolicy()

    for i in range(params.num_crossers):
        slot_manager.assign_controller(i, ScriptedController(crosser_ai))
    for k in range(params.num_guards):
        slot_manager.assign_controller(6 + k, ScriptedController(guard_ai))
    slot_manager.assign_controller(6 + 8, ScriptedController(kili_ai))

    jitted_step = jax.jit(lambda k, s, a: env.step(k, s, a, params))

    key = jax.random.PRNGKey(42)
    scores = []
    all_out_count = 0

    for ep in range(num_innings):
        k_reset, key = jax.random.split(key)
        obs, state = env.reset(k_reset, params)
        for s in slot_manager.slots:
            ctrl = slot_manager.controllers.get(s.slot_idx)
            if ctrl:
                ctrl.reset(s.slot_idx, ep)

        for t in range(200):  # 10s per inning
            joint_actions = slot_manager.get_joint_actions(obs)
            k_step, key = jax.random.split(key)
            obs, state, r_A, r_D, events = jitted_step(k_step, state, joint_actions)
            if bool(state.global_state.done):
                break

        pts = int(state.global_state.score_attack)
        scores.append(pts)
        all_out = all(int(state.crossers.status[i]) == 0 for i in range(params.num_crossers))
        if all_out:
            all_out_count += 1

    mean_ppi = float(np.mean(scores))
    all_out_rate = float(all_out_count / num_innings)
    print(
        f"r_tag={r_tag:.2f} | speeds=({crosser_speed}, {guard_speed}, {kili_speed}) -> PPI={mean_ppi:.2f} | All-out rate={all_out_rate * 100:.1f}%"
    )
    return {"ppi": mean_ppi, "all_out_rate": all_out_rate}


if __name__ == "__main__":
    evaluate_balance(num_innings=5)
