"""DAgger for the scripted attack OR defence (--side).

Round 0 is plain behaviour cloning on full-length innings (the expert acts). In later rounds the learner acts
(with probability beta the expert acts instead, beta halves each round) while the scripted expert, run on the
same observations, provides the labels. Training uses windows of ``--window`` ticks with a zero carry at the
start of each window. Run:  uv run python tools/dagger.py --rounds 5
"""
import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from kilithattu_agents import NetworkPolicy, ScriptedAttack, ScriptedDefence, load_policy, save_checkpoint
from kilithattu_agents.networks import Actor
from kilithattu_agents.policy import init_networks, join_actions, log_prob_heads, slice_obs, team_forward, team_slice
from kilithattu_core import load_params, make_env
from kilithattu_train.config import NetCfg

tm = jax.tree_util.tree_map


HEADS = ("move", "dash", "turn", "clap", "call")


def sanitize(a, mask):
    """Replace labels the action mask forbids by 0 (stay / no-op), exactly what the environment would execute."""
    out = {}
    for h in HEADS:
        x = getattr(a, h)
        ok = jnp.take_along_axis(getattr(mask, h), x[:, None], 1)[:, 0]
        out[h] = jnp.where(ok, x, 0).astype(jnp.int32)
    return type(a)(**out)


def make_roller(p, net, side, expert, opp, n_envs, T):
    """Roll the mixed learner/expert policy of ``side`` against ``opp``; label every state with the expert's action."""
    env = make_env(p)
    sl = team_slice(p, side)
    n = sl.stop - sl.start

    def run(actor_params, beta, key):
        learner = NetworkPolicy(side, p, net, actor_params)

        def one(key):
            k0, k1, k2, k3 = jax.random.split(key, 4)
            obs, s = env.reset(k0)
            init = (s, obs, expert.init_carry(k1), opp.init_carry(k2), learner.init_carry(), jnp.ones((), bool), k3)

            def body(c, _):
                s, obs, ce, co, cl, reset, key = c
                key, ka, kb, kc, kd, ke, kf = jax.random.split(key, 7)
                ot = slice_obs(obs, sl)
                ae, ce2, _ = expert.act(ka, ce, obs)
                ae = sanitize(ae, ot.mask)
                al, cl2, _ = learner.act(kb, cl, obs, reset=jnp.broadcast_to(reset, (n,)))
                use_exp = jax.random.uniform(kc) < beta
                own = tm(lambda e, l: jnp.where(use_exp, e, l), ae, al)
                ao, co2, _ = opp.act(kd, co, obs)
                acts = join_actions(own, ao) if side == "attack" else join_actions(ao, own)
                obs2, s2, _, done, _ = env.step(ke, s, acts)
                robs, rs = env.reset(kf)
                pick = lambda a, b: jnp.where(done, a, b)
                s3, obs3 = tm(pick, rs, s2), tm(pick, robs, obs2)
                ce3 = tm(pick, expert.init_carry(kf), ce2)
                co3 = tm(pick, opp.init_carry(kf), co2)
                cl3 = jnp.where(done, 0.0, cl2)
                return (s3, obs3, ce3, co3, cl3, done, key), (ot, ae, reset)

            return jax.lax.scan(body, init, None, length=T)[1]

        return jax.vmap(one)(jax.random.split(key, n_envs))

    return jax.jit(run)


def to_windows(x, W):
    """(E, T, ...) -> (W, E*T//W, ...) with the window axis first."""
    E, T = x.shape[:2]
    x = x[:, : T // W * W].reshape((E, T // W, W) + x.shape[2:])
    return jnp.moveaxis(x, 2, 0).reshape((W, E * (T // W)) + x.shape[3:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", choices=["attack", "defence"], default="attack")
    ap.add_argument("--opponents", default=None, help="comma-separated policy refs for the OTHER side, used round-robin")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rules", default="classic", help="classic | plus | path")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--envs", type=int, default=6)
    ap.add_argument("--ticks", type=int, default=2400)
    ap.add_argument("--window", type=int, default=128)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--minibatch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-episodes", type=int, default=32)
    ap.add_argument("--reaction", type=float, default=0.4)
    ap.add_argument("--crossers", type=int, default=4)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-gru", action="store_true", help="feed-forward actor (no recurrent state, so no train/test carry mismatch)")
    ap.add_argument("--init", default=None, help="optional actor checkpoint to start from")
    a = ap.parse_args()

    p = load_params(a.rules).replace(n_crossers=a.crossers)
    net = NetCfg(gru=not a.no_gru).build()
    side = a.side
    sl = team_slice(p, side)
    n = sl.stop - sl.start
    out_dir = a.out or f"runs/dagger/{side}"
    if side == "attack":
        expert = ScriptedAttack(p)
        refs = (a.opponents or f"scripted:guard?reaction={a.reaction}").split(",")
        opps = [load_policy(r, "defence", p) for r in refs]
    else:
        expert = ScriptedDefence(p, reaction=a.reaction)
        refs = (a.opponents or "scripted:crosser").split(",")
        opps = [load_policy(r, "attack", p) for r in refs]
    rollers = {}
    _, _, actor_params, _ = init_networks(jax.random.PRNGKey(a.seed), p, net)
    if a.init:
        from kilithattu_agents import load_checkpoint
        actor_params = load_checkpoint(a.init)[0]
    actor = Actor(net)
    opt = optax.adam(a.lr)
    ost = opt.init(actor_params)

    def loss_fn(params, mb):
        o, y, rs = mb

        def f(c, x):
            ob, r = x
            c2, lg = jax.vmap(lambda cc, oo, rr: team_forward(actor, params, cc, oo, rr))(c, ob, jnp.broadcast_to(r[:, None], (r.shape[0], n)))
            return c2, lg

        _, logits = jax.lax.scan(f, jnp.zeros((rs.shape[1], n, net.hidden)), (o, rs))
        lp, _ = log_prob_heads(logits, o.mask, y)
        return -lp.mean()

    @jax.jit
    def step(params, ost, mb):
        l, g = jax.value_and_grad(loss_fn)(params, mb)
        up, ost = opt.update(g, ost, params)
        return optax.apply_updates(params, up), ost, l

    data = None
    rng = np.random.default_rng(a.seed)
    for r in range(a.rounds):
        beta = 1.0 if r == 0 else 0.5 ** r
        t0 = time.time()
        oi = r % len(opps)
        if oi not in rollers:
            rollers[oi] = make_roller(p, net, side, expert, opps[oi], a.envs, a.ticks)
        obs_s, exp_s, rs = rollers[oi](actor_params, jnp.float32(beta), jax.random.PRNGKey(1000 * a.seed + 100 + r))
        new = tm(lambda x: to_windows(x, a.window), (obs_s, exp_s, rs))
        new = (new[0], new[1], new[2].at[0].set(True))               # carry is zero at the start of every window
        data = new if data is None else tm(lambda x, y: jnp.concatenate([x, y], 1), data, new)
        B = data[2].shape[1]
        losses = []
        for _ in range(a.steps):
            idx = jnp.asarray(rng.choice(B, size=min(a.minibatch, B), replace=False))
            actor_params, ost, l = step(actor_params, ost, tm(lambda x: x[:, idx], data))
            losses.append(float(l))
        msg = f"round {r} beta={beta:.2f} windows={B} loss {losses[0]:.3f}->{losses[-1]:.3f}  ({time.time() - t0:.0f}s)"
        if a.eval_episodes:
            from kilithattu_eval.matches import play_match
            for ref, opp in zip(refs, opps):
                label = ref.split("/")[-2] if ref.startswith("path:") else ref.split("?")[0].replace("scripted:", "")
                for det in (False, True):
                    pol = NetworkPolicy(side, p, net, actor_params, deterministic=det)
                    m = play_match(p, pol, opp, a.eval_episodes, 7 + r) if side == "attack" else play_match(p, opp, pol, a.eval_episodes, 7 + r)
                    msg += f" | vs {label} {'det' if det else 'samp'}: pts {m['points_mean']:.1f} trips {m['trips_mean']:.2f}"
        print(msg, flush=True)
        save_checkpoint(out_dir, actor_params, p, net, side, "dagger", r)
    print("saved", out_dir)


if __name__ == "__main__":
    main()
