"""A team whose members are controlled by two different policies (for cross-play / teammate evaluation)."""
from __future__ import annotations

import jax
import jax.numpy as jnp

tm = jax.tree_util.tree_map


class MixedTeam:
    """Slot i is played by ``pol_b`` where ``mask[i]`` is True, else by ``pol_a``. Both policies see the full observation."""

    def __init__(self, pol_a, pol_b, mask):
        assert pol_a.side == pol_b.side
        self.side, self.pol_a, self.pol_b, self.mask = pol_a.side, pol_a, pol_b, jnp.asarray(mask, bool)

    def init_carry(self, key):
        ka, kb = jax.random.split(key)
        return (self.pol_a.init_carry(ka), self.pol_b.init_carry(kb))

    def act(self, key, carry, obs):
        k1, k2 = jax.random.split(key)
        a, ca, _ = self.pol_a.act(k1, carry[0], obs)
        b, cb, _ = self.pol_b.act(k2, carry[1], obs)
        return tm(lambda x, y: jnp.where(self.mask, y, x), a, b), (ca, cb), {}
