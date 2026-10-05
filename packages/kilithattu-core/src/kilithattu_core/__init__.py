"""Kilithattu-Core: Rules engine, formal geometry, and vectorised JAX simulation."""

from kilithattu_core.config import EnvParams, load_env_params, rules_hash
from kilithattu_core.env import JointActions, KilithattuEnv, Observations, make_env
from kilithattu_core.events import StepEvents
from kilithattu_core.replay import read_replay, verify_replay, write_replay
from kilithattu_core.state import EnvState

__version__ = "0.1.0"
ENV_VERSION = "1.0.0"

__all__ = [
    "ENV_VERSION",
    "EnvParams",
    "EnvState",
    "JointActions",
    "KilithattuEnv",
    "Observations",
    "StepEvents",
    "load_env_params",
    "make_env",
    "read_replay",
    "rules_hash",
    "verify_replay",
    "write_replay",
]
