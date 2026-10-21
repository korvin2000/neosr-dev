"""Process-wide numpy stream for data and degradation code. Imports nothing from neosr.

Consumers bind it once (`from neosr.utils.rng import rng`); it is reseeded IN PLACE and never
rebound, so every binding sees the new state.
"""

from typing import Any

import numpy as np

rng: np.random.Generator = np.random.default_rng()


def seed_rng(seed: int) -> None:
    rng.bit_generator.state = np.random.PCG64(seed).state


def get_state() -> dict[str, Any]:
    return rng.bit_generator.state


def set_state(state: dict[str, Any]) -> None:
    rng.bit_generator.state = state
