"""Stage 3 — renewal cost sampler.

Renewal cost for a component is ``unit_rate * extent`` scaled by a lognormal
multiplier with unit mean and coefficient of variation ``cv``. The lognormal is
parameterised so its mean is exactly 1.0, leaving the expected cost equal to the
deterministic ``unit_rate * extent`` while introducing right-skewed dispersion.
"""
from __future__ import annotations

import math

import numpy as np


def sample_cost(
    unit_rate: float, cv: float, extent: float, rng: np.random.Generator
) -> float:
    """Draw one renewal cost for an item of size ``extent``.

    With ``cv <= 0`` the cost is deterministic. Otherwise a unit-mean lognormal
    multiplier is applied: ``sigma = sqrt(log(1 + cv**2))`` and
    ``mu = -0.5 * sigma**2`` give ``E[multiplier] = 1``, so the expected cost is
    ``unit_rate * extent``.
    """
    base = unit_rate * extent
    if cv <= 0:
        return float(base)
    sigma = math.sqrt(math.log(1.0 + cv * cv))
    mu = -0.5 * sigma * sigma
    return float(base * rng.lognormal(mu, sigma))
