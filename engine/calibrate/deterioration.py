"""Stage 3 — deterioration priors.

Useful life is drawn from a Weibull distribution whose shape ``k`` is implied by
the coefficient of variation ``cv = sigma / mu`` of the component's useful-life
prior, and whose scale is chosen so the distribution mean equals ``mu``. Asset
condition then follows a convex age curve rising from 1.0 (as-new) at age 0 to
5.0 (failed) at the realised useful life.
"""
from __future__ import annotations

import math

import numpy as np


def _weibull_shape_from_cv(cv: float) -> float:
    """Solve for Weibull shape ``k`` matching a target coefficient of variation.

    The Weibull cv depends only on ``k`` via
    ``cv = sqrt(gamma(1 + 2/k) / gamma(1 + 1/k)**2 - 1)``. That relation is
    monotonically decreasing in ``k``, so we bisect for the ``k`` that reproduces
    the requested ``cv``.
    """
    if cv <= 0:
        return 50.0

    def cv_of_k(k: float) -> float:
        g1 = math.gamma(1.0 + 1.0 / k)
        g2 = math.gamma(1.0 + 2.0 / k)
        return math.sqrt(g2 / (g1 * g1) - 1.0)

    lo, hi = 0.05, 50.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if cv_of_k(mid) > cv:
            # cv too large → distribution too dispersed → need larger shape k
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def weibull_useful_life(mu: float, sigma: float, rng: np.random.Generator) -> float:
    """Draw one useful-life value from a Weibull with mean ``mu`` and cv ``sigma/mu``."""
    cv = sigma / mu if mu > 0 else 0.0
    k = _weibull_shape_from_cv(cv)
    scale = mu / math.gamma(1.0 + 1.0 / k)
    return float(scale * rng.weibull(k))


def condition_at_age(age: float, useful_life: float, exponent: float = 2.2) -> float:
    """Condition index (1.0 = as-new … 5.0 = failed) at ``age`` years.

    Convex curve ``1.0 + 4.0 * t**exponent`` with ``t`` the clamped fraction of
    useful life elapsed, so age 0 → 1.0 and age == useful_life → 5.0.
    """
    if useful_life <= 0:
        return 5.0
    t = max(0.0, min(1.0, age / useful_life))
    return 1.0 + 4.0 * t**exponent
