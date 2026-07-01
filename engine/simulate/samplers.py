"""Monte Carlo samplers for asset condition trajectories.

A trajectory is the year-by-year condition path of a single asset under a
no-renewal assumption. Each annual step combines three contributions:

* a deterministic base decay increment from the convex age curve in
  :mod:`engine.calibrate.deterioration`,
* a climate increment, the dot product of per-hazard ``component_factors`` with
  that year's ``hazard_intensities`` column, and
* Gaussian observation noise.

Because no renewal occurs, condition can only worsen (monotonic non-decreasing)
and is capped at the failed state of 5.0.
"""
from __future__ import annotations

import numpy as np

from engine.calibrate.deterioration import condition_at_age


def sample_trajectory(
    initial_condition: float,
    initial_age: float,
    useful_life: float,
    years: int,
    hazard_intensities: np.ndarray,
    component_factors: np.ndarray,
    rng: np.random.Generator,
    base_noise_sigma: float = 0.03,
) -> np.ndarray:
    """Sample a no-renewal condition trajectory over ``years`` annual steps.

    Parameters
    ----------
    initial_condition:
        Condition index (1.0 = as-new … 5.0 = failed) at the start of year 0.
    initial_age:
        Asset age in years at the start of the trajectory.
    useful_life:
        Realised useful life in years, used by the deterioration age curve.
    years:
        Number of annual steps to simulate.
    hazard_intensities:
        Array of shape ``(H, years)`` giving each hazard's intensity per year.
    component_factors:
        Array of shape ``(H,)`` giving each hazard's per-unit condition impact.
    rng:
        Source of Gaussian noise.
    base_noise_sigma:
        Standard deviation of the additive annual noise term.

    Returns
    -------
    numpy.ndarray
        A ``(years,)`` float array of condition values. The path is monotonic
        non-decreasing and bounded above by 5.0.
    """
    c0 = float(initial_condition)
    # Base decay increments follow the convex age curve from the asset's
    # CHRONOLOGICAL age. [H3 — flagged, NOT applied] The review proposed re-
    # anchoring the curve to the observed condition (taking increments from the
    # effective age where the curve equals c0). Re-anchoring is theoretically
    # reasonable, but with this convex curve (very flat early) the condition→
    # effective-age inversion is hyper-sensitive: an asset observed at, say, 1.5
    # maps to ~39% of useful life and is then deteriorated from there, breaching
    # within the horizon — an implausible acceleration of good-condition assets
    # (it ~6x'd portfolio demand on the Mitcham register). HOW to project from
    # observed condition vs chronological age is a deterioration-curve calibration
    # decision for the chartered engineer, not a silent code change. Left on the
    # chronological-age basis pending that sign-off (see the calc-solver review).
    age = float(initial_age)
    base = np.array(
        [
            condition_at_age(age + t + 1.0, useful_life)
            - condition_at_age(age + t, useful_life)
            for t in range(years)
        ],
        dtype=float,
    )
    base = np.maximum(base, 0.0)
    clim = np.asarray(component_factors, dtype=float) @ hazard_intensities  # (years,)
    # [H2] Carry the Gaussian noise in a LATENT (unclamped) path so negative draws
    # genuinely offset positive ones. The previous per-step ``max(prev, …)`` floor
    # truncated every negative draw to zero effect, turning supposedly mean-zero
    # noise into a one-sided upward ratchet that compounded over the horizon.
    noise = rng.normal(0.0, base_noise_sigma, size=years)
    latent = c0 + np.cumsum(base + clim + noise)
    # Report a monotonic non-decreasing path (condition cannot improve without
    # renewal), floored at the observed condition and capped at failure (5.0). The
    # clamp is applied to the latent path at reporting time, not per step.
    return np.clip(np.maximum.accumulate(np.maximum(latent, c0)), 1.0, 5.0)
