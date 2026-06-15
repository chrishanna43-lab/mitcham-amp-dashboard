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
    traj = np.empty(years, dtype=float)
    prev = float(initial_condition)
    age = float(initial_age)
    for t in range(years):
        delta_base = max(
            0.0,
            condition_at_age(age + 1.0, useful_life)
            - condition_at_age(age, useful_life),
        )
        delta_clim = float(np.dot(component_factors, hazard_intensities[:, t]))
        noise = rng.normal(0.0, base_noise_sigma)
        prev = min(5.0, max(prev, prev + delta_base + delta_clim + noise))
        traj[t] = prev
        age += 1.0
    return traj
