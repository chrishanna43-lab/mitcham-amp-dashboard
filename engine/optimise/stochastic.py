"""Two-stage stochastic renewal optimisation with a CVaR risk term.

The first-stage decision ``x[i, t] in [0, 1]`` schedules renewal of decision
unit ``i`` in year ``t`` — read as the optimal renewal *intensity* (how much of
unit ``i``'s renewal is brought to year ``t``), with at most one full renewal
per unit over the horizon and a per-year capex budget. Second-stage cost is the
criticality-weighted level-of-service breach a unit incurs in every year it
remains un-renewed, summed across years, plus a small end-of-horizon residual
liability. The objective minimises the expected second-stage cost across the
sampled climate realisations plus ``lam`` times its CVaR at level ``alpha`` —
trading mean cost against tail (worst-realisation) exposure.

Default is the **convex relaxation** (continuous ``x``): with CVaR it is a
linear/conic program that Mosek solves to a clean optimum in well under a
second and that scales to component granularity. The renewal year for the works
programme is read off the optimal policy as the modal year. Set ``relax=False``
for an exact binary MILP (slower; opt-in).
"""
from __future__ import annotations

import cvxpy as cp
import numpy as np


def solve_stochastic(
    cost: np.ndarray,        # (n, T) expected replacement cost per unit (shared, broadcast over years)
    breach: np.ndarray,      # (S, n, T) criticality-weighted breach units per scenario if NOT renewed by t
    budget: np.ndarray,      # (T,)
    lam: float = 1.0,        # CVaR weight (0 => risk-neutral)
    w_service: float = 1.0,
    w_liability: float = 1e-3,
    alpha: float = 0.95,
    solver: str = "MOSEK",
    relax: bool = True,
    mip_gap: float = 0.01,
    time_limit: float = 120.0,
    must_renew_idx: list[int] | None = None,
    must_renew_floor: float = 1.0,
):
    """Solve the two-stage stochastic renewal program with a CVaR risk term.

    Returns ``(x (n,T) in [0,1], objective, status, realised_cost (S,))`` where
    ``realised_cost[s]`` is the second-stage cost the chosen policy incurs under
    sampled scenario ``s`` — the basis for a residual-risk distribution.

    ``must_renew_idx``: row indices that must be renewed in the horizon. Each
    forced row's renewal intensity is bounded below by ``must_renew_floor``; the
    optimiser still chooses the year(s). Default floor of 1.0 pins binary
    decisions to a single year; relaxation callers typically pass the renewal-
    intensity threshold the post-solve interpretation uses (so the forced row
    clears the "renewed" bar without demanding more spend than feasible). If
    the budget cannot accommodate the forced set the solver returns infeasible.
    """
    n, T = cost.shape
    S = breach.shape[0]
    if S == 0:
        return np.zeros((n, T)), 0.0, "empty", np.zeros(0)

    if relax:
        x = cp.Variable((n, T), nonneg=True)
        cons = [x <= 1]
    else:
        x = cp.Variable((n, T), boolean=True)
        cons = []

    cons.append(cp.sum(x, axis=1) <= 1)                       # at most one renewal per unit
    for i in must_renew_idx or ():
        cons.append(cp.sum(x[i, :]) >= must_renew_floor)       # force unit i to clear the renewal threshold
    cons.append(cp.sum(cp.multiply(cost, x), axis=0) <= budget)  # per-year capex budget (vectorised over T)

    cum = cp.cumsum(x, axis=1)                                # cumulative renewal by year
    not_renewed = 1 - cum                                     # (n, T), in [0, 1]

    # Vectorised per-scenario service cost: B[s] @ vec(not_renewed) (C-order on both sides).
    nr_flat = cp.reshape(not_renewed, (n * T,), order="C")
    weighted = (w_service * breach).reshape(S, n * T)         # numpy C-order matches reshape order
    scen_service = weighted @ nr_flat                         # (S,)
    liability = w_liability * (cost[:, -1] @ not_renewed[:, T - 1])
    scen_cost = scen_service + liability                      # (S,)

    expected = cp.sum(scen_cost) / S
    eta = cp.Variable()                                       # CVaR (Rockafellar–Uryasev)
    z = cp.Variable(S, nonneg=True)
    cons += [z >= scen_cost - eta]
    cvar = eta + (1.0 / ((1.0 - alpha) * S)) * cp.sum(z)

    prob = cp.Problem(cp.Minimize(expected + lam * cvar), cons)
    _solve(prob, solver, relax, mip_gap, time_limit)

    xv = (
        np.zeros((n, T))
        if x.value is None
        else np.clip(np.asarray(x.value), 0.0, 1.0)
    )
    not_renewed_val = 1.0 - np.cumsum(xv, axis=1)
    realised = (
        w_service * (breach * not_renewed_val[None, :, :]).sum(axis=(1, 2))
        + w_liability * (cost[:, -1] @ not_renewed_val[:, T - 1])
    )
    obj_val = None if prob.value is None else float(prob.value)
    return xv, obj_val, prob.status, np.asarray(realised, dtype=float)


def _solve(prob, solver, relax, mip_gap, time_limit):
    """Solve with Mosek; fall back to HiGHS if Mosek is unavailable or errors."""
    want = solver.upper()
    if want == "MOSEK" and "MOSEK" in cp.installed_solvers():
        params = {} if relax else {
            "MSK_DPAR_MIO_TOL_REL_GAP": mip_gap,
            "MSK_DPAR_OPTIMIZER_MAX_TIME": time_limit,
        }
        try:
            prob.solve(solver=cp.MOSEK, mosek_params=params)
            return
        except cp.error.SolverError:
            pass
    prob.solve(solver=cp.HIGHS)
