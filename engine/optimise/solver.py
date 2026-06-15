"""Assemble optimiser inputs from DuckDB, solve, and persist the schedule.

Pulls per-component expected replacement costs and a sample of Monte-Carlo
(realisation, scenario) condition paths out of ``mc_paths``, aggregates them to
the chosen decision granularity (whole buildings by default, individual
components on request), converts them into the ``cost``/``breach``/``budget``
arrays the stochastic MILP expects, solves once CVaR-aware and once risk-neutral
(to flag robust renewals), then writes the chosen schedule to ``opt_summary``
and the per-scenario realised-cost distribution to ``opt_solutions``.

Granularity. Councils fund capital at the facility (building) level, so the
default decision unit is the whole building (~215 units → ~5,375 binaries,
solves to true optimal in seconds). Component-level scheduling (~37,600
binaries) is available via ``granularity="component"`` but is slower; the
component-level condition, climate and Monte-Carlo modelling feed both.
"""
from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from engine.db import insert_dataframe
from engine.optimise.stochastic import solve_stochastic

CURRENT_YEAR = 2026

# Condition index above which a component breaches its level-of-service target
# (the intervention level — matches the Stage 5 renewal-need threshold).
_BREACH_THRESHOLD = 4.0

_PORTFOLIO_ID = "__portfolio__"

_SUMMARY_COLUMNS = [
    "asset_id",
    "scenario",
    "renew_year_p50",
    "renew_share",
    "robust",
    "mean_cost",
]

_SOLUTION_COLUMNS = [
    "realisation",
    "asset_id",
    "scenario",
    "renew_year",
    "cost",
]


def _fetch_components(
    conn: duckdb.DuckDBPyConnection,
) -> tuple[list[str], np.ndarray]:
    """Return (component asset_ids ordered, criticality array) aligned by index."""
    rows = conn.execute(
        "SELECT asset_id, criticality_seed FROM assets ORDER BY asset_id"
    ).fetchall()
    asset_ids = [r[0] for r in rows]
    criticality = np.array(
        [float(r[1]) if r[1] is not None else 1.0 for r in rows], dtype=float
    )
    return asset_ids, criticality


def _fetch_mean_cost(
    conn: duckdb.DuckDBPyConnection, asset_ids: list[str]
) -> np.ndarray:
    """Per-component expected replacement cost (mean of ``cost_if_renewed``)."""
    rows = conn.execute(
        "SELECT asset_id, AVG(cost_if_renewed) FROM mc_paths GROUP BY asset_id"
    ).fetchall()
    lookup = {aid: float(c) for aid, c in rows}
    return np.array([lookup.get(aid, 0.0) for aid in asset_ids], dtype=float)


def _sample_pairs(
    conn: duckdb.DuckDBPyConnection, n_scenarios: int, rng: np.random.Generator
) -> list[tuple[int, str]]:
    """Sample up to ``n_scenarios`` distinct (realisation, scenario) pairs."""
    pairs = conn.execute(
        "SELECT DISTINCT realisation, scenario FROM mc_paths "
        "ORDER BY realisation, scenario"
    ).fetchall()
    pairs = [(int(r), str(s)) for r, s in pairs]
    if len(pairs) <= n_scenarios:
        return pairs
    idx = rng.choice(len(pairs), size=n_scenarios, replace=False)
    idx.sort()
    return [pairs[i] for i in idx]


def _build_component_breach(
    conn: duckdb.DuckDBPyConnection,
    asset_ids: list[str],
    weight: np.ndarray,
    pairs: list[tuple[int, str]],
    horizon: int,
) -> np.ndarray:
    """Build the (S, n_components, horizon) weighted breach tensor.

    ``breach[s, i, t] = weight_i * max(0, condition - 4.0)`` for the sampled pair
    ``s`` at component ``i`` and year ``CURRENT_YEAR + t``; missing cells 0. The
    per-component ``weight`` carries criticality and (when value-weighted) the
    replacement value, so the optimiser minimises value-at-risk rather than a
    raw count of breaches — surfacing high-value facilities.
    """
    n = len(asset_ids)
    asset_index = {aid: i for i, aid in enumerate(asset_ids)}
    breach = np.zeros((len(pairs), n, horizon), dtype=float)
    for s, (realisation, scenario) in enumerate(pairs):
        rows = conn.execute(
            """
            SELECT asset_id, year, condition
            FROM mc_paths
            WHERE realisation = ? AND scenario = ?
              AND year >= ? AND year < ?
            """,
            [realisation, scenario, CURRENT_YEAR, CURRENT_YEAR + horizon],
        ).fetchall()
        for asset_id, year, condition in rows:
            i = asset_index.get(asset_id)
            if i is None:
                continue
            t = int(year) - CURRENT_YEAR
            if 0 <= t < horizon:
                exceed = float(condition) - _BREACH_THRESHOLD
                if exceed > 0.0:
                    breach[s, i, t] = weight[i] * exceed
    return breach


def _unit_of(asset_id: str, granularity: str) -> str:
    """Map a component asset_id to its decision unit."""
    if granularity == "building":
        return asset_id.split("-")[0]
    return asset_id


def _aggregate_to_units(
    asset_ids: list[str],
    comp_cost: np.ndarray,
    comp_breach: np.ndarray,
    granularity: str,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Aggregate component cost + breach up to decision units.

    Returns (unit_ids ordered, unit_cost (m,), unit_breach (S, m, horizon)).
    A unit's cost is the sum of its components' expected costs (renewing the
    unit renews all its components); its breach is the sum of component breaches.
    """
    unit_ids_ordered: list[str] = []
    unit_index: dict[str, int] = {}
    for aid in asset_ids:
        u = _unit_of(aid, granularity)
        if u not in unit_index:
            unit_index[u] = len(unit_ids_ordered)
            unit_ids_ordered.append(u)

    m = len(unit_ids_ordered)
    s_count, _, horizon = comp_breach.shape
    unit_cost = np.zeros(m, dtype=float)
    unit_breach = np.zeros((s_count, m, horizon), dtype=float)
    for ci, aid in enumerate(asset_ids):
        u = unit_index[_unit_of(aid, granularity)]
        unit_cost[u] += comp_cost[ci]
        unit_breach[:, u, :] += comp_breach[:, ci, :]
    return unit_ids_ordered, unit_cost, unit_breach


# A unit counts as scheduled for renewal when its total renewal intensity over
# the horizon reaches this fraction (the policy may be continuous).
_RENEW_INTENSITY = 0.5


def _renew_years(x: np.ndarray) -> list[int | None]:
    """Map each unit row of the renewal policy to its modal renewal year.

    Reads the optimal (possibly continuous) policy: a unit is scheduled when its
    total renewal intensity reaches ``_RENEW_INTENSITY``; its renewal year is the
    modal (argmax-intensity) year.
    """
    years: list[int | None] = []
    for row in x:
        if row.sum() >= _RENEW_INTENSITY:
            years.append(CURRENT_YEAR + int(np.argmax(row)))
        else:
            years.append(None)
    return years


def solve_all(
    conn: duckdb.DuckDBPyConnection,
    horizon: int,
    annual_budget: float,
    n_scenarios: int = 60,
    lam: float = 1.0,
    solver: str = "MOSEK",
    seed: int = 0,
    granularity: str = "building",
    value_weighted: bool = True,
    engagement: dict[str, float] | None = None,
    engagement_strength: float = 1.0,
    label: str = "stochastic",
    write_solutions: bool = True,
    persist: bool = True,
    robust_pass: bool = True,
    must_renew: list[str] | None = None,
) -> dict:
    """Solve the stochastic renewal program from ``mc_paths`` and persist results.

    ``value_weighted`` (default True): weight the service breach by replacement
    value as well as criticality, so the optimiser minimises value-at-risk and
    high-value facilities surface in the works programme. Set False for a pure
    criticality-weighted breach count.

    ``engagement`` (optional): a ``{building_id: 0..1 index}`` map of community /
    elected-member priority. When supplied, each component's breach weight is
    multiplied by its building's engagement index, so community- and politically
    salient facilities are surfaced — the engagement-weighted lens. The schedule
    is written to ``opt_summary`` under ``label`` (so the engineering-optimum and
    engagement-weighted programmes can coexist), and ``write_solutions`` gates the
    ``opt_solutions`` write so a second variant does not clobber the first.

    ``must_renew`` (optional): decision-unit ids (building ids at the default
    granularity) the optimiser must renew within the horizon. Unknown ids are
    dropped silently so a stale UI cache cannot crash the solve. The optimiser
    still chooses the renewal year for each forced unit; if the forced set
    exceeds the per-year budget the solver returns an infeasible status, which
    callers should surface to the user.
    """
    if granularity not in ("building", "component"):
        raise ValueError("granularity must be 'building' or 'component'")
    rng = np.random.default_rng(seed)

    comp_ids, comp_crit = _fetch_components(conn)
    comp_cost = _fetch_mean_cost(conn, comp_ids)
    pairs = _sample_pairs(conn, n_scenarios, rng)
    # Weight breach by criticality and (optionally) replacement value in $M so
    # the objective is value-at-risk, not a raw breach count.
    value_factor = (comp_cost / 1e6) if value_weighted else np.ones_like(comp_cost)
    if engagement is not None:
        idx = np.array(
            [engagement.get(aid.split("-")[0], 0.5) for aid in comp_ids], dtype=float
        )
        mean_idx = float(idx.mean()) if idx.size else 1.0
        rel = idx / mean_idx if mean_idx else np.ones_like(idx)
        # Blend by strength: 0 → no engagement effect (engineering optimum),
        # 1 → full community/elected weighting (relative to the mean).
        eng_factor = (1.0 - engagement_strength) + engagement_strength * rel
    else:
        eng_factor = np.ones_like(comp_cost)
    comp_weight = comp_crit * value_factor * eng_factor
    comp_breach = _build_component_breach(conn, comp_ids, comp_weight, pairs, horizon)

    unit_ids, unit_cost, unit_breach = _aggregate_to_units(
        comp_ids, comp_cost, comp_breach, granularity
    )
    n = len(unit_ids)
    cost = np.repeat(unit_cost[:, None], horizon, axis=1)  # shared across years
    budget = np.full(horizon, float(annual_budget))

    # Resolve forced-renew ids to row indices; unknown ids are dropped so a stale
    # UI cache cannot crash the solve. Same indices fed to both passes below.
    unit_index = {uid: i for i, uid in enumerate(unit_ids)}
    must_renew_idx = (
        [unit_index[u] for u in must_renew if u in unit_index]
        if must_renew else None
    )

    # CVaR-aware schedule and (optionally) a risk-neutral reference for robustness.
    # Forced rows must clear ``_RENEW_INTENSITY`` (the same threshold used to read
    # the relaxed optimal policy as "renewed"), not the strict full-renewal of 1.
    # On the continuous relaxation a building whose total cost exceeds the
    # ``budget × horizon`` cap can't sum to 1 but can still meet the renewal bar.
    x_cvar, obj, status, realised = solve_stochastic(
        cost, unit_breach, budget, lam=lam, solver=solver, time_limit=120.0,
        must_renew_idx=must_renew_idx, must_renew_floor=_RENEW_INTENSITY,
    )
    renewed_cvar = x_cvar.sum(axis=1) >= _RENEW_INTENSITY
    if robust_pass:
        x_rn, _, _, _ = solve_stochastic(
            cost, unit_breach, budget, lam=0.0, solver=solver, time_limit=120.0,
            must_renew_idx=must_renew_idx, must_renew_floor=_RENEW_INTENSITY,
        )
        robust = renewed_cvar & (x_rn.sum(axis=1) >= _RENEW_INTENSITY)
    else:
        robust = renewed_cvar
    renew_years = _renew_years(x_cvar)

    # Per-unit programme (always returned; used by the live engagement lens).
    units = [
        {
            "asset_id": unit_ids[i],
            "renewed": bool(renewed_cvar[i]),
            "renew_year": (int(renew_years[i]) if renew_years[i] is not None else None),
            "robust": bool(robust[i]),
            "mean_cost": float(unit_cost[i]),
        }
        for i in range(n)
    ]

    if persist:
        # opt_summary — one row per decision unit, under ``label``.
        summary = pd.DataFrame(
            {
                "asset_id": unit_ids,
                "scenario": [label] * n,
                "renew_year_p50": [
                    (int(y) if y is not None else None) for y in renew_years
                ],
                "renew_share": [1.0 if renewed_cvar[i] else 0.0 for i in range(n)],
                "robust": [bool(robust[i]) for i in range(n)],
                "mean_cost": [float(unit_cost[i]) for i in range(n)],
            },
            columns=_SUMMARY_COLUMNS,
        )
        summary["renew_year_p50"] = summary["renew_year_p50"].astype("Int64")
        conn.execute("DELETE FROM opt_summary WHERE scenario = ?", [label])
        insert_dataframe(conn, "opt_summary", summary)

        # opt_solutions — realised cost per sampled scenario (funding-gap dist).
        # Gated so a second (engagement) variant does not clobber the first.
        if write_solutions:
            solutions = pd.DataFrame(
                {
                    "realisation": list(range(len(pairs))),
                    "asset_id": [_PORTFOLIO_ID] * len(pairs),
                    "scenario": [climate for _r, climate in pairs],
                    "renew_year": pd.array([None] * len(pairs), dtype="Int64"),
                    "cost": [float(c) for c in realised],
                },
                columns=_SOLUTION_COLUMNS,
            )
            conn.execute("DELETE FROM opt_solutions")
            insert_dataframe(conn, "opt_solutions", solutions)

    return {
        "n_assets": n,
        "granularity": granularity,
        "n_renewed": int(renewed_cvar.sum()),
        "n_robust": int(robust.sum()),
        "units": units,
        "status": status,
        "objective": obj,
        "realised_p05": float(np.quantile(realised, 0.05)) if len(realised) else 0.0,
        "realised_p50": float(np.quantile(realised, 0.50)) if len(realised) else 0.0,
        "realised_p95": float(np.quantile(realised, 0.95)) if len(realised) else 0.0,
    }
