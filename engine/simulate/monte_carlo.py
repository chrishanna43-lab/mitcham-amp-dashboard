"""Stage 5 — Monte Carlo deterioration / funding-gap simulation.

For each realisation, every asset's no-renewal condition trajectory is sampled
under each climate scenario via :func:`engine.simulate.samplers.sample_trajectory`,
and a single renewal cost is drawn per (realisation, asset). The per-year rows
(condition, renewal-need flag, cost-if-renewed) are written to ``mc_paths`` in
one bulk DataFrame insert per realisation to bound peak memory while keeping the
write path vectorised.
"""
from __future__ import annotations

import hashlib

import duckdb
import numpy as np
import pandas as pd

from engine.calibrate.costs import sample_cost
from engine.db import insert_dataframe
from engine.simulate.samplers import sample_trajectory

CURRENT_YEAR = 2026

# Hazard order is fixed across the engine: heat, flood, bushfire.
_HAZARDS: tuple[str, ...] = ("heat", "flood", "bushfire")

# A condition index at or above this threshold flags a renewal need. Set to the
# intervention level (4.0) — the same threshold the Stage 6 optimiser uses for a
# level-of-service breach — so the funding gap and the optimiser speak the same
# language. (5.0 is full failure; renewal is warranted at the 4.0 intervention.)
_RENEW_THRESHOLD = 4.0

_PATH_COLUMNS = [
    "realisation",
    "asset_id",
    "scenario",
    "year",
    "condition",
    "renew_need",
    "cost_if_renewed",
]


def _asset_rng(seed: int, realisation: int, asset_id: str) -> np.random.Generator:
    """A per-(seed, realisation, asset) RNG, independent of register composition.

    [H7] The simulation previously drew from a single global RNG consumed in
    asset-scan order, so inserting a What-If proposal row shifted every later
    asset's draws and changed the canonical assets' sampled breach years —
    contaminating the before/after diff. Seeding each asset's stream from a hash
    of its id makes a canonical asset sample identically with or without a
    proposal present, so a displacement is a genuine re-prioritisation.
    """
    digest = hashlib.blake2b(
        f"{seed}:{realisation}:{asset_id}".encode(), digest_size=8
    ).digest()
    return np.random.default_rng(int.from_bytes(digest, "little"))


def _fetch_assets(conn: duckdb.DuckDBPyConnection) -> list[dict]:
    """Return asset rows needed for simulation, ordered by ``asset_id``."""
    rows = conn.execute(
        """
        SELECT asset_id, asset_type, component, install_year,
               useful_life_years, condition, extent, grc, criticality_seed
        FROM assets
        ORDER BY asset_id
        """
    ).fetchall()
    cols = [d[0] for d in conn.description]
    return [dict(zip(cols, r)) for r in rows]


def _fetch_priors(conn: duckdb.DuckDBPyConnection) -> dict[str, dict[str, float]]:
    """Return ``{component: {"unit_rate": ..., "unit_rate_cv": ...}}``."""
    rows = conn.execute(
        "SELECT component, unit_rate, unit_rate_cv FROM priors"
    ).fetchall()
    return {
        component: {"unit_rate": float(unit_rate), "unit_rate_cv": float(cv)}
        for component, unit_rate, cv in rows
    }


def _fetch_climate_factors(
    conn: duckdb.DuckDBPyConnection,
) -> dict[str, dict[str, float]]:
    """Return ``{component: {hazard: k}}`` acceleration coefficients."""
    factors: dict[str, dict[str, float]] = {}
    for component, hazard, k in conn.execute(
        "SELECT component, hazard, k FROM climate_factors"
    ).fetchall():
        factors.setdefault(component, {})[hazard] = float(k)
    return factors


def _fetch_exposure(
    conn: duckdb.DuckDBPyConnection,
    asset_ids: list[str],
    scenarios: tuple[str, ...],
    horizon: int,
) -> np.ndarray:
    """Build a dense exposure array shape ``(n_scenarios, n_assets, n_hazards, horizon)``.

    Hazard axis order is heat/flood/bushfire; the year axis is indexed by
    ``year - CURRENT_YEAR`` over ``[CURRENT_YEAR, CURRENT_YEAR + horizon)``.
    Missing (scenario, asset, hazard, year) cells default to zero intensity.
    """
    asset_index = {aid: i for i, aid in enumerate(asset_ids)}
    scenario_index = {s: i for i, s in enumerate(scenarios)}
    hazard_index = {h: i for i, h in enumerate(_HAZARDS)}

    expo = np.zeros(
        (len(scenarios), len(asset_ids), len(_HAZARDS), horizon), dtype=float
    )

    rows = conn.execute(
        """
        SELECT scenario, asset_id, hazard, year, intensity
        FROM climate_exposure
        WHERE scenario IN ?
          AND year >= ?
          AND year < ?
        """,
        [list(scenarios), CURRENT_YEAR, CURRENT_YEAR + horizon],
    ).fetchall()

    for scenario, asset_id, hazard, year, intensity in rows:
        si = scenario_index.get(scenario)
        ai = asset_index.get(asset_id)
        hi = hazard_index.get(hazard)
        if si is None or ai is None or hi is None:
            continue
        t = int(year) - CURRENT_YEAR
        if 0 <= t < horizon:
            expo[si, ai, hi, t] = float(intensity)

    return expo


def run_monte_carlo(
    conn: duckdb.DuckDBPyConnection,
    n_realisations: int,
    horizon: int,
    scenarios: tuple[str, ...] = ("no_climate", "rcp45", "rcp85"),
    seed: int = 20260527,
) -> int:
    """Sample ``n_realisations`` no-renewal paths per asset/scenario into ``mc_paths``.

    Returns the total number of ``mc_paths`` rows written, which equals
    ``n_realisations * n_assets * len(scenarios) * horizon``.
    """
    assets = _fetch_assets(conn)
    asset_ids = [a["asset_id"] for a in assets]
    priors = _fetch_priors(conn)
    climate_factors = _fetch_climate_factors(conn)
    exposure = _fetch_exposure(conn, asset_ids, scenarios, horizon)

    # Per-asset constants used inside the realisation loop.
    age0 = np.array([CURRENT_YEAR - int(a["install_year"]) for a in assets], dtype=float)
    comp_factors = [
        np.array(
            [climate_factors.get(a["component"], {}).get(h, 0.0) for h in _HAZARDS],
            dtype=float,
        )
        for a in assets
    ]

    conn.execute("DELETE FROM mc_paths")

    years = [CURRENT_YEAR + t for t in range(horizon)]

    for r in range(n_realisations):
        realisations: list[int] = []
        out_asset_ids: list[str] = []
        out_scenarios: list[str] = []
        out_years: list[int] = []
        conditions: list[float] = []
        renew_needs: list[bool] = []
        costs: list[float] = []

        for i, asset in enumerate(assets):
            asset_id = asset["asset_id"]
            # [H7] Per-asset RNG so draws are independent of register composition.
            arng = _asset_rng(seed, r, asset_id)
            # Replacement cost is the asset's gross replacement cost (anchored to
            # the published portfolio total), varied by the component cost CV.
            # The prior unit_rate feeds the deterioration model, not the costing.
            prior = priors.get(asset["component"], {})
            cv = prior.get("unit_rate_cv", 0.2)
            grc = float(asset["grc"]) if asset["grc"] is not None else 0.0
            cost = sample_cost(grc, cv, 1.0, arng)

            initial_condition = float(asset["condition"])
            useful_life = float(asset["useful_life_years"])

            for si, scenario in enumerate(scenarios):
                traj = sample_trajectory(
                    initial_condition,
                    age0[i],
                    useful_life,
                    horizon,
                    exposure[si, i],
                    comp_factors[i],
                    arng,
                )
                for t in range(horizon):
                    cond = float(traj[t])
                    realisations.append(r)
                    out_asset_ids.append(asset_id)
                    out_scenarios.append(scenario)
                    out_years.append(years[t])
                    conditions.append(cond)
                    renew_needs.append(bool(cond >= _RENEW_THRESHOLD))
                    costs.append(cost)

        df = pd.DataFrame(
            {
                "realisation": realisations,
                "asset_id": out_asset_ids,
                "scenario": out_scenarios,
                "year": out_years,
                "condition": conditions,
                "renew_need": renew_needs,
                "cost_if_renewed": costs,
            },
            columns=_PATH_COLUMNS,
        )
        insert_dataframe(conn, "mc_paths", df)

    return conn.execute("SELECT count(*) FROM mc_paths").fetchone()[0]
