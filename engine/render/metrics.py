"""Funding-gap metrics — the dollar headline for the dashboard.

The Stage 6 optimiser objective is a residual *service-risk index*, not dollars,
so the council-facing headline (the unfunded renewal liability) is computed here
directly from Monte Carlo renewal demand versus budgeted capacity.
"""
from __future__ import annotations

import duckdb
import numpy as np

CURRENT_YEAR = 2026
INTERVENTION = 4.0


def unfunded_liability(
    conn: duckdb.DuckDBPyConnection,
    scenario: str,
    horizon: int,
    annual_budget: float,
) -> dict:
    """25-year renewal demand and unfunded gap, distributed across realisations.

    Per realisation: an asset needs renewal if its condition reaches the 4.0
    intervention level in any year within the horizon; its renewal cost
    (cost_if_renewed, constant per realisation/asset) is counted once. Renewal
    demand = sum of those costs. Unfunded gap = max(0, demand - annual_budget*horizon).
    Returns p05/p50/p95 of both demand and gap across realisations, plus the
    funded capacity.
    """
    rows = conn.execute(
        """
        WITH needed AS (
          SELECT realisation, asset_id, MAX(cost_if_renewed) AS cost
          FROM mc_paths
          WHERE scenario = ? AND year < ? AND condition >= ?
          GROUP BY realisation, asset_id
        )
        SELECT realisation, SUM(cost) AS demand
        FROM needed GROUP BY realisation
        """,
        [scenario, CURRENT_YEAR + horizon, INTERVENTION],
    ).fetchall()
    demand = np.array([float(d) for _r, d in rows], dtype=float)
    if demand.size == 0:
        demand = np.array([0.0])
    capacity = float(annual_budget) * horizon
    gap = np.maximum(0.0, demand - capacity)
    q = lambda a, p: float(np.quantile(a, p))  # noqa: E731
    return {
        "scenario": scenario,
        "capacity": capacity,
        "demand_p05": q(demand, 0.05), "demand_p50": q(demand, 0.50), "demand_p95": q(demand, 0.95),
        "gap_p05": q(gap, 0.05), "gap_p50": q(gap, 0.50), "gap_p95": q(gap, 0.95),
    }


# A–F service grades (ASCE report-card convention — no E band). Bands are read
# off the 1 (new) – 5 (failed) condition index; the F band starts at the 4.0
# intervention level, so an F asset is in renewal backlog.
GRADE_ORDER: tuple[str, ...] = ("A", "B", "C", "D", "F")
GRADE_BANDS: tuple[tuple[float, str], ...] = (
    (1.5, "A"),
    (2.5, "B"),
    (3.3, "C"),
    (4.0, "D"),
)


def condition_grade(condition: float) -> str:
    """Map a condition index to an A–F service grade (F at/above intervention)."""
    for ceiling, grade in GRADE_BANDS:
        if condition < ceiling:
            return grade
    return "F"


def report_card(conn: duckdb.DuckDBPyConnection) -> dict:
    """Portfolio and per-building A–F service grades from current condition.

    A building's condition is taken as its worst component (MAX); its value is
    the sum of component replacement costs (grc). Returns the value-weighted
    portfolio grade, the count and value of buildings in each band, and the
    per-building rows (building_id, name, condition, value, grade) ordered by
    value. The grade is a present-day snapshot, not a horizon projection.
    """
    rows = conn.execute(
        """
        SELECT split_part(asset_id, '-', 1) AS building_id,
               COALESCE(MAX(name), split_part(asset_id, '-', 1)) AS name,
               MAX(condition) AS condition,
               SUM(grc) AS value
        FROM assets
        GROUP BY building_id
        ORDER BY value DESC
        """
    ).fetchall()

    grade_count = {g: 0 for g in GRADE_ORDER}
    grade_value = {g: 0.0 for g in GRADE_ORDER}
    buildings: list[dict] = []
    total_value = 0.0
    weighted_condition = 0.0
    for building_id, name, condition, value in rows:
        condition = float(condition)
        value = float(value or 0.0)
        grade = condition_grade(condition)
        grade_count[grade] += 1
        grade_value[grade] += value
        total_value += value
        weighted_condition += condition * value
        buildings.append(
            {"building_id": building_id, "name": name,
             "condition": condition, "value": value, "grade": grade}
        )

    portfolio_condition = weighted_condition / total_value if total_value else 0.0
    return {
        "portfolio_grade": condition_grade(portfolio_condition),
        "portfolio_condition": portfolio_condition,
        "grade_count": grade_count,
        "grade_value": grade_value,
        "n_buildings": len(buildings),
        "total_value": total_value,
        "buildings": buildings,
    }


def gap_trajectory(
    conn: duckdb.DuckDBPyConnection,
    scenario: str,
    horizon: int,
    annual_budget: float,
) -> dict:
    """Cumulative renewal demand vs funded capacity, year by year.

    For each realisation an asset's renewal cost is counted once, in the first
    year its condition reaches the intervention level. Cumulative demand to year
    ``t`` is the running total of those costs; the p50 across realisations is the
    'do nothing' bill that accumulates. Capacity is ``annual_budget`` cumulated.
    The widening gap between the two is the cost of deferring renewal. Returns
    ``years``, ``demand_p50`` (non-decreasing), and ``capacity``.
    """
    import pandas as pd

    rows = conn.execute(
        """
        SELECT realisation, asset_id,
               MIN(year) AS breach_year, MAX(cost_if_renewed) AS cost
        FROM mc_paths
        WHERE scenario = ? AND condition >= ? AND year < ?
        GROUP BY realisation, asset_id
        """,
        [scenario, INTERVENTION, CURRENT_YEAR + horizon],
    ).fetchall()
    n_real = conn.execute(
        "SELECT COUNT(DISTINCT realisation) FROM mc_paths WHERE scenario = ?",
        [scenario],
    ).fetchone()[0] or 1

    years = list(range(CURRENT_YEAR, CURRENT_YEAR + horizon))
    df = pd.DataFrame(rows, columns=["realisation", "asset_id", "breach_year", "cost"])
    demand_p50: list[float] = []
    for t in years:
        if df.empty:
            demand_p50.append(0.0)
            continue
        sums = df[df["breach_year"] <= t].groupby("realisation")["cost"].sum().to_numpy()
        if len(sums) < n_real:  # realisations with no breach-by-t contribute zero
            sums = np.concatenate([sums, np.zeros(n_real - len(sums))])
        demand_p50.append(float(np.quantile(sums, 0.50)))

    capacity = [float(annual_budget) * (i + 1) for i in range(horizon)]
    return {"years": years, "demand_p50": demand_p50, "capacity": capacity}


def climate_exposure_value(
    conn: duckdb.DuckDBPyConnection,
    scenario: str,
    horizon: int,
) -> dict:
    """Replacement value sitting under rising climate hazard, by year and hazard.

    For each building and year a hazard-exposure index in [0, 1] is read from
    ``climate_exposure`` (mean component intensity per hazard); the climate-
    exposed value is the building's replacement value times that index. The
    ``by_hazard`` series sum exposure across hazards (to show the hazard mix);
    the ``total`` series uses the worst hazard per building (max), so it never
    exceeds portfolio value. This is an exposure measure — value under hazard —
    not an actuarial loss. Returns ``years``, ``by_hazard`` and ``total``.
    """
    import pandas as pd

    rows = conn.execute(
        """
        SELECT split_part(ce.asset_id, '-', 1) AS bid, ce.hazard, ce.year,
               AVG(ce.intensity) AS intensity
        FROM climate_exposure ce
        WHERE ce.scenario = ? AND ce.year < ?
        GROUP BY bid, ce.hazard, ce.year
        """,
        [scenario, CURRENT_YEAR + horizon],
    ).fetchall()
    if not rows:
        return {"years": [], "by_hazard": {}, "total": []}

    expo = pd.DataFrame(rows, columns=["bid", "hazard", "year", "intensity"])
    bld = pd.DataFrame(
        conn.execute(
            "SELECT split_part(asset_id, '-', 1) AS bid, SUM(grc) AS value "
            "FROM assets WHERE council = 'mitcham' GROUP BY bid"
        ).fetchall(),
        columns=["bid", "value"],
    )
    m = expo.merge(bld, on="bid", how="inner")
    m["cev"] = m["value"] * m["intensity"]
    years = sorted(int(y) for y in m["year"].unique())
    hazards = sorted(str(h) for h in m["hazard"].unique())
    by_hazard = {
        h: [float(m[(m["hazard"] == h) & (m["year"] == y)]["cev"].sum()) for y in years]
        for h in hazards
    }
    comb = (
        m.groupby(["bid", "year"])
        .agg(value=("value", "first"), max_int=("intensity", "max"))
        .reset_index()
    )
    comb["cev"] = comb["value"] * comb["max_int"]
    total = [float(comb[comb["year"] == y]["cev"].sum()) for y in years]
    return {"years": years, "by_hazard": by_hazard, "total": total}
