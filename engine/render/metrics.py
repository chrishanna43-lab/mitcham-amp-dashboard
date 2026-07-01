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

# SA local-government asset-management target bands (financial indicators panel).
# ARFR is the SA-mandated I&AMP-based indicator (FSIP No. 9, Nov 2024; rolling
# average target 1.00). ACR is one-sided (high = young stock, healthy). ASR is
# the national depreciation-based cross-check. Backlog is operational, not an
# SA-regulated ratio.
ARFR_BAND = (0.80, 1.20)   # FSIP No. 9 (Nov 2024); rolling avg target 1.00
ACR_BAND = (0.40, 0.80)
ASR_BAND = (0.90, 1.10)
BACKLOG_BAND = (0.05, 0.15)  # illustrative; backlog is not an SA-regulated ratio


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
    n_real = conn.execute(
        "SELECT COUNT(DISTINCT realisation) FROM mc_paths WHERE scenario = ?",
        [scenario],
    ).fetchone()[0] or 1
    demand = np.array([float(d) for _r, d in rows], dtype=float)
    # [H4] Realisations with no in-horizon breach produce no row; pad them back as
    # zero demand before taking quantiles (mirrors cumulative_need's zero-pad).
    # Dropping them silently biases demand/gap high on a benign portfolio and makes
    # this headline ARFR diverge from the zero-padded cumulative_need/gap_trajectory.
    if demand.size < n_real:
        demand = np.concatenate([demand, np.zeros(n_real - demand.size)])
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


# IPWEA-style numeric condition rating: 1 = excellent/near-new … 5 = failed
# (renewal backlog). The numeric face of the A–F band above (A→1 … F→5).
GRADE_NUMBER: dict[str, int] = {"A": 1, "B": 2, "C": 3, "D": 4, "F": 5}


def condition_grade_number(condition: float) -> int:
    """Condition as a 1–5 rating (1 = excellent … 5 = failed)."""
    return GRADE_NUMBER[condition_grade(condition)]


def report_card(conn: duckdb.DuckDBPyConnection, council: str = "mitcham") -> dict:
    """Portfolio and per-building A–F service grades from current condition.

    A building's condition is taken as its worst component (MAX); its value is
    the sum of component replacement costs (grc). Returns the value-weighted
    portfolio grade, the count and value of buildings in each band, and the
    per-building rows (building_id, name, condition, value, grade) ordered by
    value. The grade is a present-day snapshot, not a horizon projection.

    ``council`` (default ``'mitcham'``) filters the population so the report
    card shares ``asset_economics``' population (same council). Pass ``None`` to
    aggregate across every council. On the Mitcham-only canonical DB the default
    is identical to no filter.
    """
    where = "WHERE council = ?" if council is not None else ""
    params = [council] if council is not None else []
    rows = conn.execute(
        f"""
        SELECT split_part(asset_id, '-', 1) AS building_id,
               COALESCE(MAX(name), split_part(asset_id, '-', 1)) AS name,
               MAX(condition) AS condition,
               SUM(grc) AS value
        FROM assets
        {where}
        GROUP BY building_id
        ORDER BY value DESC
        """,
        params,
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


def asset_economics(conn: duckdb.DuckDBPyConnection) -> dict:
    """Gross replacement cost, straight-line written-down value, and annual
    straight-line depreciation across the portfolio.

    DRC = GRC * clamp((useful_life - age) / useful_life, 0, 1), age = CURRENT_YEAR
    - install_year. Annual depreciation = sum(GRC / useful_life). Straight-line is
    a documented assumption (S.5); calibratable if the council shares valuations.

    Note the deliberate asymmetry: an over-age asset charges full GRC/useful_life
    to annual depreciation but clamps to DRC=0 (you cannot consume below zero) —
    correct straight-line convention; covered by a fixture test (S.6).
    """
    row = conn.execute(
        """
        SELECT
          SUM(grc) AS grc_total,
          SUM(grc / NULLIF(useful_life_years, 0)) AS annual_depreciation,
          SUM(grc * GREATEST(0.0, LEAST(1.0,
              (useful_life_years - (? - install_year))
              / CAST(NULLIF(useful_life_years, 0) AS DOUBLE)))) AS drc_total
        FROM assets WHERE council = 'mitcham'
        """,
        [CURRENT_YEAR],
    ).fetchone()
    grc, dep, drc = (float(x or 0.0) for x in row)
    return {"grc_total": grc, "annual_depreciation": dep, "drc_total": drc}


def marginal_economics(new_rows: list[dict], current_year: int = 2026) -> dict:
    """Exact Channel-A deltas from proposal component rows (S.2.2).

    Mirrors ``asset_economics`` per component (NOT a single blended life —
    [FIX-B2] that form is forbidden because per-component useful lives differ):
    ``DRC = grc * clamp((u - age)/u, 0, 1)``, ``age = current_year - install_year``,
    annual depreciation ``= grc / u``. Returns ``{'d_grc', 'd_drc',
    'd_depreciation', 'd_f_value'}`` summed over the rows. The F-value
    contribution counts ``grc`` only where ``condition >= 4.0`` (the F-grade band,
    the renewal-backlog floor).

    ``new_rows`` items may be dicts or pydantic ``Asset`` models (attribute or
    key access both work).
    """
    def _get(r, key):
        return r[key] if isinstance(r, dict) else getattr(r, key)

    d_grc = d_drc = d_dep = d_fval = 0.0
    for r in new_rows:
        u = float(_get(r, "useful_life_years"))
        g = float(_get(r, "grc"))
        age = current_year - int(_get(r, "install_year"))
        f = max(0.0, min(1.0, (u - age) / u)) if u else 0.0
        d_grc += g
        d_drc += g * f
        d_dep += (g / u) if u else 0.0
        if float(_get(r, "condition")) >= 4.0:
            d_fval += g
    return {
        "d_grc": d_grc,
        "d_drc": d_drc,
        "d_depreciation": d_dep,
        "d_f_value": d_fval,
    }


def cumulative_need(
    conn: duckdb.DuckDBPyConnection,
    scenario: str,
    horizon: int,
    quantile: float = 0.50,
) -> list[float]:
    """Cumulative renewal need to each year at ``quantile`` across realisations.

    Each asset's renewal cost is counted once, in its first breach year. Mirrors
    ``gap_trajectory``'s existing logic exactly, generalised to any quantile —
    including the zero-pad-to-``n_real`` step (realisations with no breach-by-t
    contribute zero), without which every low-year high quantile is biased high.
    """
    import pandas as pd

    rows = conn.execute(
        """
        SELECT realisation, asset_id, MIN(year) AS breach_year, MAX(cost_if_renewed) AS cost
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
    out: list[float] = []
    for t in years:
        if df.empty:
            out.append(0.0)
            continue
        sums = df[df["breach_year"] <= t].groupby("realisation")["cost"].sum().to_numpy()
        if len(sums) < n_real:  # realisations with no breach-by-t = 0
            sums = np.concatenate([sums, np.zeros(n_real - len(sums))])
        out.append(float(np.quantile(sums, quantile)))
    return out


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

    Delegates the cumulative-need series to :func:`cumulative_need` at the p50
    quantile (provably identical to the previous inline computation); the return
    shape is unchanged — existing gap-chart callers depend on it.
    """
    years = list(range(CURRENT_YEAR, CURRENT_YEAR + horizon))
    demand_p50 = cumulative_need(conn, scenario, horizon, 0.50)
    capacity = [float(annual_budget) * (i + 1) for i in range(horizon)]
    return {"years": years, "demand_p50": demand_p50, "capacity": capacity}


def band_status(value, band, warn_pad: float = 0.10) -> str:
    """Two-sided band (ARFR, ASR): in-band good, just-outside warn, else bad.

    Returns a RAG chip token: ``"good" | "warn" | "bad" | ""`` (empty when the
    value is ``None``).
    """
    if value is None:
        return ""
    lo, hi = band
    if lo <= value <= hi:
        return "good"
    if lo - warn_pad <= value <= hi + warn_pad:
        return "warn"
    return "bad"


def acr_status(value) -> str:
    """One-sided ACR status: high is healthy (young stock), so high is never bad.

    green >= 0.40, amber 0.30–0.40, red < 0.30. Returns ``""`` when ``None``.
    """
    if value is None:
        return ""
    if value >= 0.40:
        return "good"
    if value >= 0.30:
        return "warn"
    return "bad"


def backlog_status(value) -> str:
    """Renewal-backlog status: lower is better. green < 5%, amber 5–15%, red > 15%.

    Returns ``""`` when ``None``.
    """
    if value is None:
        return ""
    if value < 0.05:
        return "good"
    if value <= 0.15:
        return "warn"
    return "bad"


def financial_indicators(
    conn: duckdb.DuckDBPyConnection,
    scenario: str,
    horizon: int,
    annual_budget: float,
) -> dict:
    """The four SA local-government asset financial ratios + their status inputs.

    Canonical arg order ``(conn, scenario, horizon, annual_budget)`` — matches
    ``unfunded_liability`` / ``gap_trajectory``. All call sites MUST pass
    ``horizon`` / ``annual_budget`` as keyword args (positional order silently
    swapped them in an earlier draft). Reuses ``unfunded_liability`` (capacity,
    demand_p50, gap_p50), ``gap_trajectory`` (the rolling-ARFR series),
    ``report_card`` (backlog + grade, council-filtered), ``asset_economics``, and
    the optimiser's scheduled renewal from ``opt_summary``.
    """
    ul = unfunded_liability(conn, scenario, horizon, annual_budget)
    gt = gap_trajectory(conn, scenario, horizon, annual_budget)
    econ = asset_economics(conn)
    rc = report_card(conn)  # council-filtered (shares asset_economics' population)

    capacity = ul["capacity"]
    demand = ul["demand_p50"] or 0.0

    # ARFR headline = funded capacity / modelled renewal need (need-based; the
    # I&AMP-proposed renewal a rigorous AMP would carry).
    arfr = capacity / demand if demand else None
    arfr_series = [
        (c / d if d else None) for c, d in zip(gt["capacity"], gt["demand_p50"])
    ]

    # ACR = written-down value / gross replacement cost.
    acr = econ["drc_total"] / econ["grc_total"] if econ["grc_total"] else None

    # ASR = optimiser-SCHEDULED renewal (avg annual) / annual depreciation. NOT
    # budget/depreciation (which is circular — capacity/horizon == annual_budget
    # exactly). Scheduled renewal is the planned WORKS the optimiser commits
    # within the horizon, from opt_summary.
    scheduled = conn.execute(
        "SELECT COALESCE(SUM(mean_cost), 0) FROM opt_summary "
        "WHERE scenario = 'stochastic' AND renew_year_p50 BETWEEN ? AND ?",
        [CURRENT_YEAR, CURRENT_YEAR + horizon - 1],
    ).fetchone()[0] or 0.0
    avg_annual_renewal = scheduled / horizon if horizon else 0.0
    asr = (
        avg_annual_renewal / econ["annual_depreciation"]
        if econ["annual_depreciation"] and avg_annual_renewal > 0
        else None
    )

    # Backlog ratio = F-grade value / total value (report_card, council-filtered
    # so it shares asset_economics' population).
    backlog = (
        rc["grade_value"]["F"] / rc["total_value"]
        if rc and rc["total_value"]
        else None
    )

    return {
        "arfr": arfr, "arfr_series": arfr_series, "arfr_years": gt["years"],
        "acr": acr, "asr": asr, "backlog": backlog,
        "depreciation": econ["annual_depreciation"],
        "scheduled_renewal": scheduled,
        "demand": demand, "capacity": capacity,
        # Surfaced so the What-If diffs all six cards off ONE callable:
        "gap_p50": ul["gap_p50"],
        "portfolio_grade": rc["portfolio_grade"] if rc else None,
    }


def arfr_compliance(
    conn: duckdb.DuckDBPyConnection,
    scenario: str,
    horizon: int,
    annual_budget: float,
    band: tuple[float, float] = ARFR_BAND,
) -> dict:
    """Rolling ARFR over the horizon at p50 and p95 renewal need.

    Rolling ARFR(t) = cumulative funded capacity(t) / cumulative need(t). Funded
    capacity is linear (annual_budget * t); need is non-decreasing — so the ratio
    tends to fall over time, and under RCP 8.5 it can drop through the 80% floor
    in a specific year. Reporting p95 as well lets the compliance claim stand
    against the high-demand tail, not just the median.
    """
    years = list(range(CURRENT_YEAR, CURRENT_YEAR + horizon))
    cap = [float(annual_budget) * (i + 1) for i in range(horizon)]
    lo, hi = band

    def series(quantile: float) -> list:
        need = cumulative_need(conn, scenario, horizon, quantile)
        # Suppress early-year spikes: until cumulative need clears 1% of its
        # horizon total, the ratio is a meaningless 1000s-of-% artefact.
        floor = 0.01 * (need[-1] or 0.0)
        return [(c / d if d > floor else None) for c, d in zip(cap, need)]

    def first_breach(s: list):
        return next(
            (y for y, v in zip(years, s) if v is not None and not (lo <= v <= hi)),
            None,
        )

    a50, a95 = series(0.50), series(0.95)

    def in_band(s: list) -> int:
        return sum(1 for v in s if v is not None and lo <= v <= hi)

    return {
        "years": years, "arfr_p50": a50, "arfr_p95": a95, "band": band,
        "breach_year_p50": first_breach(a50), "breach_year_p95": first_breach(a95),
        "years_in_band_p50": in_band(a50), "years_in_band_p95": in_band(a95),
        "n_years": horizon,
    }


def minimum_sustainable_budget(
    conn: duckdb.DuckDBPyConnection,
    scenario: str,
    horizon: int,
    floor: float = 0.80,
) -> dict:
    """Flat annual budget that keeps the rolling ARFR within band over the horizon.

    floor_budget  — smallest flat budget holding rolling ARFR >= floor (0.80) at
                    EVERY year: floor * max_t(cumulative_need(t) / t).
    target_budget — budget averaging ARFR = 1.00 over the horizon: total need / horizon.
    Both at p50 and p95 demand. The floor protects the worst single year; the
    target hits the horizon average. Neither dominates in general: for linear
    demand floor_budget = 0.80 * target_budget; the two cross only when demand
    front-loads hard (peak yearly need-rate >= 1.25x the horizon average).
    """
    out: dict = {}
    for q, tag in ((0.50, "p50"), (0.95, "p95")):
        need = cumulative_need(conn, scenario, horizon, quantile=q)
        if not need or need[-1] <= 0:
            out[tag] = {"floor_budget": 0.0, "target_budget": 0.0}
            continue
        floor_budget = floor * max(d / (i + 1) for i, d in enumerate(need) if d > 0)
        out[tag] = {"floor_budget": floor_budget, "target_budget": need[-1] / horizon}
    return out


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
