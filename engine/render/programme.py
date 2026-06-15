"""Works-programme views: spend-by-year, funded list, deferred list, building card.

These are read-only aggregations over the optimiser and Monte Carlo outputs that
the dashboard's THE PLAN and THE PORTFOLIO views need. No model changes — purely
shaping existing engine tables into council-facing artefacts:

* :func:`spend_by_year` — total committed renewal $ per year for a spend-vs-budget chart.
* :func:`funded_programme` — buildings the optimiser renews within the horizon, enriched
  with metadata (suburb, criticality, value) for the Gantt and enriched table.
* :func:`deferred_buildings` — buildings the optimiser does *not* fund in this horizon
  (or that have no decided renewal year), with their first-breach year, condition and
  criticality so the council can see what's been pushed.
* :func:`building_card` — every dimension a per-building drawer needs (metadata, current
  condition, decided renewal, condition + climate trajectories, optimal-timing analysis).
* :func:`force_into_programme` — what happens if the user forces deferred buildings into
  the programme: a real optimiser re-solve constrained to renew those bids, with a diff
  against the canonical schedule (came in / pushed out / shifted year).

The optimiser writes ``opt_summary`` with ``scenario = 'stochastic'`` (one solve over
the stochastic ensemble). The climate ``scenario`` argument here selects the Monte Carlo
paths and climate exposures to surface — it does *not* re-run the optimiser.
"""
from __future__ import annotations

import duckdb
import numpy as np

from engine.optimise.solver import solve_all
from engine.render.timing import optimal_timing

CURRENT_YEAR = 2026
INTERVENTION = 4.0


def spend_by_year(conn: duckdb.DuckDBPyConnection, horizon: int) -> dict:
    """Total optimiser-committed renewal $ per year over the horizon.

    Returns ``{"years": [...], "spend": [...]}`` aligned year-by-year, with zeros for
    years that have no funded renewals. Years outside the horizon are excluded.
    """
    rows = conn.execute(
        """
        SELECT renew_year_p50 AS year, SUM(mean_cost) AS spend
        FROM opt_summary
        WHERE scenario = 'stochastic' AND renew_year_p50 IS NOT NULL
          AND renew_year_p50 BETWEEN ? AND ?
        GROUP BY renew_year_p50
        ORDER BY renew_year_p50
        """,
        [CURRENT_YEAR, CURRENT_YEAR + horizon - 1],
    ).fetchall()
    years = list(range(CURRENT_YEAR, CURRENT_YEAR + horizon))
    spend_map = {int(y): float(s) for y, s in rows}
    return {"years": years, "spend": [spend_map.get(y, 0.0) for y in years]}


def funded_programme(conn: duckdb.DuckDBPyConnection, horizon: int) -> list[dict]:
    """Funded renewals within the horizon, enriched with building metadata."""
    rows = conn.execute(
        """
        WITH b AS (
            SELECT split_part(asset_id, '-', 1) AS bid,
                   ANY_VALUE(name) AS name, ANY_VALUE(suburb) AS suburb,
                   ANY_VALUE(asset_type) AS asset_type,
                   MAX(criticality_seed) AS criticality,
                   MAX(condition) AS condition_now,
                   SUM(grc) AS value
            FROM assets GROUP BY bid
        )
        SELECT o.asset_id AS bid, COALESCE(b.name, o.asset_id) AS building,
               b.suburb, b.asset_type, b.criticality, b.value, b.condition_now,
               o.renew_year_p50 AS renewal_year, o.renew_share,
               o.robust, o.mean_cost
        FROM opt_summary o
        LEFT JOIN b ON o.asset_id = b.bid
        WHERE o.scenario = 'stochastic' AND o.renew_year_p50 IS NOT NULL
          AND o.renew_year_p50 BETWEEN ? AND ?
        ORDER BY o.renew_year_p50, o.robust DESC NULLS LAST, o.mean_cost DESC
        """,
        [CURRENT_YEAR, CURRENT_YEAR + horizon - 1],
    ).fetchall()
    cols = ["bid", "building", "suburb", "asset_type", "criticality", "value",
            "condition_now", "renewal_year", "renew_share", "robust", "mean_cost"]
    return [dict(zip(cols, r)) for r in rows]


def deferred_buildings(
    conn: duckdb.DuckDBPyConnection, scenario: str, horizon: int
) -> list[dict]:
    """Buildings not in the funded programme within the horizon.

    A deferred building is one the optimiser has not committed to renew at any year
    in ``[CURRENT_YEAR, CURRENT_YEAR + horizon)``. For each, the first year its
    expected condition reaches the intervention threshold is reported so the council
    can see how soon the asset starts imposing service/risk costs.
    """
    rows = conn.execute(
        """
        WITH b AS (
            SELECT split_part(asset_id, '-', 1) AS bid,
                   ANY_VALUE(name) AS name, ANY_VALUE(suburb) AS suburb,
                   ANY_VALUE(asset_type) AS asset_type,
                   MAX(criticality_seed) AS criticality,
                   MAX(condition) AS condition_now,
                   SUM(grc) AS value
            FROM assets GROUP BY bid
        ),
        funded AS (
            SELECT asset_id AS bid FROM opt_summary
            WHERE scenario = 'stochastic' AND renew_year_p50 IS NOT NULL
              AND renew_year_p50 BETWEEN ? AND ?
        ),
        breach AS (
            SELECT split_part(asset_id, '-', 1) AS bid,
                   MIN(year) AS first_breach
            FROM mc_paths
            WHERE scenario = ? AND condition >= ? AND year < ?
            GROUP BY split_part(asset_id, '-', 1)
        )
        SELECT b.bid, b.name AS building, b.suburb, b.asset_type, b.criticality,
               b.value, b.condition_now, br.first_breach
        FROM b
        LEFT JOIN funded f ON f.bid = b.bid
        LEFT JOIN breach br ON br.bid = b.bid
        WHERE f.bid IS NULL
        ORDER BY b.value DESC NULLS LAST
        """,
        [CURRENT_YEAR, CURRENT_YEAR + horizon - 1,
         scenario, INTERVENTION, CURRENT_YEAR + horizon],
    ).fetchall()
    cols = ["bid", "building", "suburb", "asset_type", "criticality", "value",
            "condition_now", "first_breach"]
    return [
        {**dict(zip(cols, r)),
         "first_breach": int(r[-1]) if r[-1] is not None else None}
        for r in rows
    ]


def building_card(
    conn: duckdb.DuckDBPyConnection,
    building_id: str,
    scenario: str,
    horizon: int,
    r: float = 0.05,
    g: float = 0.03,
    carry_rate: float = 0.03,
) -> dict | None:
    """Everything the per-building drawer needs.

    Returns ``None`` if the building id is not in the asset register; otherwise a
    dict with metadata, decided renewal (from ``opt_summary``), condition trajectory
    quantiles, climate exposure by hazard, and the optimal-timing analysis (renewing
    at intervention vs deferring) for this asset.
    """
    meta = conn.execute(
        """
        SELECT split_part(asset_id, '-', 1) AS bid,
               ANY_VALUE(name) AS name, ANY_VALUE(suburb) AS suburb,
               ANY_VALUE(asset_type) AS asset_type,
               MAX(criticality_seed) AS criticality,
               MAX(condition) AS condition_now,
               SUM(grc) AS value
        FROM assets
        WHERE split_part(asset_id, '-', 1) = ?
        GROUP BY split_part(asset_id, '-', 1)
        """,
        [building_id],
    ).fetchone()
    if not meta:
        return None
    bid, name, suburb, asset_type, criticality, cond_now, value = meta

    opt = conn.execute(
        """
        SELECT renew_year_p50, mean_cost, robust, renew_share
        FROM opt_summary
        WHERE asset_id = ? AND scenario = 'stochastic'
        """,
        [bid],
    ).fetchone()
    if opt is None:
        renew_year, mean_cost, robust, renew_share = None, None, False, None
    else:
        renew_year, mean_cost, robust, renew_share = opt

    traj_rows = conn.execute(
        """
        SELECT year,
               AVG(condition) AS mean,
               quantile_cont(condition, 0.05) AS p05,
               quantile_cont(condition, 0.95) AS p95
        FROM mc_paths
        WHERE scenario = ? AND split_part(asset_id, '-', 1) = ?
          AND year BETWEEN ? AND ?
        GROUP BY year ORDER BY year
        """,
        [scenario, bid, CURRENT_YEAR, CURRENT_YEAR + horizon - 1],
    ).fetchall()
    trajectory = [
        {"year": int(y), "mean": float(m), "p05": float(p5), "p95": float(p95)}
        for y, m, p5, p95 in traj_rows
    ]

    clim_rows = conn.execute(
        """
        SELECT year, hazard, AVG(intensity) AS intensity
        FROM climate_exposure
        WHERE scenario = ? AND split_part(asset_id, '-', 1) = ?
          AND year BETWEEN ? AND ?
        GROUP BY year, hazard ORDER BY year, hazard
        """,
        [scenario, bid, CURRENT_YEAR, CURRENT_YEAR + horizon - 1],
    ).fetchall()
    climate = [
        {"year": int(y), "hazard": h, "intensity": float(i)}
        for y, h, i in clim_rows
    ]

    timing = None
    if trajectory:
        cond_path = np.array([t["mean"] for t in trajectory])
        # Use the decided mean_cost where the optimiser sized the job, otherwise the
        # building's total grc — the timing function only needs a representative C0.
        c0 = float(mean_cost) if mean_cost else float(value or 0.0)
        timing = optimal_timing(
            cond_path, c0, float(criticality or 0.5), r, g, carry_rate
        )

    return {
        "bid": bid,
        "name": name,
        "suburb": suburb,
        "asset_type": asset_type,
        "value": float(value or 0.0),
        "criticality": float(criticality or 0.5),
        "condition_now": float(cond_now or 1.0),
        "renew_year": int(renew_year) if renew_year is not None else None,
        "mean_cost": float(mean_cost) if mean_cost else None,
        "robust": bool(robust),
        "renew_share": float(renew_share) if renew_share is not None else None,
        "trajectory": trajectory,
        "climate": climate,
        "timing": timing,
    }


def deferral_impact(
    conn: duckdb.DuckDBPyConnection,
    scenario: str,
    horizon: int,
    override_bids: list[str],
    r: float = 0.05,
    g: float = 0.03,
    carry_rate: float = 0.03,
) -> dict:
    """What happens if the council overrides the optimiser and defers buildings.

    For each selected building that is currently in the funded programme: the
    dollars released in its funded year, the present-value cost of carrying it
    un-renewed from the intervention point to the end of the horizon (plus the
    discounted renewal at the horizon end), and the additional 'years degraded'
    the asset spends past intervention. Buildings that are not currently funded
    are skipped — they're already deferred.

    No optimiser re-solve. The freed capacity is reported per year so the council
    can see exactly which year's spend changes; whether to redirect that money
    elsewhere or hold it is a policy choice the dashboard does not pre-empt.
    """
    impacts: list[dict] = []
    for bid in override_bids:
        card = building_card(
            conn, building_id=bid, scenario=scenario, horizon=horizon,
            r=r, g=g, carry_rate=carry_rate,
        )
        if not card or card["renew_year"] is None:
            continue

        original_year = int(card["renew_year"])
        c0 = float(card["mean_cost"] or card["value"] or 0.0)
        criticality = float(card["criticality"] or 0.5)

        years_to_funded = max(0, original_year - CURRENT_YEAR)
        pv_funded = c0 / (1.0 + r) ** years_to_funded if c0 else 0.0

        timing = card["timing"]
        if timing and timing.get("t_breach") is not None:
            t_breach = int(timing["t_breach"])
            t_end = horizon - 1
            carry_per_yr = carry_rate * criticality * c0
            pv_carry = sum(
                carry_per_yr / (1.0 + r) ** tau
                for tau in range(t_breach, t_end + 1)
            )
            years_past_breach = max(0, t_end - t_breach)
            pv_renew_end = (
                c0 * (1.0 + g) ** years_past_breach / (1.0 + r) ** t_end
                if c0 else 0.0
            )
            pv_override = pv_carry + pv_renew_end
            years_degraded_extra = years_past_breach + 1
        else:
            pv_override = 0.0
            years_degraded_extra = 0

        impacts.append({
            "bid": bid,
            "name": card["name"],
            "suburb": card["suburb"],
            "funded_year": original_year,
            "freed_amount": c0,
            "pv_funded": float(pv_funded),
            "pv_override": float(pv_override),
            "pv_cost_of_override": float(max(0.0, pv_override - pv_funded)),
            "pv_saving_from_override": float(max(0.0, pv_funded - pv_override)),
            "first_breach": (
                CURRENT_YEAR + int(timing["t_breach"])
                if (timing and timing.get("t_breach") is not None) else None
            ),
            "years_degraded_extra": years_degraded_extra,
        })

    freed_by_year: dict[int, float] = {}
    for i in impacts:
        freed_by_year[i["funded_year"]] = (
            freed_by_year.get(i["funded_year"], 0.0) + i["freed_amount"]
        )

    return {
        "rows": impacts,
        "total_freed": float(sum(i["freed_amount"] for i in impacts)),
        "total_pv_cost": float(sum(i["pv_cost_of_override"] for i in impacts)),
        "total_pv_saving": float(sum(i["pv_saving_from_override"] for i in impacts)),
        "freed_by_year": freed_by_year,
    }


def _fetch_baseline_schedule(
    conn: duckdb.DuckDBPyConnection, horizon: int
) -> dict[str, dict]:
    """Canonical funded schedule keyed by building id — what the dashboard already shows."""
    rows = conn.execute(
        """
        SELECT asset_id, renew_year_p50, mean_cost
        FROM opt_summary
        WHERE scenario = 'stochastic' AND renew_year_p50 IS NOT NULL
          AND renew_year_p50 BETWEEN ? AND ?
        """,
        [CURRENT_YEAR, CURRENT_YEAR + horizon - 1],
    ).fetchall()
    return {
        bid: {"year": int(year), "mean_cost": float(mean_cost or 0.0)}
        for bid, year, mean_cost in rows
    }


def _fetch_building_meta(
    conn: duckdb.DuckDBPyConnection, bids: list[str]
) -> dict[str, dict]:
    """Name + suburb + criticality + first-breach lookup for diff rendering."""
    if not bids:
        return {}
    placeholders = ",".join(["?"] * len(bids))
    rows = conn.execute(
        f"""
        SELECT split_part(asset_id, '-', 1) AS bid,
               ANY_VALUE(name) AS name,
               ANY_VALUE(suburb) AS suburb,
               MAX(criticality_seed) AS criticality
        FROM assets
        WHERE split_part(asset_id, '-', 1) IN ({placeholders})
        GROUP BY bid
        """,
        list(bids),
    ).fetchall()
    return {
        bid: {"name": name, "suburb": suburb, "criticality": float(criticality or 0.5)}
        for bid, name, suburb, criticality in rows
    }


def _is_infeasible(status: str | None) -> bool:
    """CVXPY status strings that mean the problem has no feasible solution."""
    return bool(status) and "infeasible" in status.lower()


def force_into_programme(
    conn: duckdb.DuckDBPyConnection,
    scenario: str,
    horizon: int,
    force_bids: list[str],
    annual_budget: float,
    n_scenarios: int = 40,
    lam: float = 1.0,
) -> dict:
    """Re-solve the optimiser with ``force_bids`` constrained to renew.

    Symmetric to :func:`deferral_impact` but for the deferred side of the
    programme: rather than ask "what does deferring this building cost in
    present-value terms?" it asks "if we force this building IN, which other
    building gets pushed out to make room — and where does the optimiser place
    it?". A full re-solve is unavoidable because forcing an asset in consumes
    per-year budget capacity that's then unavailable for others.

    The forced solve is run with ``persist=False`` (the canonical schedule in
    ``opt_summary`` is untouched) and ``robust_pass=False`` (the diff is about
    placement, not robustness across realisations; saves a second solve).

    Returns a diff dict:

    - ``came_in``: bids in the new schedule that were deferred in the canonical
      one (includes the forced bids plus any incidental brought-ins).
    - ``pushed_out``: bids the canonical schedule funded that the new schedule
      defers (the renewals that lost their slot).
    - ``shifted_year``: bids both schedules fund but in different years.
    - ``original_spend_by_year`` / ``new_spend_by_year``: side-by-side bars.
    - ``infeasible`` / ``status``: signal that the per-year budget cannot fit
      the forced set so the caller can tell the user to drop bids or raise the
      budget.

    Empty ``force_bids`` short-circuits to a no-op response without solving.
    """
    force_bids = list(dict.fromkeys(force_bids))  # de-duplicate, preserve order
    baseline = _fetch_baseline_schedule(conn, horizon=horizon)
    original_spend_by_year = {
        y: sum(b["mean_cost"] for b in baseline.values() if b["year"] == y)
        for y in range(CURRENT_YEAR, CURRENT_YEAR + horizon)
    }
    original_spend_by_year = {y: v for y, v in original_spend_by_year.items() if v > 0.0}

    if not force_bids:
        return {
            "force_bids": [],
            "infeasible": False,
            "status": "skipped",
            "came_in": [],
            "pushed_out": [],
            "shifted_year": [],
            "original_spend_by_year": original_spend_by_year,
            "new_spend_by_year": dict(original_spend_by_year),
            "n_came_in": 0,
            "n_pushed_out": 0,
            "n_shifted_year": 0,
        }

    result = solve_all(
        conn,
        horizon=horizon,
        annual_budget=annual_budget,
        n_scenarios=n_scenarios,
        lam=lam,
        must_renew=force_bids,
        persist=False,
        robust_pass=False,
    )
    status = result["status"]
    if _is_infeasible(status):
        return {
            "force_bids": force_bids,
            "infeasible": True,
            "status": status,
            "came_in": [],
            "pushed_out": [],
            "shifted_year": [],
            "original_spend_by_year": original_spend_by_year,
            "new_spend_by_year": {},
            "n_came_in": 0,
            "n_pushed_out": 0,
            "n_shifted_year": 0,
        }

    new_schedule = {
        u["asset_id"]: {
            "year": int(u["renew_year"]) if u["renew_year"] else None,
            "mean_cost": float(u["mean_cost"]),
        }
        for u in result["units"]
        if u["renewed"] and u["renew_year"]
        and CURRENT_YEAR <= int(u["renew_year"]) < CURRENT_YEAR + horizon
    }

    old_keys = set(baseline)
    new_keys = set(new_schedule)
    forced_set = set(force_bids)
    came_in_bids = sorted(new_keys - old_keys)
    pushed_out_bids = sorted(old_keys - new_keys)
    shifted_bids = sorted({
        bid for bid in (old_keys & new_keys)
        if baseline[bid]["year"] != new_schedule[bid]["year"]
    })

    meta = _fetch_building_meta(
        conn, list(set(came_in_bids) | set(pushed_out_bids) | set(shifted_bids))
    )

    came_in = [
        {
            "bid": bid,
            "name": meta.get(bid, {}).get("name", bid),
            "suburb": meta.get(bid, {}).get("suburb"),
            "criticality": meta.get(bid, {}).get("criticality", 0.5),
            "renewal_year": new_schedule[bid]["year"],
            "mean_cost": new_schedule[bid]["mean_cost"],
            "was_forced": bid in forced_set,
        }
        for bid in came_in_bids
    ]
    pushed_out = [
        {
            "bid": bid,
            "name": meta.get(bid, {}).get("name", bid),
            "suburb": meta.get(bid, {}).get("suburb"),
            "criticality": meta.get(bid, {}).get("criticality", 0.5),
            "original_year": baseline[bid]["year"],
            "freed_amount": baseline[bid]["mean_cost"],
        }
        for bid in pushed_out_bids
    ]
    shifted_year = [
        {
            "bid": bid,
            "name": meta.get(bid, {}).get("name", bid),
            "suburb": meta.get(bid, {}).get("suburb"),
            "original_year": baseline[bid]["year"],
            "new_year": new_schedule[bid]["year"],
            "mean_cost": new_schedule[bid]["mean_cost"],
        }
        for bid in shifted_bids
    ]

    new_spend_by_year: dict[int, float] = {}
    for sched in new_schedule.values():
        year = sched["year"]
        if year is not None:
            new_spend_by_year[year] = new_spend_by_year.get(year, 0.0) + sched["mean_cost"]

    return {
        "force_bids": force_bids,
        "infeasible": False,
        "status": status,
        "came_in": came_in,
        "pushed_out": pushed_out,
        "shifted_year": shifted_year,
        "original_spend_by_year": original_spend_by_year,
        "new_spend_by_year": new_spend_by_year,
        "n_came_in": len(came_in),
        "n_pushed_out": len(pushed_out),
        "n_shifted_year": len(shifted_year),
    }
