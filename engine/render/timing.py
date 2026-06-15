"""Optimal renewal timing — the present-value trade-off behind deferral.

The funding-gap view in ``metrics.py`` treats renewal demand as a fixed bill and
sums it in nominal dollars. That answers "how much is unfunded", but not "when
should each asset be renewed" — and deferral is not always a cost. For an asset
that has reached its intervention condition, the decision to renew now or wait is
a race between two forces:

* the **time value of money** — capital not yet spent earns the council's real
  discount rate ``r``; this rewards waiting;
* the **cost of decay** — once past intervention the job drifts from renewal
  toward reconstruction (escalation ``g``), and the asset imposes a service/risk
  carrying cost each year it is left un-renewed; these reward acting.

Present value of renewing an asset ``k`` years after it breaches intervention
(measuring escalation and carry from the breach point, discounting to today)::

    TPV(k) = sum_{j=0}^{k-1} carry / (1+r)^(t_b + j)        # carry it while waiting
             + C0 * (1+g)^k / (1+r)^(t_b + k)               # discounted renewal

with ``carry = carry_rate * criticality * C0`` per year past intervention. The
optimal deferral ``k*`` minimises ``TPV``. The marginal rule (defer one more year
while the saving on delayed capex beats the extra carrying cost) reduces to::

    defer while   carry  <  C_k * (r - g) / (1 + r)

so ``g >= r`` forces immediate renewal at the breach point, while a low-criticality
asset with ``r > g`` is correctly renewed *later* — deferral saves money.

``r`` is a policy input (a council/Treasury real discount rate); ``g`` and
``carry_rate`` are calibrated, illustrative engineering assumptions and must carry
chartered-engineer sign-off before any real capital decision. Decision support,
not decision authority.
"""
from __future__ import annotations

import duckdb
import numpy as np

CURRENT_YEAR = 2026
INTERVENTION = 4.0


def optimal_timing(
    condition: np.ndarray,
    c0: float,
    criticality: float,
    r: float,
    g: float,
    carry_rate: float,
) -> dict:
    """Optimal renewal timing for one asset from its expected condition path.

    ``condition`` is the year-by-year expected condition index (1 new .. 5 failed)
    over the horizon. Returns the breach year offset, the optimal renewal offset,
    the years deferred past breach, the present-value saving versus renewing at
    breach, and the full ``TPV`` curve (one value per candidate renewal year from
    breach to the horizon end) for plotting. If the asset never reaches the
    intervention condition within the horizon, ``t_breach`` is ``None`` and no
    renewal is due.
    """
    cond = np.asarray(condition, dtype=float)
    horizon = cond.size
    breached = np.where(cond >= INTERVENTION)[0]
    if breached.size == 0:
        return {
            "t_breach": None, "t_star": None, "defer_years": 0,
            "saving": 0.0, "tpv": [], "candidate_years": [], "act_now": False,
        }
    t_b = int(breached[0])
    carry = carry_rate * criticality * c0
    max_k = horizon - 1 - t_b
    tpv: list[float] = []
    for k in range(max_k + 1):
        pv_carry = sum(carry / (1.0 + r) ** (t_b + j) for j in range(k))
        pv_renew = c0 * (1.0 + g) ** k / (1.0 + r) ** (t_b + k)
        tpv.append(pv_carry + pv_renew)
    tpv_arr = np.array(tpv)
    k_star = int(np.argmin(tpv_arr))
    return {
        "t_breach": t_b,
        "t_star": t_b + k_star,
        "defer_years": k_star,
        "saving": float(tpv_arr[0] - tpv_arr[k_star]),
        "tpv": [float(v) for v in tpv_arr],
        "candidate_years": [CURRENT_YEAR + t_b + k for k in range(max_k + 1)],
        "act_now": k_star == 0,
    }


def _building_condition_paths(
    conn: duckdb.DuckDBPyConnection, scenario: str, horizon: int
) -> dict:
    """Per-building inputs for the timing model, aggregated from ``mc_paths``.

    A building's condition path is the worst (max) expected component condition
    in each year; its renewal cost is the sum of component renewal costs; its
    criticality is the worst component's. Returns ``{building_id: {...}}``.
    """
    rows = conn.execute(
        """
        WITH comp AS (
            SELECT split_part(p.asset_id, '-', 1) AS bid,
                   p.asset_id AS aid, p.year AS year,
                   AVG(p.condition) AS cond,
                   AVG(p.cost_if_renewed) AS cost
            FROM mc_paths p
            WHERE p.scenario = ? AND p.year < ?
            GROUP BY bid, p.asset_id, p.year
        ),
        comp_cost AS (
            SELECT bid, aid, AVG(cost) AS comp_cost FROM comp GROUP BY bid, aid
        )
        SELECT c.bid AS bid, c.year AS year,
               MAX(c.cond) AS cond,
               (SELECT SUM(comp_cost) FROM comp_cost cc WHERE cc.bid = c.bid) AS c0
        FROM comp c
        GROUP BY c.bid, c.year
        ORDER BY c.bid, c.year
        """,
        [scenario, CURRENT_YEAR + horizon],
    ).fetchall()
    meta = {
        bid: {"name": name, "suburb": suburb, "criticality": float(crit or 0.5)}
        for bid, name, suburb, crit in conn.execute(
            """
            SELECT split_part(asset_id, '-', 1) AS bid,
                   ANY_VALUE(name), ANY_VALUE(suburb), MAX(criticality_seed)
            FROM assets GROUP BY bid
            """
        ).fetchall()
    }
    out: dict[str, dict] = {}
    for bid, year, cond, c0 in rows:
        b = out.setdefault(
            bid,
            {
                "name": meta.get(bid, {}).get("name", bid),
                "suburb": meta.get(bid, {}).get("suburb", ""),
                "criticality": meta.get(bid, {}).get("criticality", 0.5),
                "c0": float(c0 or 0.0),
                "years": [], "cond": [],
            },
        )
        b["years"].append(int(year))
        b["cond"].append(float(cond))
    return out


def deferral_timing(
    conn: duckdb.DuckDBPyConnection,
    scenario: str,
    horizon: int,
    r: float = 0.05,
    g: float = 0.03,
    carry_rate: float = 0.03,
) -> dict:
    """Portfolio optimal-timing summary under a discount/escalation/carry regime.

    For every building that reaches intervention within the horizon, computes the
    optimal renewal year. Returns counts (defer vs act-at-breach), the aggregate
    present-value saving from optimal timing versus renewing every asset at its
    breach point, a per-building table, and a portfolio "cost of waiting" curve:
    the total present-value cost if every renewal were uniformly deferred ``k``
    years past its own intervention point, with the optimal uniform deferral
    ``k_star`` marked.
    """
    paths = _building_condition_paths(conn, scenario, horizon)
    rows: list[dict] = []
    per_building_tpv: list[np.ndarray] = []
    for bid, b in paths.items():
        if b["c0"] <= 0:
            continue
        res = optimal_timing(
            np.array(b["cond"]), b["c0"], b["criticality"], r, g, carry_rate
        )
        if res["t_breach"] is None:
            continue
        rows.append({
            "building": b["name"], "suburb": b["suburb"],
            "value": b["c0"], "criticality": b["criticality"],
            "breach_year": CURRENT_YEAR + res["t_breach"],
            "optimal_year": CURRENT_YEAR + res["t_star"],
            "defer_years": res["defer_years"], "saving": res["saving"],
            "act_now": res["act_now"],
        })
        per_building_tpv.append(np.array(res["tpv"]))

    n = len(rows)
    n_defer = sum(1 for x in rows if x["defer_years"] > 0)
    total_saving = float(sum(x["saving"] for x in rows))

    # Portfolio "cost of waiting": total PV if every renewal is deferred k years
    # past its own breach. Buildings breaching late have shorter curves, so pad
    # each with its last (longest-deferral) value to keep a common k-axis.
    curve_k: list[int] = []
    curve_tpv: list[float] = []
    if per_building_tpv:
        max_len = max(len(c) for c in per_building_tpv)
        padded = np.array([
            np.concatenate([c, np.full(max_len - len(c), c[-1])]) for c in per_building_tpv
        ])
        totals = padded.sum(axis=0)
        curve_k = list(range(max_len))
        curve_tpv = [float(v) for v in totals]
        k_star = int(np.argmin(totals))
    else:
        k_star = 0

    return {
        "r": r, "g": g, "carry_rate": carry_rate,
        "n_assets": n, "n_defer": n_defer, "n_act_now": n - n_defer,
        "total_saving": total_saving,
        "k_star": k_star,
        "curve_k": curve_k, "curve_tpv": curve_tpv,
        "rows": sorted(rows, key=lambda x: x["saving"], reverse=True),
    }
