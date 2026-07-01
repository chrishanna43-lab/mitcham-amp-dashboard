"""Headless, Streamlit-free logic for the What-If view (S.3.5, S.4.3, S.5).

Everything here is pure: it never imports ``streamlit`` and takes only plain
data (proposal dicts, ``financial_indicators`` dicts, ``solve_all`` ``units``
lists). The Streamlit render functions in ``engine/render/dashboard/app.py``
call into it so the load-bearing logic — the content-addressed cache key, the
net-new-liability predicate, the Channel-A marginal delta assembly, and the
optimiser displacement diff — is unit-testable without a display.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from engine.render.metrics import marginal_economics

CURRENT_YEAR = 2026


# --------------------------------------------------------------------------- #
# Content-addressed cache key (S.3.5, [FIX-D6]).
# --------------------------------------------------------------------------- #
def row_digest(rows: list[Any]) -> str:
    """Stable content hash of a proposal's expanded :class:`Asset` rows.

    Keys the recompute cache on the *content* of the rows, not the building id,
    so editing a proposal's GRC / condition / install_year invalidates a stale
    diff ([FIX-D6]). ``rows`` items expose ``model_dump`` (pydantic ``Asset``)
    or are plain dicts.
    """
    payload = sorted(
        tuple(sorted(_as_dict(r).items(), key=lambda kv: kv[0]))
        for r in rows
    )
    return hashlib.sha256(
        json.dumps(payload, default=str, sort_keys=True).encode()
    ).hexdigest()


def _as_dict(row: Any) -> dict:
    if isinstance(row, dict):
        return dict(row)
    if hasattr(row, "model_dump"):
        return row.model_dump()
    raise TypeError(f"Cannot digest row of type {type(row)!r}")


def whatif_cache_key(
    scenario: str,
    budget: float,
    horizon: int,
    resolve_flag: bool,
    proposals: list[dict],
) -> tuple:
    """Content-addressed recompute cache key (S.3.5).

    Only *enabled* proposals participate; the per-proposal contribution is the
    ``(building_id, row_digest)`` pair, so a slider change that re-solves does
    not invalidate the proposals (they are portfolio facts) but an edit to a
    proposal's rows does. ``budget`` is rounded to the nearest $10k so trivial
    slider jitter does not thrash the cache.
    """
    return (
        scenario,
        round(float(budget), -4),
        int(horizon),
        bool(resolve_flag),
        frozenset(
            (p["building_id"], row_digest(p["rows"]))
            for p in proposals
            if p.get("enabled")
        ),
    )


def baseline_cache_key(
    scenario: str,
    budget: float,
    horizon: int,
    n_real: int,
    n_scen: int,
    resolve_flag: bool,
) -> tuple:
    """Key for the canonical-only baseline cache (S.3.5 baseline caching).

    EXACTLY the inputs the baseline pass depends on — and NOTHING proposal-
    dependent — so a cached baseline is reused across proposal edits. ``budget`` is
    rounded to $10k to match :func:`whatif_cache_key`; the caller MUST compute the
    baseline metrics at the same rounded value (threads ``budget_q`` through both)
    so the key can never gate a baseline computed at a different budget. ``n_scen``
    participates ONLY under ``resolve_flag``, because the cached ``base_units`` come
    from ``baseline_solve(..., n_scenarios=n_scen)``.

    Single source of truth: both ``_tab_whatif`` and the tests import this — never
    re-derive the tuple inline (a replicated copy is how the granularities drifted).
    """
    return (
        scenario,
        round(float(budget), -4),
        int(horizon),
        int(n_real),
        bool(resolve_flag),
        int(n_scen) if resolve_flag else None,
    )


# --------------------------------------------------------------------------- #
# Net-new-liability badge predicate (S.5, [FIX-MINOR1]).
# --------------------------------------------------------------------------- #
def min_component_life(rows: list[Any]) -> float:
    """Shortest useful life across a proposal's component rows (the earliest breach)."""
    return min(float(_get(r, "useful_life_years")) for r in rows)


def net_new_liability(
    rows: list[Any],
    commission_year: int,
    horizon: int,
    current_year: int = CURRENT_YEAR,
) -> bool:
    """True iff the proposal's earliest component breach lands beyond the horizon.

    Fires when ``commission_year + min(component life) >= current_year + horizon``
    — i.e. the asset flatters the near-term ratios but embeds a renewal liability
    that only the longer (50-yr) corridor reveals (S.1.2, S.5). For a standard
    build HVAC (20 yr) is the binding component. [L11] Uses ``>=`` to match the
    half-open demand window (``year < current_year + horizon``): a breach landing
    exactly at the horizon end is already out-of-corridor, so the badge must fire.
    """
    return commission_year + min_component_life(rows) >= current_year + horizon


# --------------------------------------------------------------------------- #
# Channel-A marginal delta assembly (S.2.2, S.5).
# --------------------------------------------------------------------------- #
def channel_a_metrics(
    baseline_econ: dict,
    baseline_rc: dict,
    new_rows: list[Any],
    current_year: int = CURRENT_YEAR,
) -> dict:
    """Add ``marginal_economics`` deltas to baseline Channel-A figures (S.2.2).

    Channel A is register-derived and exact: ACR, ASR-denominator (depreciation),
    renewal backlog and portfolio grade move by a closed-form delta the instant a
    proposal is known — no solve. This does NOT re-derive the ratio formulae; it
    threads ``marginal_economics`` deltas through the baseline values the
    indicators panel already exposes ([FIX-G7]).

    Returns a dict of ``{base, proposed, delta}`` per metric plus the raw
    marginal-economics deltas. ARFR / gap / corridor are Channel B and are NOT
    computed here (they need a Stage-5 re-run, S.2.1).
    """
    me = marginal_economics(new_rows, current_year=current_year)

    grc0 = float(baseline_econ.get("grc_total") or 0.0)
    drc0 = float(baseline_econ.get("drc_total") or 0.0)
    dep0 = float(baseline_econ.get("annual_depreciation") or 0.0)
    f0 = float(baseline_rc["grade_value"].get("F", 0.0)) if baseline_rc else 0.0
    total0 = float(baseline_rc.get("total_value") or 0.0) if baseline_rc else 0.0

    grc1 = grc0 + me["d_grc"]
    drc1 = drc0 + me["d_drc"]
    dep1 = dep0 + me["d_depreciation"]
    f1 = f0 + me["d_f_value"]
    total1 = total0 + me["d_grc"]

    acr0 = drc0 / grc0 if grc0 else None
    acr1 = drc1 / grc1 if grc1 else None
    backlog0 = f0 / total0 if total0 else None
    backlog1 = f1 / total1 if total1 else None

    return {
        "marginal": me,
        "acr": _triple(acr0, acr1),
        "depreciation": _triple(dep0, dep1),   # ASR denominator
        "backlog": _triple(backlog0, backlog1),
        "grc_total": _triple(grc0, grc1),
        "drc_total": _triple(drc0, drc1),
        "f_value": _triple(f0, f1),
    }


def _triple(base: float | None, proposed: float | None) -> dict:
    delta = (
        (proposed - base)
        if (base is not None and proposed is not None)
        else None
    )
    return {"base": base, "proposed": proposed, "delta": delta}


# --------------------------------------------------------------------------- #
# Optimiser displacement diff from two unit lists (S.4.3, Tier B').
# --------------------------------------------------------------------------- #
def _renewed_in_horizon(units: list[dict], horizon: int, current_year: int) -> dict[str, dict]:
    """Map asset_id -> unit for units renewed within the horizon."""
    out: dict[str, dict] = {}
    for u in units or []:
        if not u.get("renewed"):
            continue
        yr = u.get("renew_year")
        if yr is None:
            continue
        yr = int(yr)
        if current_year <= yr < current_year + horizon:
            out[u["asset_id"]] = u
    return out


def displacement_diff(
    baseline_units: list[dict],
    proposed_units: list[dict],
    horizon: int,
    *,
    proposal_prefixes: tuple[str, ...] = ("p", "u"),
    current_year: int = CURRENT_YEAR,
) -> dict:
    """Diff two ``solve_all`` ``units`` lists into a programme displacement view.

    Both lists are persisted under ``label='stochastic'`` on the SAME shadow at
    the SAME ``n_scenarios``/seed (S.2.5, [FIX-G6]) so a "displaced" building is a
    genuine re-prioritisation, not solver sampling churn. Splits the proposed
    schedule into:

    - ``new_in``: proposal buildings (``p``/``u`` ids) the proposed solve funds —
      tinted GOLD in the UI.
    - ``displaced``: canonical buildings funded in the baseline solve but pushed
      out of the proposed solve — tinted RED.
    - ``shifted``: buildings both solves fund but in different years.

    Returns counts plus the per-building rows so the renderer is presentation-only.
    """
    base = _renewed_in_horizon(baseline_units, horizon, current_year)
    prop = _renewed_in_horizon(proposed_units, horizon, current_year)

    def _is_proposal(aid: str) -> bool:
        head = aid.split("-", 1)[0]
        return any(head.startswith(pfx) for pfx in proposal_prefixes)

    base_keys = set(base)
    prop_keys = set(prop)

    new_in = sorted(
        aid for aid in (prop_keys - base_keys) if _is_proposal(aid)
    )
    # Displaced: funded in baseline, not in proposed, and a canonical building.
    displaced = sorted(
        aid for aid in (base_keys - prop_keys) if not _is_proposal(aid)
    )
    shifted = sorted(
        aid for aid in (base_keys & prop_keys)
        if base[aid].get("renew_year") != prop[aid].get("renew_year")
    )

    return {
        "new_in": [
            {"asset_id": aid, "renew_year": prop[aid].get("renew_year"),
             "mean_cost": prop[aid].get("mean_cost")}
            for aid in new_in
        ],
        "displaced": [
            {"asset_id": aid, "original_year": base[aid].get("renew_year"),
             "freed_amount": base[aid].get("mean_cost")}
            for aid in displaced
        ],
        "shifted": [
            {"asset_id": aid, "original_year": base[aid].get("renew_year"),
             "new_year": prop[aid].get("renew_year")}
            for aid in shifted
        ],
        "n_new_in": len(new_in),
        "n_displaced": len(displaced),
        "n_shifted": len(shifted),
    }


def _get(row: Any, key: str):
    return row[key] if isinstance(row, dict) else getattr(row, key)
