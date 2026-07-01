"""What-If recompute orchestration (the load-bearing backend, S.2.4).

Re-runs Stage 2 (climate exposure) + Stage 5 (Monte Carlo + aggregate) — and,
on demand, Stage 6 (optimiser) — on a **shadow** connection over whatever assets
it holds. The canonical DB is never touched; every write here lands on the
shadow (defended by ``_is_shadow``).

Two correctness pins, both verified against source:
  * [FIX-WHATIF-BLOCKER-1] ``build_exposure`` does attribute access
    (``asset.asset_id``), so it needs Asset OBJECTS, not the dicts
    ``_fetch_assets`` returns. ``_fetch_asset_models`` rebuilds Asset models
    exactly as ``engine/pipeline.py``'s climate stage feeds them.
  * [FIX-WHATIF-BLOCKER-2] the proposed re-solve persists under
    ``label='stochastic'`` so the existing programme renderers (which hard-filter
    ``scenario='stochastic'``) read it on the shadow unchanged; the baseline
    ``units`` are snapshotted in memory before the re-solve for the displacement
    diff (S.4.3).
"""
from __future__ import annotations

from engine.climate.exposure import build_exposure, write_exposure
from engine.ingest.mitcham_public import _COLUMNS  # canonical Asset column tuple
from engine.db import insert_models
from engine.schemas import Asset
from engine.simulate.aggregate import aggregate
from engine.simulate.monte_carlo import run_monte_carlo
from engine.optimise.solver import solve_all
from engine.whatif.shadow import _is_shadow

SCENARIOS = ("no_climate", "rcp45", "rcp85")
SHADOW_SEED = 20260527

# Full Asset model field set, in schema order — superset of the ingest
# ``_COLUMNS`` (it also carries ``suburb``). Used to rehydrate Asset objects from
# the shadow for the climate stage.
_ASSET_FIELDS = (
    "asset_id", "name", "suburb", "council", "asset_type", "asset_class",
    "component", "install_year", "useful_life_years", "condition", "grc",
    "extent", "lat", "lon", "criticality_seed",
)


def _fetch_asset_models(conn) -> list[Asset]:
    """Return all ``assets`` rows as validated :class:`Asset` objects.

    Mirrors how ``engine/pipeline.py``'s climate stage feeds ``build_exposure``
    (which passes ``generate_register()`` Asset objects directly). ``build_exposure``
    uses attribute access (``asset.asset_id``), so dicts from ``_fetch_assets``
    would raise ``AttributeError`` ([FIX-WHATIF-BLOCKER-1]).
    """
    cols = ", ".join(_ASSET_FIELDS)
    rows = conn.execute(
        f"SELECT {cols} FROM assets ORDER BY asset_id"
    ).fetchall()
    return [Asset(**dict(zip(_ASSET_FIELDS, r))) for r in rows]


def run_stage_2_5(conn, *, horizon, n_realisations, scenarios=SCENARIOS, seed=SHADOW_SEED):
    """Re-run climate (Stage 2) + Monte Carlo (Stage 5) + aggregate on the shadow.

    ``DELETE FROM mc_paths`` inside ``run_monte_carlo`` is SAFE because ``conn`` is
    the shadow copy, not the canonical handle.
    """
    assets = _fetch_asset_models(conn)                                  # Asset objects
    write_exposure(conn, build_exposure(assets, horizon=horizon))       # Stage 2
    run_monte_carlo(
        conn, n_realisations=n_realisations, horizon=horizon,
        scenarios=scenarios, seed=seed,
    )                                                                    # Stage 5
    aggregate(conn)                                                      # mc_summary


def recompute_with_new_assets(
    conn,
    new_rows,
    *,
    scenario,
    horizon,
    annual_budget,
    n_realisations,
    n_scenarios,
    resolve=False,
    baseline_units=None,
    mc_horizon=None,
):
    """Insert proposal rows on the shadow, re-run Stage 2/5 (+6 if ``resolve``).

    ``conn`` MUST be the shadow. ``new_rows`` is a list of :class:`Asset` models
    (e.g. from ``build_proposed_building``). When ``resolve`` is True the proposed
    Stage-6 solve persists under ``label='stochastic'`` so existing programme
    renderers read it unchanged ([FIX-WHATIF-BLOCKER-2]); pass the in-memory
    baseline ``units`` as ``baseline_units`` (snapshotted before this call) for
    the displacement diff. Returns the proposed solve's ``units`` (or ``None``
    when ``resolve`` is False).

    [H6] ``mc_horizon`` (optional) runs the Monte Carlo to a longer horizon than
    the decision/solve ``horizon`` — so the dual-horizon (e.g. 50-year) compliance
    corridor reads real simulated data instead of a table truncated at ``horizon``.
    The Stage-6 solve and displacement diff stay at ``horizon``.
    """
    assert _is_shadow(conn)                              # defence in depth (S.2.3)

    if new_rows:
        insert_models(conn, "assets", _COLUMNS, new_rows)  # strict Asset model validated bounds

    run_stage_2_5(conn, horizon=mc_horizon or horizon, n_realisations=n_realisations)

    units = None
    if resolve:                                          # Tier B' only
        res = solve_all(
            conn, horizon=horizon, annual_budget=annual_budget,
            n_scenarios=n_scenarios, seed=SHADOW_SEED,
            robust_pass=False, persist=True, label="stochastic",
        )
        units = res["units"]
    return units


def baseline_solve(conn, *, horizon, annual_budget, n_scenarios):
    """Run the baseline Stage-6 solve on the shadow and return its ``units``.

    Same ``n_scenarios``/seed as the proposed solve so a "displaced" building is
    flagged only when its renew status actually changes, not from sampling churn
    ([FIX-G6]). Persists under ``label='stochastic'`` (the renderers' filter).
    """
    assert _is_shadow(conn)
    res = solve_all(
        conn, horizon=horizon, annual_budget=annual_budget,
        n_scenarios=n_scenarios, seed=SHADOW_SEED,
        robust_pass=False, persist=True, label="stochastic",
    )
    return res["units"]
