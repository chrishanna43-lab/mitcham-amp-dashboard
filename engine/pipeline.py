"""End-to-end pipeline orchestrator — every stage in order, one call.

Drives the seven engine stages (schema -> ingest -> climate -> calibrate ->
score -> simulate -> optimise) against a single DuckDB file, then renders the
Mitcham-branded renewal-outlook report. Reproducible from a seed; the working
DB is disposable. Returns the headline funding-gap figure plus row counts so a
caller (CLI or test) can report the result without re-querying.
"""
from __future__ import annotations

from pathlib import Path

from engine import db as _db
from engine.calibrate import write_priors
from engine.climate.exposure import build_exposure, write_exposure
from engine.config import DEMO_ANNUAL_BUDGET
from engine.geo.placement import place_buildings
from engine.ingest.mitcham_public import generate_register, write_register
from engine.optimise.solver import solve_all
from engine.render.metrics import unfunded_liability
from engine.render.report.generate import write_report
from engine.score.aggregation import score_all
from engine.simulate.aggregate import aggregate
from engine.simulate.monte_carlo import run_monte_carlo


def run_pipeline(
    db_path: str | Path,
    horizon: int = 25,
    n_realisations: int = 200,
    annual_budget: float = DEMO_ANNUAL_BUDGET,
    n_scenarios: int = 40,
    scenarios: tuple[str, ...] = ("no_climate", "rcp45", "rcp85"),
    lam: float = 1.0,
    solver: str = "MOSEK",
    granularity: str = "building",
    report_out: str | Path | None = None,
    scenario_for_report: str = "rcp45",
    seed: int = 20260527,
) -> dict:
    """Run all engine stages end to end and render the report.

    Returns a dict with ``n_components`` (asset rows), ``mc_paths`` (Monte Carlo
    rows), ``opt_units`` (optimised decision units), ``opt`` (the solver result),
    ``headline`` (the unfunded-liability figure for ``scenario_for_report``), and
    ``report`` (the path written).
    """
    db_path = Path(db_path)
    conn = _db.connect(db_path)
    try:
        _db.bootstrap_schema(conn)

        # Stage 1 — ingest the Mitcham buildings register.
        rows = generate_register(seed=seed)
        write_register(conn, rows)
        # Place each building at a real coordinate within its actual suburb.
        place_buildings(conn, seed=seed)

        # Stage 2 — climate exposure over the horizon.
        write_exposure(conn, build_exposure(rows, horizon=horizon))

        # Stage 3 — deterioration / cost / climate-factor priors.
        write_priors(conn)

        # Stage 4 — multi-criteria scores.
        score_all(conn)

        # Stage 5 — Monte Carlo deterioration paths + per-scenario summary.
        run_monte_carlo(
            conn,
            n_realisations=n_realisations,
            horizon=horizon,
            scenarios=scenarios,
            seed=seed,
        )
        aggregate(conn)

        # Stage 6 — two-stage stochastic renewal optimisation.
        opt_result = solve_all(
            conn,
            horizon=horizon,
            annual_budget=annual_budget,
            n_scenarios=n_scenarios,
            lam=lam,
            solver=solver,
            seed=seed,
            granularity=granularity,
        )

        # Council-facing headline (dollars), computed from Monte Carlo demand.
        headline = unfunded_liability(
            conn,
            scenario=scenario_for_report,
            horizon=horizon,
            annual_budget=annual_budget,
        )

        n_components = conn.execute("SELECT count(*) FROM assets").fetchone()[0]
        mc_paths = conn.execute("SELECT count(*) FROM mc_paths").fetchone()[0]
        opt_units = conn.execute("SELECT count(*) FROM opt_summary").fetchone()[0]
    finally:
        conn.close()

    # Stage 7 — render the branded report (opens its own connection).
    report_out = Path(report_out) if report_out is not None else (
        db_path.parent / "mitcham-buildings-renewal-outlook.html"
    )
    write_report(
        db_path,
        report_out,
        scenario=scenario_for_report,
        annual_budget=annual_budget,
        horizon=horizon,
    )

    return {
        "n_components": n_components,
        "mc_paths": mc_paths,
        "opt_units": opt_units,
        "opt": opt_result,
        "headline": headline,
        "report": report_out,
    }
