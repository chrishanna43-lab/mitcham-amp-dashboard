"""LGA-AMP CLI — one sub-command per pipeline stage."""
from __future__ import annotations

from pathlib import Path

import typer

from engine import db as _db
from engine.config import DEFAULT_DB, DEMO_ANNUAL_BUDGET

app = typer.Typer(help="LGA-AMP — Council asset-management lifecycle engine.")


@app.command("db-init")
def db_init(db: Path = DEFAULT_DB) -> None:
    """Create (or refresh) the DuckDB schema."""
    conn = _db.connect(db)
    _db.bootstrap_schema(conn)
    conn.close()
    typer.echo(f"schema bootstrapped at {db}")


@app.command()
def ingest(db: Path = DEFAULT_DB, seed: int = 42) -> None:
    """Stage 1: ingest Mitcham public buildings data into the asset register."""
    from engine.ingest.mitcham_public import generate_register, write_register
    conn = _db.connect(db)
    _db.bootstrap_schema(conn)
    rows = generate_register(seed=seed)
    write_register(conn, rows)
    n = conn.execute("SELECT count(*) FROM assets").fetchone()[0]
    conn.close()
    n_buildings = len({r.asset_id.split('-')[0] for r in rows})
    typer.echo(f"ingested {n} component records ({n_buildings} buildings)")


@app.command()
def geo(db: Path = DEFAULT_DB, seed: int = 20260528) -> None:
    """Place buildings at real coordinates within their actual Mitcham suburb."""
    from engine.geo.placement import place_buildings
    conn = _db.connect(db)
    r = place_buildings(conn, seed=seed)
    conn.close()
    typer.echo(
        f"placed {r['n_buildings']} buildings across {len(r['suburb_counts'])} suburbs "
        f"({r['n_unmatched']} unmatched)"
    )


@app.command()
def climate(db: Path = DEFAULT_DB, horizon: int = 25) -> None:
    """Stage 2: build climate exposure for every asset over the horizon."""
    from engine.climate.exposure import build_exposure, write_exposure
    from engine.schemas import Asset
    conn = _db.connect(db)
    rows = conn.execute("SELECT * FROM assets").fetchall()
    cols = [d[0] for d in conn.description]
    assets = [Asset(**dict(zip(cols, r))) for r in rows]
    expo = build_exposure(assets, horizon=horizon)
    write_exposure(conn, expo)
    n = conn.execute("SELECT count(*) FROM climate_exposure").fetchone()[0]
    conn.close()
    typer.echo(f"wrote {n} climate-exposure rows over {horizon} years x 3 scenarios x 3 hazards")


@app.command()
def calibrate(db: Path = DEFAULT_DB) -> None:
    """Stage 3: load deterioration / cost / climate-factor priors."""
    from engine.calibrate import write_priors
    conn = _db.connect(db)
    _db.bootstrap_schema(conn)
    write_priors(conn)
    n_priors = conn.execute("SELECT count(*) FROM priors").fetchone()[0]
    n_factors = conn.execute("SELECT count(*) FROM climate_factors").fetchone()[0]
    conn.close()
    typer.echo(f"wrote {n_priors} priors, {n_factors} climate factors")


@app.command()
def score(db: Path = DEFAULT_DB) -> None:
    """Stage 4: derive multi-criteria scores from public-signal proxies."""
    from engine.score.aggregation import score_all
    conn = _db.connect(db)
    n = score_all(conn)
    conn.close()
    typer.echo(f"wrote {n} criteria-score rows (4 axes per asset)")


@app.command()
def simulate(db: Path = DEFAULT_DB, n: int = 200, horizon: int = 25, seed: int = 20260527) -> None:
    """Stage 5: Monte Carlo deterioration paths."""
    from engine.simulate.monte_carlo import run_monte_carlo
    from engine.simulate.aggregate import aggregate
    conn = _db.connect(db)
    rows = run_monte_carlo(conn, n_realisations=n, horizon=horizon, seed=seed)
    m = aggregate(conn)
    conn.close()
    typer.echo(f"wrote {rows} mc_paths rows and {m} mc_summary rows")


@app.command()
def optimise(db: Path = DEFAULT_DB, horizon: int = 25, annual_budget: float = DEMO_ANNUAL_BUDGET,
             n_scenarios: int = 60, lam: float = 1.0, solver: str = "MOSEK",
             granularity: str = "building", value_weighted: bool = True) -> None:
    """Stage 6: two-stage stochastic renewal optimisation with CVaR.

    granularity: 'building' (default, fast/optimal) or 'component' (finer, slower).
    value_weighted: weight breach by replacement value so major facilities surface
    (--no-value-weighted for a pure criticality-weighted breach count).
    """
    from engine.optimise.solver import solve_all
    conn = _db.connect(db)
    r = solve_all(conn, horizon=horizon, annual_budget=annual_budget,
                  n_scenarios=n_scenarios, lam=lam, solver=solver,
                  granularity=granularity, value_weighted=value_weighted)
    conn.close()
    typer.echo(
        f"optimised {r['n_assets']} {r['granularity']}-level units: "
        f"{r['n_renewed']} renewed, {r['n_robust']} robust (status={r['status']}); "
        f"residual service-risk p05-p95 {r['realised_p05']:,.0f}-{r['realised_p95']:,.0f}"
    )


@app.command()
def engagement(db: Path = DEFAULT_DB, horizon: int = 25, annual_budget: float = 800_000.0,
               n_scenarios: int = 60, solver: str = "MOSEK", seed: int = 0) -> None:
    """Solve both lenses at one budget: engineering optimum + engagement-weighted.

    Writes the engineering programme to opt_summary (scenario='stochastic') and
    the community/elected engagement-weighted programme alongside it
    (scenario='stochastic_engagement'), so the dashboard can show both.
    """
    from engine.engage.scores import building_multipliers
    from engine.optimise.solver import solve_all
    conn = _db.connect(db)
    eng = building_multipliers(conn)
    base = solve_all(conn, horizon=horizon, annual_budget=annual_budget,
                     n_scenarios=n_scenarios, solver=solver, seed=seed,
                     label="stochastic", write_solutions=True)
    adj = solve_all(conn, horizon=horizon, annual_budget=annual_budget,
                    n_scenarios=n_scenarios, solver=solver, seed=seed,
                    engagement=eng, label="stochastic_engagement", write_solutions=False)
    conn.close()
    typer.echo(
        f"engineering optimum: {base['n_renewed']} renewed; "
        f"engagement-weighted: {adj['n_renewed']} renewed"
    )


@app.command()
def render(
    db: Path = DEFAULT_DB,
    target: str = "dashboard",
    out_path: Path | None = None,
    legacy: bool = False,
) -> None:
    """Stage 7: render dashboard (target=dashboard) or branded report (target=report).

    Pass ``--legacy`` to launch the six-view v1 dashboard preserved at
    ``engine/render/dashboard/app_v1.py`` for side-by-side comparison with the
    rebuilt four-view dashboard.
    """
    if target == "dashboard":
        import subprocess
        app_path = (
            "engine/render/dashboard/app_v1.py" if legacy
            else "engine/render/dashboard/app.py"
        )
        subprocess.run(
            ["streamlit", "run", app_path, "--", "--db", str(db)],
            check=True,
        )
    elif target == "report":
        from engine.render.report.generate import write_report
        out = out_path or (db.parent / "mitcham-buildings-renewal-outlook.html")
        write_report(db_path=db, out_path=out)
        typer.echo(f"wrote report to {out}")
    else:
        raise typer.BadParameter(f"unknown render target {target!r} (use 'dashboard' or 'report')")


@app.command()
def pipeline(db: Path = DEFAULT_DB, horizon: int = 25, n: int = 200,
             annual_budget: float = DEMO_ANNUAL_BUDGET, n_scenarios: int = 40,
             granularity: str = "building", seed: int = 20260527) -> None:
    """Run the full pipeline end to end (all stages -> branded report)."""
    from engine.pipeline import run_pipeline
    out = run_pipeline(db_path=db, horizon=horizon, n_realisations=n,
                       annual_budget=annual_budget, n_scenarios=n_scenarios,
                       granularity=granularity, seed=seed)
    h = out["headline"]
    typer.echo(
        f"pipeline complete: {out['n_components']} components, {out['mc_paths']} mc rows, "
        f"{out['opt_units']} units optimised. Unfunded liability (RCP4.5) p50 "
        f"${h['gap_p50']/1e6:.1f}M (demand ${h['demand_p50']/1e6:.1f}M vs capacity "
        f"${h['capacity']/1e6:.0f}M). Report: {out['report']}"
    )


if __name__ == "__main__":
    app()
