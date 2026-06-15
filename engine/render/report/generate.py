"""Generate the Mitcham-branded buildings renewal-outlook HTML report.

Pulls the headline unfunded-liability figure from :func:`engine.render.metrics.
unfunded_liability` (never recomputed here), the works programme from
``opt_summary``, and renders a four-page A4 report from a Jinja2 template with
the brand stylesheet inlined. Output is decision-support, not decision-authority.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import duckdb
from jinja2 import Environment, FileSystemLoader, select_autoescape

from engine.render.metrics import unfunded_liability

_HERE = Path(__file__).parent
_MOTTO = "POSTERIS AEDIFICEMUS"
_SCENARIO_LABELS = {
    "no_climate": "No climate change",
    "rcp45": "RCP 4.5 (moderate warming)",
    "rcp85": "RCP 8.5 (high warming)",
}
# Gap counts as "material" for the narrative once it clears this floor.
_MATERIAL_GAP = 1_000_000.0


def _fmt_money(value: float | None) -> str:
    """Compact AUD as ``$X.YM`` (millions); falls back to ``$X.Yk`` / ``$X``."""
    if value is None:
        return "n/a"
    v = float(value)
    if abs(v) >= 1_000_000:
        return f"${v / 1_000_000:,.1f}M"
    if abs(v) >= 1_000:
        return f"${v / 1_000:,.0f}k"
    return f"${v:,.0f}"


def _table_is_empty(conn: duckdb.DuckDBPyConnection, table: str) -> bool:
    """True if ``table`` is missing or holds no rows."""
    try:
        return conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 0
    except duckdb.Error:
        return True


def _fetch_works(conn: duckdb.DuckDBPyConnection) -> list[dict]:
    """Top ~20 buildings: renew_year_p50 present, robust DESC then year ASC.

    The optimiser solves a single two-stage *stochastic* programme spanning all
    climate scenarios, so ``opt_summary.scenario`` is always ``'stochastic'`` —
    the works programme is not filtered by the climate scenario used for the
    headline (mirrors the dashboard's behaviour).
    """
    if _table_is_empty(conn, "opt_summary"):
        return []
    cur = conn.execute(
        """
        SELECT COALESCE(a.name, o.asset_id) AS asset_id,
               o.renew_year_p50, o.renew_share, o.robust, o.mean_cost
        FROM opt_summary o
        LEFT JOIN (
            SELECT DISTINCT split_part(asset_id, '-', 1) AS bid, name FROM assets
        ) a ON o.asset_id = a.bid
        WHERE o.renew_year_p50 IS NOT NULL AND o.scenario = 'stochastic'
        ORDER BY o.robust DESC, o.renew_year_p50 ASC
        LIMIT 20
        """
    )
    cols = [d[0] for d in cur.description]
    works: list[dict] = []
    for row in cur.fetchall():
        rec = dict(zip(cols, row))
        share = rec.get("renew_share")
        rec["renew_share_pct"] = f"{float(share) * 100:.0f}%" if share is not None else "n/a"
        rec["mean_cost_fmt"] = _fmt_money(rec.get("mean_cost"))
        rec["robust"] = bool(rec.get("robust"))
        works.append(rec)
    return works


def _build_context(
    conn: duckdb.DuckDBPyConnection,
    *,
    scenario: str,
    council_name: str,
    annual_budget: float,
    horizon: int,
) -> dict:
    """Assemble the full template context from the DB (headline + works + meta)."""
    if _table_is_empty(conn, "mc_paths"):
        headline = None
    else:
        headline = unfunded_liability(
            conn, scenario=scenario, horizon=horizon, annual_budget=annual_budget
        )

    label = _SCENARIO_LABELS.get(scenario, scenario)
    ctx: dict = {
        "council_name": council_name,
        "motto": _MOTTO,
        "horizon": horizon,
        "scenario": scenario,
        "scenario_label": label,
        "scenario_label_lower": label[0].lower() + label[1:] if label else label,
        "annual_budget": _fmt_money(annual_budget),
        "generated_on": date.today().strftime("%d/%m/%Y"),
        "end_year": 2026 + horizon,
        "headline": headline,
        "works": _fetch_works(conn),
    }

    if headline is not None:
        ctx.update(
            {
                "capacity": _fmt_money(headline["capacity"]),
                "demand_p05": _fmt_money(headline["demand_p05"]),
                "demand_p50": _fmt_money(headline["demand_p50"]),
                "demand_p95": _fmt_money(headline["demand_p95"]),
                "gap_p05": _fmt_money(headline["gap_p05"]),
                "gap_p50": _fmt_money(headline["gap_p50"]),
                "gap_p95": _fmt_money(headline["gap_p95"]),
                "gap_is_material": headline["gap_p50"] >= _MATERIAL_GAP,
            }
        )
    return ctx


def write_report(
    db_path: str | Path,
    out_path: str | Path,
    scenario: str = "rcp45",
    council_name: str = "City of Mitcham",
    annual_budget: float = 1_200_000.0,
    horizon: int = 25,
) -> Path:
    """Render the renewal-outlook report to ``out_path`` (UTF-8 HTML).

    Returns the path written. The CSS is inlined into the template via
    ``{{ stylesheet }}`` so the file is fully self-contained.
    """
    out = Path(out_path)
    conn = duckdb.connect(str(db_path))
    try:
        ctx = _build_context(
            conn,
            scenario=scenario,
            council_name=council_name,
            annual_budget=annual_budget,
            horizon=horizon,
        )
    finally:
        conn.close()

    stylesheet = (_HERE / "styles.css").read_text(encoding="utf-8")
    env = Environment(
        loader=FileSystemLoader(str(_HERE)),
        autoescape=select_autoescape(["html", "j2"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template("template.html.j2")
    html = template.render(stylesheet=stylesheet, **ctx)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out
