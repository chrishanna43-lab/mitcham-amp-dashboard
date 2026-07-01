"""City of Mitcham — Buildings Renewal Outlook dashboard.

Headless mode (``render_dashboard(..., headless=True)``) gathers a ``panels``
dict straight from DuckDB and returns it WITHOUT importing Streamlit, so the
test suite runs without a display. Interactive mode imports Streamlit and
Plotly lazily inside the render path and draws a Mitcham-branded report.

Run interactively::

    streamlit run engine/render/dashboard/app.py -- --db <path-to.duckdb>
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb

from engine.config import DEFAULT_DB, HORIZON_STRESS
from engine.render.metrics import (
    ARFR_BAND,
    ASR_BAND,
    GRADE_NUMBER,
    acr_status,
    arfr_compliance,
    backlog_status,
    band_status,
    climate_exposure_value,
    condition_grade,
    condition_grade_number,
    financial_indicators,
    gap_trajectory,
    minimum_sustainable_budget,
    report_card,
    unfunded_liability,
)
from engine.render.programme import (
    building_card,
    deferral_impact,
    deferred_buildings,
    force_into_programme,
    funded_programme,
    spend_by_year,
)
from engine.render.timing import deferral_timing

CURRENT_YEAR = 2026

# "Bloom" pastel palette (dusty lavender + sage) — the refreshed look. The
# product remains the City of Mitcham demo; only the colour system changed.
# Semantic colours (good/watch/act) are kept accessible and are always paired
# with a glyph + word in the UI; the embers ramp is reserved for climate/hazard.
INK = "#2d293a"
DEEP = "#4a4567"          # deep structural tone — brand bar, heading emphasis
ACCENT = "#897eb0"        # primary accent (lavender)
ACCENT_DEEP = "#6e639b"
ACCENT_2 = "#bcb2d6"
RULE = "#c79fb6"          # soft mauve rule accent — never text, never data
GOOD = "#3f8a6b"
WATCH = "#b07a12"
ACT = "#c0603a"
EMBERS = ("#e7d9ad", "#e0a85a", "#cf7233", "#9a3b2e", "#6c2742")
# Continuous colourscales (Plotly form) — lavender for temporal data, the embers
# ramp for climate/condition heat. Replaces Viridis / YlOrRd / RdYlGn (the last
# being colour-blind-unsafe) so the maps match the Bloom palette.
LAVENDER_SCALE = [[0.0, "#6e639b"], [0.5, "#a99ec9"], [1.0, "#e2dcef"]]
EMBERS_SCALE = [[0.0, "#e7d9ad"], [0.25, "#e0a85a"], [0.5, "#cf7233"], [0.75, "#9a3b2e"], [1.0, "#6c2742"]]
# Back-compat aliases (legacy names still referenced across the module).
MITCHAM_GREEN = DEEP
BRIGHT_GREEN = ACCENT
GOLD = RULE
CHARCOAL = INK
MOTTO = "POSTERIS AEDIFICEMUS"
SCENARIOS = ("no_climate", "rcp45", "rcp85")
SCENARIO_LABELS = {
    "no_climate": "No climate change",
    "rcp45": "RCP 4.5 (moderate)",
    "rcp85": "RCP 8.5 (high)",
}


def _table_is_empty(conn: duckdb.DuckDBPyConnection, table: str) -> bool:
    """True if ``table`` is missing or holds no rows (every panel guards on this)."""
    try:
        return conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 0
    except duckdb.Error:
        return True


def _fetch_dicts(conn: duckdb.DuckDBPyConnection, sql: str, params: list) -> list[dict]:
    """Run a query and return rows as a list of column->value dicts."""
    cur = conn.execute(sql, params)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def _building_frame(conn: duckdb.DuckDBPyConnection) -> list[dict]:
    """Per-building rows for the map: id, name, suburb, lat/lon, value, current
    condition, A–F grade, and the optimiser's renewal year / robust flag.
    """
    if _table_is_empty(conn, "assets"):
        return []
    rows = _fetch_dicts(
        conn,
        """
        SELECT b.building_id, b.name, b.suburb, b.lat, b.lon, b.value, b.condition,
               o.renew_year_p50 AS renew_year, o.robust
        FROM (
            SELECT split_part(asset_id, '-', 1) AS building_id,
                   MAX(name) AS name, MAX(suburb) AS suburb,
                   AVG(lat) AS lat, AVG(lon) AS lon,
                   SUM(grc) AS value, MAX(condition) AS condition
            FROM assets WHERE council = 'mitcham'
            GROUP BY building_id
        ) b
        LEFT JOIN opt_summary o ON o.asset_id = b.building_id AND o.scenario = 'stochastic'
        """,
        [],
    )
    for r in rows:
        r["grade"] = condition_grade(float(r["condition"]))
        r["robust"] = bool(r["robust"]) if r["robust"] is not None else False
    return rows


def _suburb_summary(buildings: list[dict]) -> list[dict]:
    """Aggregate the building frame to suburb level for the choropleth: count,
    value, value-weighted condition, A–F grade and share of buildings in backlog.
    """
    agg: dict[str, dict] = {}
    for b in buildings:
        suburb = b.get("suburb")
        if not suburb:
            continue
        a = agg.setdefault(
            suburb,
            {"suburb": suburb, "n_buildings": 0, "value": 0.0, "wcond": 0.0, "backlog": 0},
        )
        value = float(b["value"])
        condition = float(b["condition"])
        a["n_buildings"] += 1
        a["value"] += value
        a["wcond"] += condition * value
        if condition >= 4.0:
            a["backlog"] += 1
    out: list[dict] = []
    for a in agg.values():
        cond = a["wcond"] / a["value"] if a["value"] else 0.0
        out.append(
            {
                "suburb": a["suburb"], "n_buildings": a["n_buildings"],
                "value": a["value"], "condition": cond, "grade": condition_grade(cond),
                "backlog_share": a["backlog"] / a["n_buildings"] if a["n_buildings"] else 0.0,
            }
        )
    return sorted(out, key=lambda r: r["value"], reverse=True)


def _gather_panels(
    conn: duckdb.DuckDBPyConnection,
    scenario: str,
    annual_budget: float,
    horizon: int,
) -> dict:
    """Assemble every dashboard panel from the DB (no Streamlit dependency).

    Keys: headline, summary, works, buildings, suburbs, report_card,
    gap_trajectory, climate. Each guards on empty tables so the gatherer is safe
    to call against a freshly bootstrapped (empty) database.
    """
    has_mc = not _table_is_empty(conn, "mc_paths")
    headline = (
        unfunded_liability(conn, scenario=scenario, horizon=horizon, annual_budget=annual_budget)
        if has_mc else None
    )

    if _table_is_empty(conn, "mc_summary"):
        summary: list[dict] = []
    else:
        summary = _fetch_dicts(
            conn,
            """
            SELECT scenario, year, funding_gap_p05, funding_gap_p50, funding_gap_p95,
                   avg_condition, breach_share
            FROM mc_summary
            WHERE scenario = ?
            ORDER BY year
            """,
            [scenario],
        )

    if _table_is_empty(conn, "opt_summary"):
        works: list[dict] = []
    else:
        works = _fetch_dicts(
            conn,
            """
            SELECT COALESCE(a.name, o.asset_id) AS building,
                   a.suburb,
                   o.renew_year_p50 AS renewal_year,
                   o.renew_share,
                   o.robust,
                   o.mean_cost
            FROM opt_summary o
            LEFT JOIN (
                SELECT DISTINCT split_part(asset_id, '-', 1) AS bid, name, suburb FROM assets
            ) a ON o.asset_id = a.bid
            WHERE o.scenario = 'stochastic'
            ORDER BY o.robust DESC, o.renew_year_p50 ASC NULLS LAST
            LIMIT 25
            """,
            [],
        )

    buildings = _building_frame(conn)
    return {
        "headline": headline,
        "summary": summary,
        "works": works,
        "buildings": buildings,
        "suburbs": _suburb_summary(buildings),
        "report_card": report_card(conn) if not _table_is_empty(conn, "assets") else None,
        "gap_trajectory": (
            gap_trajectory(conn, scenario, horizon, annual_budget) if has_mc else None
        ),
        "climate": (
            climate_exposure_value(conn, scenario, horizon)
            if not _table_is_empty(conn, "climate_exposure") else None
        ),
        # Keyword args are MANDATORY for horizon/annual_budget — positional order
        # silently swaps them (→ horizon=annual_budget, a garbage screen, no raise).
        "financial_indicators": (
            financial_indicators(conn, scenario, horizon=horizon, annual_budget=annual_budget)
            if has_mc and not _table_is_empty(conn, "assets") else None
        ),
        "arfr_compliance": (
            arfr_compliance(conn, scenario, horizon=horizon, annual_budget=annual_budget)
            if has_mc else None
        ),
        "min_budget": (
            minimum_sustainable_budget(conn, scenario, horizon=horizon) if has_mc else None
        ),
    }


def render_dashboard(
    db_path: str | Path = DEFAULT_DB,
    headless: bool = False,
    scenario: str = "rcp45",
    annual_budget: float = 800_000.0,
    horizon: int = 25,
) -> dict:
    """Render (or, headless, gather) the buildings renewal outlook.

    Returns the ``panels`` dict in both modes. In headless mode it never imports
    Streamlit, so it is safe to call without a display.

    Raises ``FileNotFoundError`` if ``db_path`` does not already exist. DuckDB
    silently creates an empty database at any path it is given, which can mask a
    mangled ``--db`` argument (a relative path, a shell that ate the backslashes,
    a typo) by serving an empty dashboard with no errors. Failing loudly here is
    cheaper than a quiet wrong answer.
    """
    db_path = Path(db_path)
    if not db_path.is_file():
        raise FileNotFoundError(
            f"Dashboard DB not found at {db_path!s}. Run `lga-amp pipeline` to "
            f"create it, or pass --db <path-to-existing-file>."
        )
    # [FIX-D1] Open the canonical connection READ-ONLY. The render path is
    # SELECT-only and the sole solve_all call site passes persist=False, so this
    # is safe; it converts any mis-wired write into a loud exception instead of
    # silent corruption of the canonical DB (which the What-If shadow flow relies
    # on as an invariant, not a convention).
    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        panels = _gather_panels(conn, scenario, annual_budget, horizon)
        if headless:
            return panels
        # Thread the canonical PATH (not the read-only conn) into the render path
        # so the What-If view can mint a writable file-copy shadow from it.
        _render_streamlit(conn, panels, scenario, annual_budget, horizon, db_path)
        return panels
    finally:
        conn.close()


def _fmt_money(value: float) -> str:
    """Compact AUD formatting (e.g. $4.2M, $850k)."""
    v = float(value)
    if abs(v) >= 1_000_000:
        return f"${v / 1_000_000:,.1f}M"
    if abs(v) >= 1_000:
        return f"${v / 1_000:,.0f}k"
    return f"${v:,.0f}"


def _inject_brand_css(st) -> None:
    """The 'Bloom' refreshed look — pastel lavender, no framework chrome, a
    branded top bar with horizontal nav (the sidebar is hidden; controls live up
    top). Data stays crisp dark; semantic colours are accessible + glyph-paired."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        :root{
          --ink:#2d293a; --ink-2:#423c52; --muted:#746d82; --muted-2:#9f96ac;
          --bg:#f1eef6; --paper-2:#e9e3f1; --card:#ffffff; --line:#e6e0ee; --line-2:#d9d1e4;
          --deep:#4a4567; --accent:#897eb0; --accent-d:#6e639b; --accent-2:#bcb2d6;
          --accent-soft:rgba(137,126,176,0.16); --mint:#ece5f4; --rule:#c79fb6;
          --good:#3f8a6b; --good-bg:#e2efe8; --watch:#b07a12; --watch-bg:#f6ecd6;
          --act:#c0603a; --act-bg:#f6e6df; --bar:#4a4567;
          --shadow:0 1px 2px rgba(45,41,58,0.05),0 6px 20px rgba(45,41,58,0.07);
        }
        .stApp{ background:var(--bg); color:var(--ink); }
        html, body, [class*="css"], .stMarkdown, p, li, button, input, label, select, textarea{ font-family:'Inter', sans-serif; }
        h1,h2,h3,h4,h5{ font-family:'Inter', sans-serif !important; color:var(--ink) !important; font-weight:700; letter-spacing:-0.015em; }
        .block-container{ max-width:1340px !important; padding:0.4rem 2rem 1rem !important; }
        .kpi-val,.hc-val,.bul-val,.bluf-fig,.dlt-val,[data-testid="stMetricValue"]{ font-variant-numeric:tabular-nums; }

        /* ---- kill framework chrome ---- */
        #MainMenu, [data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"],
        .stDeployButton, [data-testid="stDeployButton"], header[data-testid="stHeader"], footer{ display:none !important; }
        /* ---- hide the sidebar (nav + controls live in the top bar) ---- */
        [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"]{ display:none !important; }

        /* ---- top brand bar ---- */
        .appbar{ background:var(--bar); border-bottom:2px solid var(--rule); border-radius:14px;
          padding:13px 20px; margin:0 0 0.5rem; display:flex; align-items:center; gap:14px; }
        .appbar .nm{ font-weight:700; font-size:1.02rem; color:#fff; line-height:1.15; }
        .appbar .sub{ font-size:0.66rem; letter-spacing:0.18em; text-transform:uppercase; color:rgba(255,255,255,0.6); font-weight:600; }
        .appbar .prep{ margin-left:auto; text-align:right; font-size:0.72rem; color:rgba(255,255,255,0.66); }
        .appbar .prep b{ color:#fff; } .appbar .prep .motto{ font-style:italic; color:var(--accent-2); }
        .appbar .demo-badge{ display:inline-block; margin-left:11px; vertical-align:middle;
          font-size:0.58rem; font-weight:800; letter-spacing:0.15em; text-transform:uppercase;
          color:#fff; background:var(--rule); border-radius:6px; padding:2px 8px; }
        .demo-note{ background:var(--card); border:1px solid var(--rule);
          border-left:5px solid var(--rule); border-radius:12px; padding:0.7rem 1.05rem;
          margin:0 0 0.95rem; font-size:0.85rem; color:var(--ink-2); line-height:1.46; }
        .demo-note .lead{ color:var(--act); font-weight:800; letter-spacing:0.02em; }
        .demo-note b{ color:var(--ink-2); font-weight:700; }

        /* ---- nav + small radios rendered as pills ---- */
        div[role="radiogroup"]{ gap:0.35rem; flex-wrap:wrap; }
        div[role="radiogroup"] > label{ border-radius:999px; padding:0.32rem 0.85rem; border:1px solid var(--line); background:#fff; margin:0 !important; }
        div[role="radiogroup"] > label:hover{ background:var(--mint); }
        div[role="radiogroup"] > label:has(input:checked){ background:var(--accent); border-color:var(--accent); }
        div[role="radiogroup"] > label:has(input:checked) p{ color:#fff !important; }
        div[role="radiogroup"] label p{ font-size:0.86rem !important; font-weight:600; color:var(--ink); }
        /* hide the baseweb radio circle — these read as pills; selection shows via the fill, not a dot */
        div[role="radiogroup"] label > div:first-child{ display:none !important; }
        /* the nav radio sits in .navbar and reads as the primary tab strip */
        .navbar [role="radiogroup"] label{ padding:0.4rem 1.05rem; font-weight:600; }

        /* ---- page header ---- */
        .page-h{ display:flex; justify-content:space-between; align-items:flex-end; gap:1rem; margin:0.4rem 0 1.0rem; }
        .page-h .eyebrow{ font-size:0.78rem; letter-spacing:0.1em; text-transform:uppercase; color:var(--accent-d); font-weight:700; }
        .page-h .ttl{ font-size:1.5rem; font-weight:700; color:var(--ink); margin:0.1rem 0 0; letter-spacing:-0.02em; }
        .page-h .sub{ color:var(--muted); font-size:0.9rem; margin:0.15rem 0 0; }
        .ctx-pills{ display:flex; gap:0.4rem; flex-wrap:wrap; }
        .ctx{ background:#fff; border:1px solid var(--line); border-radius:999px; padding:0.3rem 0.75rem; font-size:0.78rem; font-weight:600; color:var(--accent-d); white-space:nowrap; }

        /* ---- BLUF hero ---- */
        .bluf{ display:grid; grid-template-columns:1.5fr 1fr; overflow:hidden; border:1px solid var(--line);
          border-radius:18px; box-shadow:var(--shadow); background:var(--card); }
        .bluf-main{ padding:1.6rem 1.8rem; }
        .bluf-claim{ font-size:1.18rem; font-weight:600; line-height:1.34; color:var(--ink); }
        .bluf-claim b{ color:var(--deep); font-weight:800; }
        .bluf-figrow{ display:flex; align-items:flex-end; gap:14px; margin-top:1rem; }
        .bluf-fig{ font-size:3.4rem; font-weight:800; line-height:0.9; letter-spacing:-0.03em; color:var(--ink); }
        .bluf-figlabel{ font-size:0.8rem; color:var(--muted); padding-bottom:0.4rem; }
        .rbar{ position:relative; height:13px; border-radius:8px; background:var(--paper-2); margin-top:1.1rem; }
        .rbar .fill{ position:absolute; inset:0; border-radius:8px; background:linear-gradient(90deg,var(--accent-soft),var(--accent),var(--accent-soft)); opacity:0.5; }
        .rbar .mark{ position:absolute; top:-5px; width:3px; height:23px; border-radius:2px; background:var(--deep); }
        .rscale{ display:flex; justify-content:space-between; font-size:0.72rem; color:var(--muted); margin-top:6px; }
        .rcap{ font-size:0.85rem; color:var(--ink-2); margin-top:0.6rem; } .rcap b{ color:var(--accent-d); }
        .rfreq{ font-size:0.79rem; color:var(--muted); margin-top:3px; }
        .bluf-side{ background:linear-gradient(160deg,var(--deep),#6e6499 135%); color:#fff; padding:1.5rem; display:flex; flex-direction:column; gap:14px; }
        .bluf-side .lbl{ font-size:0.66rem; text-transform:uppercase; letter-spacing:0.13em; color:var(--accent-2); font-weight:700; }
        .bluf-side .txt{ font-size:0.9rem; line-height:1.45; margin-top:5px; color:rgba(255,255,255,0.94); } .bluf-side .txt b{ color:#fff; }
        .bluf-pills{ display:flex; gap:6px; flex-wrap:wrap; margin-top:9px; }
        .bpill{ font-size:0.72rem; padding:3px 9px; border-radius:999px; background:rgba(255,255,255,0.16); color:#fff; font-weight:600; }
        .bluf-grade{ margin-top:auto; padding-top:12px; border-top:1px solid rgba(255,255,255,0.16); display:flex; align-items:center; gap:10px; }
        .gchip{ font-size:1.4rem; font-weight:800; color:#fff; width:44px; height:44px; border-radius:11px; display:grid; place-items:center; }
        .gtxt{ font-size:0.8rem; color:rgba(255,255,255,0.86); }

        /* ---- kpi cards ---- */
        .kpi{ background:var(--card); border:1px solid var(--line); border-radius:14px; padding:1rem 1.15rem;
          box-shadow:var(--shadow); height:100%; }
        .kpi-label{ font-size:0.76rem; color:var(--muted); font-weight:600; margin:0 0 0.35rem; }
        .kpi-val{ font-size:1.65rem; font-weight:800; color:var(--ink); line-height:1.05; letter-spacing:-0.02em; }
        .kpi-chip{ display:inline-flex; gap:5px; align-items:center; margin-top:0.55rem; font-size:0.72rem; font-weight:700; padding:0.2rem 0.55rem; border-radius:999px; background:var(--mint); color:var(--accent-d); }
        .kpi-chip.good{ background:var(--good-bg); color:var(--good); }
        .kpi-chip.warn{ background:var(--watch-bg); color:var(--watch); }
        .kpi-chip.bad{ background:var(--act-bg); color:var(--act); }

        /* ---- bullet-strip ratio card ---- */
        .bul{ background:var(--card); border:1px solid var(--line); border-radius:14px; padding:0.95rem 1.1rem; box-shadow:var(--shadow); height:100%; }
        .bul-name{ font-size:0.8rem; font-weight:600; color:var(--ink-2); }
        .bul-val{ font-size:1.55rem; font-weight:800; color:var(--ink); letter-spacing:-0.02em; line-height:1.1; }
        .bul-track{ position:relative; height:9px; border-radius:5px; background:var(--paper-2); margin:0.55rem 0 0.5rem; overflow:hidden; }
        .bul-band{ position:absolute; top:0; bottom:0; background:var(--good-bg); }
        .bul-bar{ position:absolute; top:1.5px; bottom:1.5px; left:0; border-radius:4px; }
        .bul-tgt{ position:absolute; top:-3px; width:2px; height:15px; background:var(--ink); }
        .bul-foot{ display:flex; align-items:center; justify-content:space-between; gap:8px; }
        .bul-meta{ font-size:0.71rem; color:var(--muted); margin-top:0.35rem; }
        .status{ display:inline-flex; gap:5px; align-items:center; font-size:0.71rem; font-weight:700; padding:0.15rem 0.55rem; border-radius:999px; }
        .status.good{ color:var(--good); background:var(--good-bg); }
        .status.warn{ color:var(--watch); background:var(--watch-bg); }
        .status.bad{ color:var(--act); background:var(--act-bg); }

        /* ---- chart card frame + metrics + caption + table ---- */
        [data-testid="stVerticalBlockBorderWrapper"]{ background:var(--card); border:1px solid var(--line) !important;
          border-radius:16px; box-shadow:var(--shadow); padding:0.6rem 0.9rem; }
        [data-testid="stMetric"]{ background:var(--card); border:1px solid var(--line); border-radius:14px;
          padding:0.9rem 1.05rem; box-shadow:var(--shadow); }
        [data-testid="stMetricLabel"] p{ font-size:0.74rem !important; color:var(--muted) !important; font-weight:600; }
        [data-testid="stMetricValue"]{ font-weight:800; color:var(--ink); font-size:1.5rem; }
        [data-testid="stCaptionContainer"] p{ color:var(--muted) !important; font-size:0.8rem; }
        [data-testid="stDataFrame"]{ border:1px solid var(--line); border-radius:12px; overflow:hidden; }
        hr{ border-color:var(--line); }
        ::selection{ background:rgba(137,126,176,0.22); }

        /* ---- portfolio-set option cards ---- */
        .opt{ background:var(--card); border:1px solid var(--line); border-radius:14px; padding:0.9rem; box-shadow:var(--shadow); height:100%; position:relative; }
        .opt.reco{ border-color:var(--rule); }
        .opt .tag{ position:absolute; top:-9px; left:12px; font-size:0.57rem; letter-spacing:0.08em; font-weight:800; text-transform:uppercase; background:var(--rule); color:#fff; padding:2px 7px; border-radius:5px; }
        .opt .onm{ font-size:0.92rem; font-weight:800; color:var(--ink); }
        .opt .odesc{ font-size:0.72rem; color:var(--muted); line-height:1.34; margin:3px 0 9px; min-height:3.5em; }
        .opt .orow{ display:flex; justify-content:space-between; font-size:0.74rem; padding:3px 0; border-top:1px solid var(--line); }
        .opt .orow .l{ color:var(--muted); } .opt .orow .v{ font-weight:700; color:var(--ink); font-variant-numeric:tabular-nums; }
        .opt .orow .v.good{ color:var(--good); } .opt .orow .v.bad{ color:var(--act); }

        /* ---- what-if delta cards ---- */
        .dlt{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:0.85rem 1rem; box-shadow:var(--shadow); height:100%; }
        .dlt-lbl{ font-size:0.74rem; color:var(--muted); font-weight:600; }
        .dlt-val{ font-size:1.35rem; font-weight:800; color:var(--ink); margin-top:2px; }
        .dlt-chg{ font-size:0.74rem; font-weight:700; margin-top:3px; }
        .dlt-chg.good{ color:var(--good); } .dlt-chg.bad{ color:var(--act); } .dlt-chg.flat{ color:var(--muted-2); }
        .dlt-chan{ font-size:0.62rem; text-transform:uppercase; letter-spacing:0.07em; color:var(--muted-2); font-weight:700; margin-top:2px; }

        /* ---- provenance footer + primary buttons ---- */
        .provfoot{ border-top:1px solid var(--line-2); margin-top:1.4rem; padding-top:0.8rem; font-size:0.73rem; color:var(--muted); display:flex; gap:14px; flex-wrap:wrap; }
        .provfoot b{ color:var(--ink-2); }
        .stButton > button[kind="primary"]{ background:var(--accent); border-color:var(--accent); }
        .stButton > button[kind="primary"]:hover{ background:var(--accent-d); border-color:var(--accent-d); }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _sidebar_brand(st) -> None:
    """Wordmark at the top of the sidebar (council branding removed)."""
    st.markdown(
        """
        <div class="sb-brand">
          <div>
            <div class="nm">Buildings Renewal Outlook</div>
            <div class="sub">ASSET MANAGEMENT</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# Traffic-light colours for service grades and hazard series (legibility beats
# brand restraint for a risk read-out; this is the Mitcham dashboard, not SCA).
GRADE_COLORS = {"A": "#3f8a6b", "B": "#3f8a6b", "C": "#7a9a5e", "D": "#b07a12", "F": "#c0603a"}
HAZARD_COLORS = {"heat": "#cf7233", "flood": "#9a3b2e", "bushfire": "#6c2742"}
_MAP_CENTER = {"lat": -35.005, "lon": 138.625}


def _load_suburbs_geojson() -> dict | None:
    """Load the Mitcham suburb boundaries shipped with the dashboard."""
    import json
    path = Path(__file__).parent / "geo" / "mitcham-suburbs.geojson"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


def _kpi_card(st, label: str, value: str, chip: str | None = None, tone: str = "") -> None:
    """A clean white KPI card: small label, big value, optional pill chip."""
    chip_html = f'<span class="kpi-chip {tone}">{chip}</span>' if chip else ""
    st.markdown(
        f'<div class="kpi"><p class="kpi-label">{label}</p>'
        f'<div class="kpi-val">{value}</div>{chip_html}</div>',
        unsafe_allow_html=True,
    )


_STATUS_GLYPH = {"good": ("good", "✓"), "warn": ("warn", "!"), "bad": ("bad", "✕")}


def _bullet_card(st, name, value_disp, value_pct, band, max_scale, target_pct,
                 status, word, meta) -> None:
    """A regulated-ratio bullet strip: value bar + shaded target band + target
    marker + a status chip (glyph + word, colour-blind-safe) + a one-line meta."""
    cls, glyph = _STATUS_GLYPH.get(status, ("", "•"))
    color = {"good": GOOD, "warn": WATCH, "bad": ACT}.get(status, ACCENT)
    lo, hi = band
    st.markdown(
        f'<div class="bul"><div class="bul-name">{name}</div>'
        f'<div class="bul-val">{value_disp}</div>'
        f'<div class="bul-track">'
        f'<div class="bul-band" style="left:{lo / max_scale * 100:.0f}%;width:{(hi - lo) / max_scale * 100:.0f}%"></div>'
        f'<div class="bul-bar" style="width:{min(100.0, value_pct / max_scale * 100):.0f}%;background:{color}"></div>'
        f'<div class="bul-tgt" style="left:{target_pct / max_scale * 100:.0f}%"></div></div>'
        f'<div class="bul-foot"><span class="status {cls}">{glyph} {word}</span></div>'
        f'<div class="bul-meta">{meta}</div></div>',
        unsafe_allow_html=True,
    )


def _hero_card(st, headline: dict, scenario: str, budget: float, grade: str | None, horizon: int) -> None:
    """BLUF hero: the headline gap as one number, with a designed p05–p95 honesty
    range and a plain-English read — the visible leapfrog over a single-line forecast."""
    lo, mid, hi = headline["gap_p05"], headline["gap_p50"], headline["gap_p95"]
    pos = (mid - lo) / (hi - lo) * 100 if hi > lo else 50.0
    pos = max(4.0, min(96.0, pos))
    grade_color = GRADE_COLORS.get(grade or "", ACCENT)
    grade_num = GRADE_NUMBER.get(grade or "", grade)
    grade_block = (
        f'<div class="bluf-grade"><div class="gchip" style="background:{grade_color}">{grade_num}</div>'
        f'<div class="gtxt"><b style="color:#fff">Portfolio grade {grade_num} of 5.</b><br>'
        "The funded programme below is what closes the gap.</div></div>"
    ) if grade else ""
    st.markdown(
        f"""
        <div class="bluf">
          <div class="bluf-main">
            <p class="bluf-claim">Mitcham's buildings carry a <b>{_fmt_money(mid)} renewal shortfall</b>
              over {horizon} years at the current {_fmt_money(budget)} a year.</p>
            <div class="bluf-figrow">
              <div class="bluf-fig">{_fmt_money(mid)}</div>
              <div class="bluf-figlabel">unfunded renewal gap<br>{horizon} years &middot; central estimate</div>
            </div>
            <div class="rbar"><div class="fill"></div><div class="mark" style="left:{pos:.0f}%"></div></div>
            <div class="rscale"><span>{_fmt_money(lo)}</span><span>central {_fmt_money(mid)}</span><span>{_fmt_money(hi)}</span></div>
            <p class="rcap">We're <b>90% sure</b> the gap sits between <b>{_fmt_money(lo)} and {_fmt_money(hi)}</b>.</p>
            <p class="rfreq">In about 19 of 20 modelled futures the shortfall stays under {_fmt_money(hi)} —
              the range behind the headline, not a single line.</p>
          </div>
          <aside class="bluf-side">
            <div>
              <div class="lbl">Under this strategy</div>
              <p class="txt">Renewal demand <b>{_fmt_money(headline['demand_p50'])}</b> against funded
                capacity <b>{_fmt_money(headline['capacity'])}</b> over {horizon} years.</p>
              <div class="bluf-pills">
                <span class="bpill">{SCENARIO_LABELS.get(scenario, scenario)}</span>
                <span class="bpill">{_fmt_money(budget)} / yr</span>
              </div>
            </div>
            {grade_block}
          </aside>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _financial_indicators_strip(st, go, fi) -> None:
    """Four-card SA local-government asset-ratio strip for THE CALL.

    ARFR (mandated, two-sided band), ACR (one-sided — high is healthy), ASR
    (national depreciation cross-check), and the renewal backlog. Each card's RAG
    chip is driven by the matching status helper in ``metrics``.
    """
    if not fi:
        return
    st.markdown('<div style="height:1.0rem"></div>', unsafe_allow_html=True)
    st.markdown("#### Financial sustainability indicators")
    st.caption(
        "SA local-government asset ratios derived from the renewal model. "
        "ARFR is the mandated I&AMP-based indicator (FSIP No. 9 target "
        "80–120%, rolling 100%); ASR is the national depreciation-based "
        "cross-check.")
    arfr, acr, asr, backlog = fi.get("arfr"), fi.get("acr"), fi.get("asr"), fi.get("backlog")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        s = band_status(arfr, ARFR_BAND)
        _bullet_card(st, "Asset Renewal Funding Ratio",
                     f"{arfr * 100:.0f}%" if arfr is not None else "—",
                     (arfr if arfr is not None else 0) * 100,
                     (80, 120), 140, 100, s,
                     {"good": "In band", "warn": "Watch", "bad": "Below floor"}.get(s, "—"),
                     "target 80–120% · rolling 100%")
    with c2:
        s = acr_status(acr)   # one-sided; high is NOT bad
        _bullet_card(st, "Asset Consumption Ratio",
                     f"{acr * 100:.0f}%" if acr is not None else "—",
                     (acr if acr is not None else 0) * 100,
                     (40, 100), 100, 40, s,
                     {"good": "Healthy", "warn": "Watch", "bad": "Low"}.get(s, "—"),
                     "share of value remaining — higher = younger stock")
    with c3:
        s = band_status(asr, ASR_BAND)
        _bullet_card(st, "Asset Sustainability Ratio",
                     f"{asr * 100:.0f}%" if asr is not None else "—", (asr or 0) * 100,
                     (90, 110), 140, 100, s,
                     {"good": "On track", "warn": "Watch", "bad": "Below"}.get(s, "—"),
                     "scheduled renewal vs depreciation · target 90–110%")
    with c4:
        s = backlog_status(backlog)
        _bullet_card(st, "Renewal backlog",
                     f"{backlog * 100:.0f}%" if backlog is not None else "—", (backlog or 0) * 100,
                     (0, 5), 60, 5, s,
                     {"good": "Low", "warn": "Watch", "bad": "High"}.get(s, "—"),
                     "% of value past target · target <5%")

    with st.expander("How these ratios are derived (method notes)", expanded=False):
        st.markdown(
            "- **Straight-line depreciation** assumption for ACR/ASR (age vs useful "
            "life); calibratable if Mitcham shares valuation history.\n"
            "- **ACR** is a straight-line age-vs-life proxy (the demo register holds "
            "no independent valuation) — deliberately different from the convex, "
            "sampled condition curve used for renewal *need*. A young portfolio reads "
            "ACR ~95–100%, which is **healthy, not a red flag** (hence the one-sided "
            "status).\n"
            "- **ASR is works-based**: the numerator is the optimiser's *scheduled* "
            "renewal (`opt_summary`), not the budget cap — a budget-based ASR is "
            "circular (`annual_budget ÷ depreciation`). Read it as scheduled renewal "
            "vs depreciation.\n"
            "- **ARFR numerator** = funded capacity (`annual_budget × horizon`) — "
            "committed funding, not optimiser spend. The **denominator is modelled "
            "renewal need**: where a council's AMP under-proposes against need, its "
            "reported ARFR can look healthy while this need-based ARFR reveals the true "
            "gap. The corridor axis is **cumulative-to-date** ARFR, not the trailing "
            "rolling average FSIP No. 9 prescribes.\n"
            "- **Sources:** definitions/targets → LGA SA FSIP No. 9 and the Model "
            "Financial Statements (Financial Indicators note); council-specific "
            "commentary → ESCOSA, which has a legislated advisory role on council "
            "financial sustainability following the 2021 LG reforms. Decision support, "
            "not authority.")


def _compliance_corridor(st, go, comp, minb) -> None:
    """Rolling-ARFR compliance corridor: p50 + p95 against the 80–120% band.

    Above the chart, the compliance verdict and the minimum-sustainable-budget
    cards (average-100% and never-below-80%, p50 and p95) from ``minb``.
    """
    if not comp:
        return
    lo, hi = comp["band"]

    # The cumulative ARFR ramps up from a low early value (year-1 budget against
    # the first lumpy need) toward its endpoint — the headline ARFR (cumulative
    # funding / cumulative need over the whole plan). Judge compliance on that
    # MATURE endpoint, not the ramp: a "below floor in 2026" reading is an artefact
    # of the cumulative measure, not under-funding.
    years, p50 = comp["years"], comp["arfr_p50"]
    final = next((v for v in reversed(p50) if v is not None), None)
    last_year = years[-1] if years else None
    if final is None:
        pass
    elif final < lo:
        st.warning(
            f"By {last_year} cumulative ARFR reaches only {final * 100:.0f}% — below "
            f"the {lo * 100:.0f}% floor. Renewal is under-funded across the plan.")
    elif final > hi:
        st.info(
            f"By {last_year} cumulative ARFR reaches {final * 100:.0f}% — above the "
            f"{hi * 100:.0f}% band. Funded capacity exceeds modelled renewal need at "
            f"this budget and scenario.")
    else:
        st.success(
            f"By {last_year} cumulative ARFR reaches {final * 100:.0f}% — within the "
            f"{lo * 100:.0f}–{hi * 100:.0f}% band.")

    if minb:
        p50 = minb.get("p50", {})
        p95 = minb.get("p95", {})
        b1, b2 = st.columns(2)
        with b1:
            _kpi_card(st, "Budget to average 100%",
                      f"{_fmt_money(p50.get('target_budget', 0.0))}/yr",
                      f"p95 contingency {_fmt_money(p95.get('target_budget', 0.0))}/yr")
        with b2:
            _kpi_card(st, "Budget to never drop below 80%",
                      f"{_fmt_money(p50.get('floor_budget', 0.0))}/yr",
                      f"p95 contingency {_fmt_money(p95.get('floor_budget', 0.0))}/yr",
                      tone="warn")

    fig = go.Figure()
    # shaded 80–120% compliance band + 100% target line
    fig.add_hrect(y0=lo * 100, y1=hi * 100, fillcolor="rgba(26,122,58,0.10)", line_width=0)
    fig.add_hline(y=100, line=dict(color="#1a7a3a", width=1, dash="dash"),
                  annotation_text="target 100%", annotation_position="top left")
    fig.add_trace(go.Scatter(
        x=comp["years"], y=[v * 100 if v else None for v in comp["arfr_p50"]],
        name="ARFR (central)", line=dict(color=BRIGHT_GREEN, width=3)))
    fig.add_trace(go.Scatter(
        x=comp["years"], y=[v * 100 if v else None for v in comp["arfr_p95"]],
        name="ARFR (high demand, p95)", line=dict(color="#e8590c", width=2, dash="dot")))
    # Y-range adapts to the data: a 160% floor keeps the 80–120% band well-framed
    # at normal budgets, but it expands when ARFR climbs higher (funded capacity
    # outpacing modelled need at high budgets) so the lines are never clipped.
    vals = [v * 100 for v in (comp["arfr_p50"] + comp["arfr_p95"]) if v is not None]
    ymax = max(160.0, (max(vals) if vals else 0.0) * 1.1)
    fig.update_layout(
        margin=dict(l=8, r=8, t=8, b=8), height=320,
        yaxis=dict(title="Asset Renewal Funding Ratio (%)", ticksuffix="%",
                   range=[0, ymax]),
        legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Cumulative-to-date ARFR (cumulative funding ÷ cumulative renewal need) at the "
        "central (p50) and high-demand (p95) paths, against the regulated 80–120% band. "
        "The line ramps up from a low early value toward its endpoint — the headline "
        "ARFR for the plan — so the verdict above reads the mature endpoint, not the "
        "ramp. The p50–p95 gap is the contingency for demand uncertainty. Decision "
        "support, not authority.")


def _tab_overview(st, go, pd, panels: dict, scenario: str, budget: float, horizon: int) -> None:
    """Executive overview: KPI cards, the headline gap feature card, and trends."""
    headline = panels["headline"]
    card = panels["report_card"]
    if not headline:
        st.info("No simulation data yet — run the pipeline to populate the outlook.")
        return
    grade = card["portfolio_grade"] if card else None
    n_b = card["n_buildings"] if card else 0
    backlog = card["grade_count"]["F"] if card else 0

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        _kpi_card(st, f"Renewal demand · {horizon} yr", _fmt_money(headline["demand_p50"]), "p50 (central)")
    with k2:
        _kpi_card(st, "Funded capacity", _fmt_money(headline["capacity"]), f"{_fmt_money(budget)}/yr")
    with k3:
        _kpi_card(st, "Portfolio value", _fmt_money(card["total_value"]) if card else "—",
                  f"{n_b} buildings" if card else None)
    with k4:
        _kpi_card(st, "Buildings in backlog", str(backlog),
                  f"{backlog / n_b * 100:.0f}% of portfolio" if n_b else None,
                  tone="bad" if backlog else "")

    st.markdown('<div style="height:0.9rem"></div>', unsafe_allow_html=True)
    _hero_card(st, headline, scenario, budget, grade, horizon)
    st.markdown('<div style="height:0.9rem"></div>', unsafe_allow_html=True)

    gt = panels["gap_trajectory"]
    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            st.markdown("**The cost of doing nothing**")
            if gt and gt["years"]:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=gt["years"], y=gt["capacity"], name="Funded capacity",
                    line=dict(color=BRIGHT_GREEN, width=3)))
                fig.add_trace(go.Scatter(
                    x=gt["years"], y=gt["demand_p50"], name="Renewal bill if deferred",
                    line=dict(color="#d6212b", width=3), fill="tonexty",
                    fillcolor="rgba(214,33,43,0.13)"))
                fig.update_layout(
                    margin=dict(l=8, r=8, t=8, b=8), height=300,
                    legend=dict(orientation="h", y=-0.2))
                st.plotly_chart(fig, width="stretch")
    with right:
        with st.container(border=True):
            st.markdown("**Share of asset base below target condition (%)**")
            summary = panels["summary"]
            if summary:
                df = pd.DataFrame(summary)
                fig2 = go.Figure(go.Scatter(
                    x=df["year"], y=df["breach_share"] * 100.0,
                    line=dict(color="#e8590c", width=3), fill="tozeroy",
                    fillcolor="rgba(232,89,12,0.12)"))
                fig2.update_layout(margin=dict(l=8, r=8, t=8, b=8), height=300)
                st.plotly_chart(fig2, width="stretch")


def _tab_map(st, go, pd, panels: dict) -> None:
    """Spatial view: suburb risk choropleth + buildings by grade and value."""
    buildings = panels["buildings"]
    suburbs = panels["suburbs"]
    if not buildings:
        st.info("No buildings yet — run the pipeline (and `lga-amp geo`).")
        return
    geojson = _load_suburbs_geojson()
    bdf = pd.DataFrame(buildings)
    sdf = pd.DataFrame(suburbs)

    shade = st.radio("Shade suburbs by", ["Backlog share", "Average condition"], horizontal=True)
    fig = go.Figure()
    if geojson is not None and not sdf.empty:
        z = sdf["backlog_share"] if shade == "Backlog share" else sdf["condition"]
        fig.add_trace(go.Choroplethmap(
            geojson=geojson, locations=sdf["suburb"], featureidkey="properties.name",
            z=z, colorscale="YlOrRd", marker_opacity=0.4, marker_line_width=0.5,
            marker_line_color="#888", showscale=True, colorbar_title=shade,
            text=sdf["suburb"], hovertemplate="%{text}<br>" + shade + ": %{z:.2f}<extra></extra>"))

    bdf["color"] = bdf["grade"].map(GRADE_COLORS)
    vmax = float(bdf["value"].max()) or 1.0
    hover = [
        f"{r['name']} ({r['suburb']})<br>Grade {r['grade']} · {_fmt_money(r['value'])}"
        + (f"<br>Renewal {int(r['renew_year'])}" if pd.notna(r.get("renew_year")) else "")
        for _i, r in bdf.iterrows()
    ]
    fig.add_trace(go.Scattermap(
        lat=bdf["lat"], lon=bdf["lon"], mode="markers",
        marker=dict(size=(bdf["value"] / vmax * 22 + 6), color=bdf["color"]),
        text=hover, hoverinfo="text", name="buildings"))
    fig.update_layout(
        map_style="carto-positron", map_zoom=10.6, map_center=_MAP_CENTER,
        margin=dict(l=0, r=0, t=8, b=0), height=560, showlegend=False)
    st.plotly_chart(fig, width="stretch")
    st.caption("Dot size = replacement value · colour = service grade (green A → red F). "
               "Buildings are illustrative, placed within their real suburb.")

    options = ["(all suburbs)"] + list(sdf["suburb"]) if not sdf.empty else ["(all suburbs)"]
    pick = st.selectbox("Inspect a suburb", options)
    table = bdf if pick == "(all suburbs)" else bdf[bdf["suburb"] == pick]
    show = table[["name", "suburb", "grade", "value", "condition", "renew_year"]].copy()
    show["value"] = show["value"].map(_fmt_money)
    st.dataframe(show.sort_values("grade"), width="stretch", hide_index=True)


def _timing_panel(st, conn, scenario: str, horizon: int) -> None:
    """Optimal renewal timing: when deferral lowers present-value cost, and when it doesn't.

    Plays the discount rate (favours waiting) against escalation and a service/risk
    carrying cost (favour acting). For each building past intervention, finds the
    renewal year that minimises present-value cost; aggregates how many are best
    deferred versus renewed at breach, and plots the portfolio cost of waiting.
    """
    import plotly.graph_objects as go

    st.markdown("#### Optimal timing — when deferral pays")
    st.caption(
        "Deferring a renewal is not always a cost. A dollar of capital not yet spent earns the "
        "council's discount rate; against that sits the escalation as a deteriorating asset drifts "
        "from renewal toward reconstruction, plus the service and risk cost of carrying it past "
        "intervention. Where the discount rate beats decay, deferral lowers present-value cost — "
        "and the model recommends it.")

    with st.expander("How the three assumptions interact", expanded=False):
        st.markdown(
            "For every building past intervention, the model weighs three forces against each "
            "other on a present-value basis:\n\n"
            "- **Real discount rate (r)** — what a dollar of *uncommitted* capital is worth to "
            "the council each year it stays uncommitted. Higher r ⇒ waiting is more valuable ⇒ "
            "more buildings recommended for deferral. Typical AU real benchmarks: 4–5% "
            "(Productivity Commission, social CBA), 7% (Commonwealth CBA Guidance central "
            "case), 3–10% as sensitivity bands. Use real (not nominal) because the model's "
            "costs are in today's dollars.\n"
            "- **Cost escalation (g)** — how much more a renewal costs each year it is deferred "
            "past intervention, as the asset drifts from renewal toward reconstruction. Higher "
            "g ⇒ waiting is more expensive ⇒ act-at-intervention.\n"
            "- **Carrying cost (% of value/yr)** — service/risk premium the council bears while "
            "the asset sits degraded, weighted by the building's criticality.\n\n"
            "**The crossover.** The model defers while `carry < value × (r − g) / (1 + r)`. So:\n\n"
            "- if `g ≥ r`, the recommendation is always *act at intervention*;\n"
            "- if `r > g` and the carrying cost is low, deferral lowers present-value cost — "
            "and the model says so.\n\n"
            "All three are calibrated, illustrative inputs. Treat outputs as decision support, "
            "not authority — chartered-engineer sign-off required before any real capital "
            "decision.")

    c1, c2, c3 = st.columns(3)
    with c1:
        r = st.slider(
            "Real discount rate (%/yr)", 0.0, 10.0, 5.0, 0.5, key="tim_r",
            help=(
                "The council's real discount rate. A dollar of capital not yet spent "
                "effectively earns this rate each year. Higher = deferral is more attractive. "
                "Australian benchmarks: 4–5% (PC social CBA), 7% (Commonwealth CBA Guidance "
                "central case), 3–10% as sensitivity bands. Real, not nominal."
            ),
        ) / 100.0
    with c2:
        g = st.slider(
            "Cost escalation if deferred (%/yr)", 0.0, 10.0, 3.0, 0.5, key="tim_g",
            help=(
                "How much more a renewal costs for each year it is deferred past intervention, "
                "as decay shifts the job from renewal toward reconstruction. Higher = waiting "
                "loses faster. Calibrated; chartered-engineer sign-off required for real use."
            ),
        ) / 100.0
    with c3:
        carry = st.slider(
            "Carrying cost past intervention (%/yr of value)", 0.0, 10.0, 3.0, 0.5,
            key="tim_carry",
            help=(
                "Service / risk premium the council bears each year the asset sits past "
                "intervention, expressed as a % of replacement value and scaled by the "
                "building's criticality. Higher = waiting hurts more on the service side. "
                "Illustrative."
            ),
        ) / 100.0

    out = deferral_timing(conn, scenario=scenario, horizon=horizon, r=r, g=g, carry_rate=carry)
    if out["n_assets"] == 0:
        st.info("No buildings reach the intervention condition within the horizon at this scenario.")
        return

    k1, k2, k3 = st.columns(3)
    with k1:
        _kpi_card(st, "Buildings where deferral pays", str(out["n_defer"]),
                  f"of {out['n_assets']} due for renewal")
    with k2:
        _kpi_card(st, "Renew at intervention", str(out["n_act_now"]),
                  "escalation/criticality outweighs waiting",
                  tone="bad" if out["n_act_now"] else "")
    with k3:
        _kpi_card(st, "Present-value saving from timing", _fmt_money(out["total_saving"]),
                  "vs renewing every asset at breach", tone="warn")

    if out["curve_tpv"]:
        fig = go.Figure(go.Scatter(
            x=out["curve_k"], y=out["curve_tpv"], mode="lines",
            line=dict(color=BRIGHT_GREEN, width=3), fill="tozeroy",
            fillcolor="rgba(9,117,86,0.10)", name="Portfolio present-value cost"))
        fig.add_vline(x=out["k_star"], line=dict(color="#b8930a", width=2, dash="dash"))
        fig.add_annotation(
            x=out["k_star"], y=out["curve_tpv"][out["k_star"]],
            text=("renew at intervention" if out["k_star"] == 0
                  else f"defer {out['k_star']} yr past intervention"),
            showarrow=True, arrowhead=2, ax=40, ay=-40, font=dict(size=12, color="#01310c"))
        fig.update_layout(
            margin=dict(l=8, r=8, t=8, b=8), height=300, showlegend=False,
            xaxis=dict(title="Years renewal is deferred past each asset's intervention point"),
            yaxis=dict(title="Portfolio present-value cost", tickprefix="$", tickformat="~s"))
        st.plotly_chart(fig, width="stretch")

    movers = [x for x in out["rows"] if x["defer_years"] > 0][:12]
    if movers:
        import pandas as pd
        t = pd.DataFrame([
            {"Building": m["building"], "Suburb": m["suburb"],
             "Value": _fmt_money(m["value"]), "Reaches intervention": m["breach_year"],
             "Optimal renewal": m["optimal_year"], "Defer (yr)": m["defer_years"],
             "PV saving": _fmt_money(m["saving"])}
            for m in movers
        ])
        with st.container(border=True):
            st.markdown("**Buildings where deferring renewal lowers present-value cost**")
            st.dataframe(t, width="stretch", hide_index=True)

    st.caption(
        f"Discount rate is a council/Treasury policy input; escalation ({g * 100:.1f}%/yr) and "
        f"carrying cost ({carry * 100:.1f}%/yr of value, weighted by asset criticality) are "
        "calibrated, illustrative engineering assumptions. They must carry chartered-engineer "
        "sign-off before any real capital decision. Decision support, not authority.")


def _deferral_panel(st, conn, scenario: str, budget: float, horizon: int) -> None:
    """What deferral costs: the unfunded gap as a function of the annual budget.

    Renewal demand is fixed by the asset base; funded capacity is the budget
    cumulated over the horizon. The gap therefore falls linearly as the budget
    rises, closing at the break-even spend (demand / horizon). Every dollar held
    back each year compounds — over the horizon — into the liability deferred onto
    future budgets. No deferral-cost escalation is modelled here (that is phase B).
    """
    import plotly.graph_objects as go

    base = unfunded_liability(conn, scenario=scenario, horizon=horizon, annual_budget=budget)
    demand = base["demand_p50"]
    cur_gap = base["gap_p50"]
    if demand <= 0:
        return
    break_even = demand / horizon
    per_100k = 100_000.0 * horizon  # 25-yr cost of holding back $100k/yr while a gap remains

    st.markdown("#### What deferral costs")
    c1, c2, c3 = st.columns(3)
    with c1:
        _kpi_card(st, f"Unfunded at {_fmt_money(budget)}/yr", _fmt_money(cur_gap),
                  f"over {horizon} years", tone="bad" if cur_gap > 0 else "")
    with c2:
        _kpi_card(st, "Spend that clears the need", f"{_fmt_money(break_even)}/yr",
                  "central (p50) case")
    with c3:
        _kpi_card(st, "Cost of deferring $100k/yr", _fmt_money(per_100k),
                  f"added to the {horizon}-yr bill", tone="warn")

    budgets = [b * 100_000.0 for b in range(2, 81)]  # $0.2M … $8.0M
    gaps = [max(0.0, demand - b * horizon) for b in budgets]
    fig = go.Figure(go.Scatter(
        x=[b / 1_000_000 for b in budgets], y=gaps, mode="lines",
        line=dict(color="#d6212b", width=3), fill="tozeroy",
        fillcolor="rgba(214,33,43,0.12)", name="Unfunded gap"))
    fig.add_vline(x=budget / 1_000_000, line=dict(color=BRIGHT_GREEN, width=2, dash="dash"))
    fig.add_annotation(
        x=budget / 1_000_000, y=cur_gap, text=f"you are here · {_fmt_money(cur_gap)} unfunded",
        showarrow=True, arrowhead=2, ax=50, ay=-40, font=dict(size=12, color="#01310c"))
    # Minimum sustainable (never-below-80%) flat budget — the compliant floor on
    # the slider the user already drives.
    minb = minimum_sustainable_budget(conn, scenario=scenario, horizon=horizon)
    floor_budget = minb.get("p50", {}).get("floor_budget", 0.0)
    if floor_budget > 0:
        fig.add_vline(
            x=floor_budget / 1_000_000, line=dict(color="#b8930a", width=2, dash="dot"),
            annotation_text=f"never below 80% · {_fmt_money(floor_budget)}/yr",
            annotation_position="top right")
    fig.update_layout(
        margin=dict(l=8, r=8, t=8, b=8), height=320, showlegend=False,
        xaxis=dict(title="Annual renewal budget", tickprefix="$", ticksuffix="M"),
        yaxis=dict(title=f"{horizon}-yr unfunded gap", tickprefix="$", tickformat="~s"))
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "A nominal (undiscounted) view of unfunded renewal need, not a verdict that deferral is "
        "always costly. Renewal need is set by the asset base; the council chooses how much to "
        "fund each year, and where annual funding meets the central need the gap closes. This view "
        "does not weigh the time value of money against the cost of decay — deferring a renewal "
        "can be the lower-cost choice once both are counted. The optimal-timing panel below makes "
        "that trade-off explicit. Central (p50) case. Decision support, not authority.")


def _tab_scenarios(st, px, pd, conn, panels: dict, scenario: str, budget: float, horizon: int) -> None:
    """Decision tools: scenario A vs B, and a play-button ageing animation."""
    st.markdown("#### Compare two funding strategies")

    def _side(col, key: str, default_scen: str, default_budget: float) -> None:
        with col:
            scen = st.selectbox(
                f"Scenario {key}", SCENARIOS, key=f"scen_{key}",
                index=SCENARIOS.index(default_scen) if default_scen in SCENARIOS else 1,
                format_func=lambda s: SCENARIO_LABELS.get(s, s))
            bud = 1_000_000.0 * st.slider(
                f"Annual budget {key}", 0.2, 8.0, float(default_budget) / 1_000_000,
                0.1, format="$%.1fM", key=f"bud_{key}")
            h = unfunded_liability(conn, scenario=scen, horizon=horizon, annual_budget=bud)
            st.metric(f"{horizon}-yr unfunded gap", _fmt_money(h["gap_p50"]))
            st.metric("Renewal demand (p50)", _fmt_money(h["demand_p50"]))
            st.metric("Funded capacity", _fmt_money(h["capacity"]))

    a, b = st.columns(2)
    _side(a, "A", scenario, budget)
    _side(b, "B", "rcp85", max(200_000.0, budget * 0.6))

    st.markdown("#### Watch the portfolio age — press play")
    rows = conn.execute(
        "SELECT split_part(asset_id,'-',1) AS bid, year, AVG(condition) AS cond "
        "FROM mc_paths WHERE scenario = ? GROUP BY bid, year ORDER BY year",
        [scenario],
    ).fetchall()
    if not rows or not panels["buildings"]:
        st.info("Run the simulate stage to enable the animation.")
        return
    coords = pd.DataFrame(panels["buildings"])[["building_id", "lat", "lon", "value", "name"]]
    coords = coords.rename(columns={"building_id": "bid"})
    cdf = pd.DataFrame(rows, columns=["bid", "year", "cond"]).merge(coords, on="bid")
    fig = px.scatter_map(
        cdf, lat="lat", lon="lon", color="cond", size="value", animation_frame="year",
        color_continuous_scale=list(EMBERS), range_color=(1.0, 5.0), size_max=20,
        hover_name="name", zoom=10.4, center=_MAP_CENTER, map_style="carto-positron",
        height=560, labels={"cond": "condition"})
    fig.update_layout(margin=dict(l=0, r=0, t=8, b=0))
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Condition under {SCENARIO_LABELS.get(scenario, scenario)} — pale (good) to deep "
               "plum (needs renewal). No intervention modelled: this is the do-nothing trajectory.")


def _tab_climate(st, go, conn, panels: dict, scenario: str) -> None:
    """Climate view: exposed value by hazard over time + a per-year exposure map."""
    cev = panels["climate"]
    if not cev or not cev["years"]:
        st.info("No climate exposure yet — run the climate stage.")
        return
    years = cev["years"]
    fig = go.Figure()
    for hazard, series in cev["by_hazard"].items():
        fig.add_trace(go.Scatter(
            x=years, y=series, name=hazard.title(), stackgroup="one",
            line=dict(width=0.5, color=HAZARD_COLORS.get(hazard, "#888"))))
    fig.update_layout(
        title="Climate-exposed asset value by hazard ($)", plot_bgcolor="white",
        margin=dict(l=10, r=10, t=44, b=10), legend=dict(orientation="h", y=-0.18))
    st.plotly_chart(fig, width="stretch")
    st.caption(
        f"Climate-exposed value rises from {_fmt_money(cev['total'][0])} in {years[0]} to "
        f"{_fmt_money(cev['total'][-1])} in {years[-1]} under "
        f"{SCENARIO_LABELS.get(scenario, scenario)}. Exposure = replacement value under "
        "rising hazard, not an actuarial loss.")

    if panels["buildings"]:
        import pandas as pd
        yr = st.select_slider("Exposure year", options=years, value=years[len(years) // 2])
        rows = conn.execute(
            "SELECT split_part(asset_id,'-',1) AS bid, AVG(intensity) AS ex "
            "FROM climate_exposure WHERE scenario = ? AND year = ? GROUP BY bid",
            [scenario, int(yr)],
        ).fetchall()
        exdf = pd.DataFrame(rows, columns=["bid", "ex"])
        coords = pd.DataFrame(panels["buildings"])[["building_id", "lat", "lon", "value", "name"]]
        coords = coords.rename(columns={"building_id": "bid"})
        mdf = coords.merge(exdf, on="bid", how="left").fillna({"ex": 0.0})
        fig2 = go.Figure(go.Scattermap(
            lat=mdf["lat"], lon=mdf["lon"], mode="markers",
            marker=dict(
                size=(mdf["value"] / (float(mdf["value"].max()) or 1.0) * 20 + 6),
                color=mdf["ex"], colorscale=EMBERS_SCALE, cmin=0.0, cmax=1.0,
                showscale=True, colorbar_title="exposure"),
            text=mdf["name"], hoverinfo="text"))
        fig2.update_layout(
            map_style="carto-positron", map_zoom=10.5, map_center=_MAP_CENTER,
            margin=dict(l=0, r=0, t=8, b=0), height=520)
        st.plotly_chart(fig2, width="stretch")


def _tab_call(st, go, pd, conn, panels: dict, scenario: str, budget: float, horizon: int) -> None:
    """THE CALL — executive summary. The one screen a CEO/finance chair reads first.

    Anchors the same headline figure as v1's Overview, then adds: a near-term
    'recommended this year and next' panel that surfaces specific buildings the
    optimiser would fund, a cost-of-deferral callout that names the unfunded
    figure and the break-even spend, the 25-year cost-of-doing-nothing and
    backlog-share charts, and pointers into the deeper views.
    """
    headline = panels["headline"]
    card = panels["report_card"]
    if not headline:
        st.info("No simulation data yet — run the pipeline to populate the outlook.")
        return
    grade = card["portfolio_grade"] if card else None
    n_b = card["n_buildings"] if card else 0
    backlog = card["grade_count"]["F"] if card else 0

    st.markdown(
        '<div class="demo-note"><span class="lead">Demonstration only.</span> '
        "This dashboard runs on data <b>synthesised from public City of Mitcham "
        "figures</b> — not the council's actual asset register — with climate inputs "
        "<b>calibrated to be representative of</b> NARCliM / CSIRO / BoM ranges, not a "
        "site-specific data pull. It is <b>decision support, not decision authority</b>: "
        "the renewal-timing assumptions require chartered-engineer sign-off before any "
        "real capital decision. Prepared by Social Capital Advisory.</div>",
        unsafe_allow_html=True,
    )

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        _kpi_card(st, f"Renewal demand · {horizon} yr",
                  _fmt_money(headline["demand_p50"]), "p50 (central)")
    with k2:
        _kpi_card(st, "Funded capacity", _fmt_money(headline["capacity"]),
                  f"{_fmt_money(budget)}/yr")
    with k3:
        _kpi_card(st, "Portfolio value",
                  _fmt_money(card["total_value"]) if card else "—",
                  f"{n_b} buildings" if card else None)
    with k4:
        _kpi_card(st, "Buildings in backlog", str(backlog),
                  f"{backlog / n_b * 100:.0f}% of portfolio" if n_b else None,
                  tone="bad" if backlog else "")

    st.markdown('<div style="height:0.9rem"></div>', unsafe_allow_html=True)
    _hero_card(st, headline, scenario, budget, grade, horizon)
    _financial_indicators_strip(st, go, panels.get("financial_indicators"))

    funded = funded_programme(conn, horizon=horizon)
    near_term = sorted(
        [f for f in funded
         if f["renewal_year"] in (CURRENT_YEAR, CURRENT_YEAR + 1)],
        key=lambda f: (
            int(f["renewal_year"]),
            0 if f["robust"] else 1,
            -float(f["mean_cost"] or 0),
        ),
    )[:4]
    st.markdown('<div style="height:1.0rem"></div>', unsafe_allow_html=True)
    st.markdown("#### Recommended for this year and next")
    if not near_term:
        st.caption(
            "The optimiser has no buildings scheduled in the next two years at this "
            "budget and scenario. See THE PLAN for the full programme.")
    else:
        st.caption(
            f"Top renewals the optimiser has committed for {CURRENT_YEAR} and "
            f"{CURRENT_YEAR + 1} at {_fmt_money(budget)}/yr under "
            f"{SCENARIO_LABELS.get(scenario, scenario)}. See THE PLAN for the full programme.")
        cols = st.columns(len(near_term))
        for i, f in enumerate(near_term):
            with cols[i]:
                with st.container(border=True):
                    st.markdown(f"**{f['building']}**")
                    st.caption(f"{f['suburb'] or '—'} · {f['asset_type'] or '—'}")
                    st.markdown(
                        f"**Renew {int(f['renewal_year'])}** · "
                        f"{_fmt_money(f['mean_cost'] or 0)}")
                    bits = []
                    if f["robust"]:
                        bits.append("✓ robust")
                    bits.append(f"value {_fmt_money(f['value'] or 0)}")
                    bits.append(f"criticality {float(f['criticality'] or 0.5):.2f}")
                    st.caption(" · ".join(bits))

    demand = headline["demand_p50"]
    gap = headline["gap_p50"]
    break_even = demand / horizon if horizon else 0.0
    deferred = deferred_buildings(conn, scenario=scenario, horizon=horizon)
    st.markdown('<div style="height:1.0rem"></div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("**Cost of deferral at the current budget**")
        kc1, kc2, kc3 = st.columns(3)
        with kc1:
            _kpi_card(st, f"Unfunded over {horizon} yr",
                      _fmt_money(gap), tone="warn" if gap else "")
        with kc2:
            _kpi_card(st, "Spend that clears the need",
                      f"{_fmt_money(break_even)}/yr", "central case")
        with kc3:
            _kpi_card(st, "Buildings outside the funded horizon",
                      str(len(deferred)),
                      "see THE PLAN → Deferred",
                      tone="warn" if deferred else "")
        st.caption(
            "Nominal, undiscounted view. The Trade-offs → Timing & deferral panel "
            "weighs the discount rate against decay to find where deferral genuinely saves.")

    gt = panels["gap_trajectory"]
    st.markdown('<div style="height:1.0rem"></div>', unsafe_allow_html=True)
    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            st.markdown("**The cost of doing nothing**")
            if gt and gt["years"]:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=gt["years"], y=gt["capacity"], name="Funded capacity",
                    line=dict(color=BRIGHT_GREEN, width=3)))
                fig.add_trace(go.Scatter(
                    x=gt["years"], y=gt["demand_p50"], name="Renewal bill if deferred",
                    line=dict(color="#d6212b", width=3), fill="tonexty",
                    fillcolor="rgba(214,33,43,0.13)"))
                fig.update_layout(
                    margin=dict(l=8, r=8, t=8, b=8), height=300,
                    legend=dict(orientation="h", y=-0.2))
                st.plotly_chart(fig, width="stretch")
    with right:
        with st.container(border=True):
            st.markdown("**Share of asset base below target condition (%)**")
            summary = panels["summary"]
            if summary:
                df = pd.DataFrame(summary)
                fig2 = go.Figure(go.Scatter(
                    x=df["year"], y=df["breach_share"] * 100.0,
                    line=dict(color="#e8590c", width=3), fill="tozeroy",
                    fillcolor="rgba(232,89,12,0.12)"))
                fig2.update_layout(margin=dict(l=8, r=8, t=8, b=8), height=300)
                st.plotly_chart(fig2, width="stretch")

    st.markdown('<div style="height:1.0rem"></div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**Continue to**")
    nc1, nc2, nc3 = st.columns(3)
    with nc1:
        st.markdown(
            "**THE PLAN** — spend by year, the funded-programme Gantt, and the "
            "explicit deferred list with first-breach years")
    with nc2:
        st.markdown(
            "**THE PORTFOLIO** — every council building on the map with selectable "
            "shading, plus a per-building detail card")
    with nc3:
        st.markdown(
            "**THE TRADE-OFFS** — community + elected engagement weighting, "
            "scenario A/B compare, and the optimal-timing economics")


def _render_building_card(st, go, pd, card: dict) -> None:
    """Render a per-building detail card with KPIs, condition trajectory, climate, timing.

    Reused across the dashboard wherever a single building's full context is shown.
    """
    with st.container(border=True):
        st.markdown(f"### {card['name']}")
        st.caption(
            f"{card['suburb'] or '—'} · {card['asset_type'] or '—'} · ID {card['bid']}")

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            _kpi_card(st, "Replacement value", _fmt_money(card['value']))
        with k2:
            _kpi_card(st, "Current grade",
                      str(condition_grade_number(card['condition_now'])),
                      f"1–5 · condition {card['condition_now']:.2f}")
        with k3:
            _kpi_card(st, "Criticality", f"{card['criticality']:.2f}",
                      "1.0 = highest service criticality")
        with k4:
            if card['renew_year']:
                decision = f"Renew {card['renew_year']}"
                sub = _fmt_money(card['mean_cost']) if card['mean_cost'] else None
                tone = ""
            else:
                decision = "Deferred"
                sub = "outside funded horizon"
                tone = "warn"
            _kpi_card(st, "Optimiser decision", decision, sub, tone=tone)

        st.markdown('<div style="height:0.6rem"></div>', unsafe_allow_html=True)

        if card["trajectory"]:
            trj = pd.DataFrame(card["trajectory"])
            tfig = go.Figure()
            tfig.add_trace(go.Scatter(
                x=trj["year"], y=trj["p95"], mode="lines",
                line=dict(width=0), showlegend=False, hoverinfo="skip"))
            tfig.add_trace(go.Scatter(
                x=trj["year"], y=trj["p05"], mode="lines",
                line=dict(width=0), fill="tonexty",
                fillcolor="rgba(232,89,12,0.18)", name="p05–p95",
                hoverinfo="skip"))
            tfig.add_trace(go.Scatter(
                x=trj["year"], y=trj["mean"], mode="lines",
                line=dict(color="#e8590c", width=3), name="expected"))
            tfig.add_hline(
                y=4.0, line=dict(color="#d6212b", width=1, dash="dot"),
                annotation_text="intervention (4.0)", annotation_position="top right")
            tfig.update_layout(
                margin=dict(l=8, r=8, t=8, b=8), height=240,
                xaxis_title="Year",
                yaxis=dict(title="Condition (1 new → 5 failed)", range=[1, 5]),
                legend=dict(orientation="h", y=-0.22))
            with st.container(border=True):
                st.markdown("**Condition trajectory**")
                st.plotly_chart(tfig, width="stretch")

        if card["climate"]:
            cdf = pd.DataFrame(card["climate"])
            cfig = go.Figure()
            for hazard in sorted(cdf["hazard"].unique()):
                sub = cdf[cdf["hazard"] == hazard]
                cfig.add_trace(go.Scatter(
                    x=sub["year"], y=sub["intensity"], mode="lines",
                    name=hazard.title(),
                    line=dict(color=HAZARD_COLORS.get(hazard, "#888"), width=2)))
            cfig.update_layout(
                margin=dict(l=8, r=8, t=8, b=8), height=220,
                xaxis_title="Year",
                yaxis=dict(range=[0, 1], title="Hazard intensity (0–1)"),
                legend=dict(orientation="h", y=-0.22))
            with st.container(border=True):
                st.markdown("**Climate exposure trajectory**")
                st.plotly_chart(cfig, width="stretch")

        if card["timing"] and card["timing"]["t_breach"] is not None:
            tm = card["timing"]
            breach_year = CURRENT_YEAR + tm["t_breach"]
            opt_year = CURRENT_YEAR + tm["t_star"]
            t1, t2, t3 = st.columns(3)
            with t1:
                _kpi_card(st, "Reaches intervention", str(breach_year),
                          "on the central condition path")
            with t2:
                _kpi_card(
                    st, "Optimal renewal year", str(opt_year),
                    f"defer {tm['defer_years']} yr past intervention"
                    if tm["defer_years"] > 0 else "act at intervention")
            with t3:
                _kpi_card(
                    st, "PV saving from timing",
                    _fmt_money(tm["saving"]),
                    "vs renewing at intervention",
                    tone="warn" if tm["saving"] <= 1.0 else "")
            st.caption(
                "Timing layer: discount rate vs decay + criticality-weighted carry. Sliders "
                "controlling these assumptions live under THE TRADE-OFFS → Timing & deferral. "
                "Illustrative; requires chartered-engineer sign-off for real capital decisions.")
        elif card["timing"] is not None:
            st.info("This building does not reach intervention condition within the horizon.")


def _tab_portfolio(st, go, px, pd, conn, panels: dict, scenario: str, horizon: int) -> None:
    """THE PORTFOLIO — spatial overview with selectable shading and per-building drilldown."""
    buildings = panels["buildings"]
    suburbs = panels["suburbs"]
    if not buildings:
        st.info("No buildings yet — run the pipeline (and `lga-amp geo`).")
        return

    bdf = pd.DataFrame(buildings)

    funded = funded_programme(conn, horizon=horizon)
    funded_year = {f["bid"]: int(f["renewal_year"]) for f in funded if f["renewal_year"]}
    bdf["renew_year"] = bdf["building_id"].map(funded_year)

    clim_rows = conn.execute(
        "SELECT split_part(asset_id,'-',1) AS bid, MAX(intensity) AS max_ex "
        "FROM climate_exposure WHERE scenario=? AND year=? GROUP BY bid",
        [scenario, CURRENT_YEAR + horizon - 1],
    ).fetchall()
    clim_map = {bid: float(ex) for bid, ex in clim_rows}
    bdf["climate_ex"] = bdf["building_id"].map(clim_map).fillna(0.0)

    sdf = pd.DataFrame(suburbs) if suburbs else pd.DataFrame()

    shading = st.radio(
        "Shade buildings by",
        ["Renewal year", "Current grade", "Climate exposure (horizon end)"],
        horizontal=True,
    )

    fig = go.Figure()
    geojson = _load_suburbs_geojson()
    if geojson is not None and not sdf.empty:
        fig.add_trace(go.Choroplethmap(
            geojson=geojson, locations=sdf["suburb"], featureidkey="properties.name",
            z=sdf["backlog_share"], colorscale="YlOrRd", marker_opacity=0.30,
            marker_line_width=0.5, marker_line_color="#888", showscale=False,
            hovertemplate="%{location}<br>backlog share: %{z:.2f}<extra></extra>"))

    vmax = float(bdf["value"].max()) or 1.0
    sizes = (bdf["value"] / vmax * 22 + 7).tolist()
    bdf["grade_num"] = bdf["grade"].map(GRADE_NUMBER)
    hover = [
        f"{r['name']} ({r['suburb']})<br>Grade {r['grade_num']} · {_fmt_money(r['value'])}"
        + (f"<br>Renewal {int(r['renew_year'])}" if pd.notna(r.get("renew_year")) else "<br>Deferred this horizon")
        for _i, r in bdf.iterrows()
    ]

    caption = ""
    if shading == "Renewal year":
        has = bdf[bdf["renew_year"].notna()]
        none = bdf[bdf["renew_year"].isna()]
        if not has.empty:
            fig.add_trace(go.Scattermap(
                lat=has["lat"], lon=has["lon"], mode="markers",
                marker=dict(
                    size=(has["value"] / vmax * 22 + 7),
                    color=has["renew_year"], colorscale=LAVENDER_SCALE,
                    cmin=CURRENT_YEAR, cmax=CURRENT_YEAR + horizon - 1,
                    showscale=True, colorbar=dict(title="Renewal year")),
                text=[
                    f"{r['name']} ({r['suburb']})<br>Grade {r['grade_num']} · "
                    f"{_fmt_money(r['value'])}<br>Renewal {int(r['renew_year'])}"
                    for _i, r in has.iterrows()],
                hoverinfo="text", name="funded"))
        if not none.empty:
            fig.add_trace(go.Scattermap(
                lat=none["lat"], lon=none["lon"], mode="markers",
                marker=dict(
                    size=(none["value"] / vmax * 22 + 7),
                    color="#9aa3a8", opacity=0.55),
                text=[
                    f"{r['name']} ({r['suburb']})<br>Grade {r['grade_num']} · "
                    f"{_fmt_money(r['value'])}<br>Deferred this horizon"
                    for _i, r in none.iterrows()],
                hoverinfo="text", name="deferred"))
        caption = ("Coloured dots: optimiser's renewal year (deep lavender = early, pale = late). "
                   "Grey dots: buildings deferred beyond the horizon. Dot size = replacement value.")
    elif shading == "Current grade":
        bdf["color_hex"] = bdf["grade"].map(GRADE_COLORS)
        fig.add_trace(go.Scattermap(
            lat=bdf["lat"], lon=bdf["lon"], mode="markers",
            marker=dict(size=sizes, color=bdf["color_hex"]),
            text=hover, hoverinfo="text", name="buildings"))
        caption = ("Dots coloured by today's service grade (1 green = near-new → "
                   "5 terracotta = failed). Dot size = replacement value.")
    else:  # Climate exposure
        fig.add_trace(go.Scattermap(
            lat=bdf["lat"], lon=bdf["lon"], mode="markers",
            marker=dict(
                size=sizes, color=bdf["climate_ex"],
                colorscale=EMBERS_SCALE, cmin=0.0, cmax=1.0,
                showscale=True, colorbar=dict(title="Max hazard")),
            text=[
                f"{r['name']} ({r['suburb']})<br>Grade {r['grade_num']} · "
                f"{_fmt_money(r['value'])}<br>Max hazard {r['climate_ex']:.2f}"
                for _i, r in bdf.iterrows()],
            hoverinfo="text"))
        caption = (f"Dots coloured by the maximum hazard intensity (heat / flood / bushfire) "
                   f"at year {CURRENT_YEAR + horizon - 1} under "
                   f"{SCENARIO_LABELS.get(scenario, scenario)}. Dot size = replacement value.")

    fig.update_layout(
        map_style="carto-positron", map_zoom=10.6, map_center=_MAP_CENTER,
        margin=dict(l=0, r=0, t=8, b=0), height=540, showlegend=False)
    st.plotly_chart(fig, width="stretch")
    st.caption(caption)

    st.markdown('<div style="height:0.5rem"></div>', unsafe_allow_html=True)
    st.markdown("#### Inspect a building")

    name_to_bid = dict(zip(
        bdf["name"] + " — " + bdf["suburb"].fillna(""), bdf["building_id"]))
    options = ["(none)"] + sorted(name_to_bid.keys())
    pick = st.selectbox("Pick a building from the register", options,
                        label_visibility="collapsed")
    if pick != "(none)":
        bid = name_to_bid[pick]
        card = building_card(conn, building_id=bid, scenario=scenario, horizon=horizon)
        if card:
            _render_building_card(st, go, pd, card)
    else:
        st.caption("Pick a building above to open its detail card.")

    st.markdown('<div style="height:0.6rem"></div>', unsafe_allow_html=True)
    cev = panels["climate"]
    if cev and cev["years"]:
        with st.container(border=True):
            st.markdown("**Climate-exposed asset value across the portfolio, by hazard**")
            cfig = go.Figure()
            for hazard, series in cev["by_hazard"].items():
                cfig.add_trace(go.Scatter(
                    x=cev["years"], y=series, name=hazard.title(), stackgroup="one",
                    line=dict(width=0.5, color=HAZARD_COLORS.get(hazard, "#888"))))
            cfig.update_layout(
                margin=dict(l=10, r=10, t=8, b=10), height=240,
                legend=dict(orientation="h", y=-0.22))
            st.plotly_chart(cfig, width="stretch")
            st.caption(
                f"Exposed value rises from {_fmt_money(cev['total'][0])} in "
                f"{cev['years'][0]} to {_fmt_money(cev['total'][-1])} in "
                f"{cev['years'][-1]} under {SCENARIO_LABELS.get(scenario, scenario)}. "
                "Exposure = replacement value under rising hazard, not an actuarial loss.")


def _optimiser_override_panel(
    st, go, pd, conn, funded: list[dict], deferred: list[dict], spend: dict,
    scenario: str, budget: float, horizon: int,
) -> None:
    """Unified 'Override the optimiser' tool — one panel, two directions.

    **Defer** a building the optimiser funded (an instant present-value overlay,
    no re-solve) or **bring forward** a deferred building into the plan (a real
    constrained re-solve that shows which renewal gets pushed out). Both act on
    the whole programme, so the panel sits below the Funded/Deferred tabs rather
    than inside either one.
    """
    with st.expander(
        "Override the optimiser — defer a funded building, or bring a deferred one forward",
        expanded=False,
    ):
        st.caption(
            "Two directions on the same programme. **Defer** pushes a funded "
            "building later and prices the present-value consequence instantly "
            "(no re-solve). **Bring forward** constrains the optimiser to renew a "
            "deferred building and re-solves to show which renewal it bumps to "
            "make room.")
        mode = st.radio(
            "Direction",
            ["Defer a funded building", "Bring a deferred building forward"],
            horizontal=True, label_visibility="collapsed",
            key="plan_override_mode")
        st.markdown('<div style="height:0.5rem"></div>', unsafe_allow_html=True)
        if mode == "Defer a funded building":
            _override_panel(st, go, pd, conn, funded, spend, scenario, budget, horizon)
        else:
            _force_in_panel(st, go, pd, conn, deferred, spend, scenario, budget, horizon)


def _override_panel(
    st, go, pd, conn, funded: list[dict], spend: dict,
    scenario: str, budget: float, horizon: int,
) -> None:
    """'What if I overrode the optimiser and deferred a building?' — quick overlay.

    Lets the user pick buildings from the funded programme and see the consequence
    without re-solving: dollars freed in the funded year, present-value carrying
    + escalation cost of leaving the asset un-renewed, additional years degraded,
    and a side-by-side spend-by-year chart so the budget impact is visible.
    """
    with st.container():
        st.caption(
            "Pick one or more buildings from the funded programme to override. The "
            "dashboard computes the consequence without re-running the optimiser: the "
            "dollars freed in that year, the present-value carrying + escalation cost "
            "of leaving the asset un-renewed, and the years it spends degraded past "
            "intervention. Assumptions (discount rate, escalation, carrying cost) "
            "inherit from the Timing & deferral panel under THE TRADE-OFFS — adjust "
            "them there if you want a different sensitivity.")

        bid_label = {
            f["bid"]: (
                f"{f['building']}  ·  renew {int(f['renewal_year'])}  ·  "
                f"{_fmt_money(f['mean_cost'] or 0)}"
            )
            for f in funded if f["renewal_year"]
        }
        if not bid_label:
            st.info("No buildings in the funded programme to override.")
            return

        labels = list(bid_label.values())
        chosen_labels = st.multiselect(
            "Override and defer these buildings", options=labels, default=[],
            key="plan_override_buildings")
        label_to_bid = {v: k for k, v in bid_label.items()}
        chosen_bids = [label_to_bid[lab] for lab in chosen_labels if lab in label_to_bid]

        if not chosen_bids:
            st.caption("Select one or more buildings above to see the consequence.")
            return

        r_pct = float(st.session_state.get("tim_r", 5.0))
        g_pct = float(st.session_state.get("tim_g", 3.0))
        carry_pct = float(st.session_state.get("tim_carry", 3.0))
        impact = deferral_impact(
            conn, scenario=scenario, horizon=horizon,
            override_bids=chosen_bids,
            r=r_pct / 100.0, g=g_pct / 100.0, carry_rate=carry_pct / 100.0,
        )

        net_pv = impact["total_pv_cost"] - impact["total_pv_saving"]
        k1, k2, k3 = st.columns(3)
        with k1:
            _kpi_card(st, "Buildings overridden", str(len(impact["rows"])),
                      f"r={r_pct:.1f}% · g={g_pct:.1f}% · carry={carry_pct:.1f}%")
        with k2:
            _kpi_card(st, "Capacity freed", _fmt_money(impact["total_freed"]),
                      "summed across the funded years", tone="warn")
        with k3:
            if net_pv >= 0:
                _kpi_card(st, "Net PV cost of overriding",
                          _fmt_money(net_pv),
                          "carrying + escalation vs original year",
                          tone="bad" if net_pv > 0 else "")
            else:
                _kpi_card(st, "Net PV saving from overriding",
                          _fmt_money(-net_pv),
                          "original year was earlier than needed",
                          tone="")

        orig_spend = {int(y): float(s) for y, s in zip(spend["years"], spend["spend"])}
        override_spend = orig_spend.copy()
        for y, amt in impact["freed_by_year"].items():
            override_spend[int(y)] = override_spend.get(int(y), 0.0) - float(amt)
        sfig = go.Figure()
        sfig.add_trace(go.Bar(
            x=list(orig_spend.keys()), y=list(orig_spend.values()),
            name="Original schedule", marker_color="rgba(9,117,86,0.40)"))
        sfig.add_trace(go.Bar(
            x=list(override_spend.keys()), y=list(override_spend.values()),
            name="After override", marker_color=BRIGHT_GREEN))
        sfig.add_hline(
            y=budget, line=dict(color="#d6212b", width=2, dash="dash"),
            annotation_text=f"Annual budget {_fmt_money(budget)}",
            annotation_position="top right")
        sfig.update_layout(
            barmode="group", margin=dict(l=8, r=8, t=8, b=8), height=300,
            legend=dict(orientation="h", y=-0.18),
            xaxis_title="Year",
            yaxis=dict(title="Renewal $", tickprefix="$", tickformat="~s"))
        st.plotly_chart(sfig, width="stretch")

        impact_rows = [
            {"Building": r_imp["name"], "Suburb": r_imp["suburb"] or "—",
             "Funded year": r_imp["funded_year"],
             "Freed": _fmt_money(r_imp["freed_amount"]),
             "Reaches intervention": (
                 str(r_imp["first_breach"]) if r_imp["first_breach"] else "after horizon"),
             "Years degraded extra": r_imp["years_degraded_extra"],
             "PV cost of override": _fmt_money(r_imp["pv_cost_of_override"]),
             "PV saving (if any)": (
                 _fmt_money(r_imp["pv_saving_from_override"])
                 if r_imp["pv_saving_from_override"] > 0 else "—")}
            for r_imp in impact["rows"]
        ]
        with st.container(border=True):
            st.markdown("**Per-building impact of overriding**")
            st.dataframe(pd.DataFrame(impact_rows), width="stretch", hide_index=True)
            st.caption(
                "PV cost = present value of carrying the asset past intervention until "
                "horizon end, plus the discounted renewal at horizon. Compared with renewing "
                "in the originally-funded year. A positive PV cost means the override is "
                "more expensive in present-value terms; a saving means the original year "
                "was earlier than the asset really needs.")


def _force_in_panel(
    st, go, pd, conn, deferred: list[dict], spend: dict,
    scenario: str, budget: float, horizon: int,
) -> None:
    """Force-into-programme: a real optimiser re-solve constrained to renew the
    chosen deferred buildings, with a diff against the canonical schedule.

    Symmetric to :func:`_override_panel` on the Funded tab — but where override
    asks "what does deferring this building cost?", force-in asks "if we bring
    this building IN, which building gets pushed out, and where does the
    optimiser place it?". Forcing an asset in consumes per-year budget capacity
    that's then unavailable for others, so a re-solve is unavoidable; the
    explicit **Re-solve** button keeps Streamlit from re-running a 5–30 s solve
    on every interaction.
    """
    with st.container():
        st.caption(
            "Pick one or more buildings from the deferred list and the optimiser "
            "re-runs with them constrained to renew within the horizon. It chooses "
            "the year(s); the dashboard shows which buildings get pushed out to make "
            "room. Solving against the stochastic ensemble takes 5–30 seconds."
        )

        bid_label = {
            d["bid"]: (
                f"{d['building']}  ·  {_fmt_money(d['value'] or 0)}  ·  "
                f"criticality {float(d['criticality'] or 0.5):.2f}  ·  "
                + (
                    f"reaches intervention {d['first_breach']}"
                    if d["first_breach"] is not None
                    else "intervention after horizon"
                )
            )
            for d in deferred
        }
        if not bid_label:
            st.info("No deferred buildings to force in at this scenario and budget.")
            return

        labels = list(bid_label.values())
        chosen_labels = st.multiselect(
            "Force these deferred buildings into the programme",
            options=labels, default=[],
            key="plan_force_buildings",
            help=(
                "Multi-select to test scenarios where you bring more than one in at "
                "once. A larger forced set is more likely to be infeasible at a tight "
                "budget — the panel surfaces that explicitly."
            ),
        )
        label_to_bid = {v: k for k, v in bid_label.items()}
        chosen_bids = [
            label_to_bid[lab] for lab in chosen_labels if lab in label_to_bid
        ]

        col_btn, col_status = st.columns([1, 4])
        with col_btn:
            trigger = st.button(
                "Re-solve", type="primary",
                disabled=not chosen_bids,
                key="plan_force_resolve",
            )
        with col_status:
            if not chosen_bids:
                st.caption("Pick at least one deferred building above.")
            else:
                st.caption(
                    f"Click **Re-solve** to constrain the optimiser to renew "
                    f"{len(chosen_bids)} "
                    f"building{'s' if len(chosen_bids) > 1 else ''} "
                    f"under {SCENARIO_LABELS.get(scenario, scenario)} at "
                    f"{_fmt_money(budget)}/yr."
                )

        cache = st.session_state.setdefault("_force_cache", {})
        current_key = (scenario, round(float(budget), -4), frozenset(chosen_bids))

        if trigger and chosen_bids:
            if current_key not in cache:
                with st.spinner("Re-solving with the forced buildings…"):
                    cache[current_key] = force_into_programme(
                        conn, scenario=scenario, horizon=horizon,
                        force_bids=chosen_bids, annual_budget=budget,
                        n_scenarios=40, lam=1.0,
                    )
            st.session_state["_force_last_key"] = current_key

        last_key = st.session_state.get("_force_last_key")
        if last_key is None or last_key not in cache:
            return
        out = cache[last_key]

        if last_key != current_key:
            st.caption(
                "Showing the previous solve. Click **Re-solve** to refresh under "
                "the current selection, scenario and budget."
            )

        if out["infeasible"]:
            st.error(
                "The forced set exceeds the per-year capex envelope — no schedule "
                "fits all of them within the budget × horizon. Drop one or more "
                "buildings, or raise the annual budget in the sidebar."
            )
            return

        k1, k2, k3 = st.columns(3)
        with k1:
            _kpi_card(st, "Brought into the programme", str(out["n_came_in"]),
                      "incidental brought-ins included")
        with k2:
            _kpi_card(st, "Pushed out", str(out["n_pushed_out"]),
                      "lost their funded slot",
                      tone="warn" if out["pushed_out"] else "")
        with k3:
            _kpi_card(st, "Year shifted", str(out["n_shifted_year"]),
                      "kept their slot but moved year")

        spend_years = sorted(
            set(out["original_spend_by_year"]) | set(out["new_spend_by_year"])
        )
        orig = [out["original_spend_by_year"].get(y, 0.0) for y in spend_years]
        new = [out["new_spend_by_year"].get(y, 0.0) for y in spend_years]
        sfig = go.Figure()
        sfig.add_trace(go.Bar(
            x=spend_years, y=orig, name="Original schedule",
            marker_color="rgba(9,117,86,0.40)"))
        sfig.add_trace(go.Bar(
            x=spend_years, y=new, name="After forcing",
            marker_color=BRIGHT_GREEN))
        sfig.add_hline(
            y=budget, line=dict(color="#d6212b", width=2, dash="dash"),
            annotation_text=f"Annual budget {_fmt_money(budget)}",
            annotation_position="top right",
        )
        sfig.update_layout(
            barmode="group", margin=dict(l=8, r=8, t=8, b=8), height=300,
            legend=dict(orientation="h", y=-0.18),
            xaxis_title="Year",
            yaxis=dict(title="Renewal $", tickprefix="$", tickformat="~s"),
        )
        st.plotly_chart(sfig, width="stretch")

        first_forced = next((r for r in out["came_in"] if r["was_forced"]), None)
        first_pushed = out["pushed_out"][0] if out["pushed_out"] else None
        net = (
            sum(out["new_spend_by_year"].values())
            - sum(out["original_spend_by_year"].values())
        )
        parts: list[str] = []
        if first_forced and first_forced["renewal_year"]:
            parts.append(
                f"Forcing **{first_forced['name']}** into the programme lands "
                f"it in {first_forced['renewal_year']} "
                f"({_fmt_money(first_forced['mean_cost'])})."
            )
        if first_pushed:
            parts.append(
                f"**{first_pushed['name']}** "
                f"(originally {first_pushed['original_year']}) is pushed out."
            )
        if abs(net) > 1.0:
            sign = "+" if net > 0 else ""
            parts.append(
                f"Net change in committed renewal value across the horizon: "
                f"{sign}{_fmt_money(net)}."
            )
        if parts:
            st.markdown(" ".join(parts))

        if out["came_in"]:
            with st.container(border=True):
                st.markdown("**Brought into the programme**")
                came_rows = [
                    {
                        "Building": r["name"], "Suburb": r["suburb"] or "—",
                        "Renewal year": (
                            int(r["renewal_year"]) if r["renewal_year"] else "—"
                        ),
                        "Cost": _fmt_money(r["mean_cost"]),
                        "Forced": "✓" if r["was_forced"] else "(incidental)",
                    }
                    for r in out["came_in"]
                ]
                st.dataframe(pd.DataFrame(came_rows), width="stretch", hide_index=True)

        if out["pushed_out"]:
            with st.container(border=True):
                st.markdown("**Pushed out of the programme**")
                pushed_rows = [
                    {
                        "Building": r["name"], "Suburb": r["suburb"] or "—",
                        "Originally funded year": r["original_year"],
                        "Freed value": _fmt_money(r["freed_amount"]),
                    }
                    for r in out["pushed_out"]
                ]
                st.dataframe(pd.DataFrame(pushed_rows), width="stretch", hide_index=True)

        if out["shifted_year"]:
            with st.container(border=True):
                st.markdown("**Kept their slot — but the year shifted**")
                shift_rows = [
                    {
                        "Building": r["name"], "Suburb": r["suburb"] or "—",
                        "Was": r["original_year"], "Now": r["new_year"],
                        "Cost": _fmt_money(r["mean_cost"]),
                    }
                    for r in out["shifted_year"]
                ]
                st.dataframe(pd.DataFrame(shift_rows), width="stretch", hide_index=True)

        st.caption(
            "The forced solve does not overwrite the canonical programme on the "
            "Funded tab — change buildings and click **Re-solve** to explore "
            "another forced set. The stochastic ensemble drifts schedules "
            "slightly turn-to-turn (an honest property of the method); the "
            "headline placement is stable."
        )


def _tab_plan(st, go, px, pd, conn, panels: dict, scenario: str, budget: float, horizon: int) -> None:
    """THE PLAN — the actionable renewal programme.

    Surfaces three layers of the programme so the user can act on it: total
    committed spend year by year against the annual budget, a Gantt of every
    funded renewal in the order it is scheduled, and the buildings the optimiser
    has *not* funded this horizon (with the year each one reaches intervention,
    so the cost of leaving them is visible). The same data the v1 dashboard hid
    in a 25-row table.
    """
    import datetime as dt

    funded = funded_programme(conn, horizon=horizon)
    deferred = deferred_buildings(conn, scenario=scenario, horizon=horizon)
    spend = spend_by_year(conn, horizon=horizon)
    if not funded and not deferred:
        st.info("No optimised programme yet — run `lga-amp pipeline` to populate the engine.")
        return

    funded_total = sum((f["mean_cost"] or 0.0) for f in funded)
    deferred_total = sum((d["value"] or 0.0) for d in deferred)
    peak_idx = (
        max(range(len(spend["spend"])), key=lambda i: spend["spend"][i])
        if any(spend["spend"]) else None
    )

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        _kpi_card(st, "Funded buildings", str(len(funded)),
                  f"{_fmt_money(funded_total)} programmed")
    with k2:
        _kpi_card(st, "Deferred buildings", str(len(deferred)),
                  f"{_fmt_money(deferred_total)} replacement value",
                  tone="warn" if deferred else "")
    with k3:
        _kpi_card(st, "Programme window",
                  f"{spend['years'][0]} – {spend['years'][-1]}",
                  f"{horizon}-year horizon")
    with k4:
        _kpi_card(st, "Peak spend year",
                  str(spend["years"][peak_idx]) if peak_idx is not None else "—",
                  _fmt_money(spend["spend"][peak_idx]) if peak_idx is not None else "$0")

    st.markdown('<div style="height:0.9rem"></div>', unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("**Renewal spend committed by year, against the annual budget**")
        sfig = go.Figure(go.Bar(
            x=spend["years"], y=spend["spend"], marker_color=BRIGHT_GREEN,
            name="Committed renewal"))
        sfig.add_hline(
            y=budget, line=dict(color="#d6212b", width=2, dash="dash"),
            annotation_text=f"Annual budget {_fmt_money(budget)}",
            annotation_position="top right")
        sfig.update_layout(
            margin=dict(l=8, r=8, t=8, b=8), height=320, showlegend=False,
            xaxis=dict(title="Year"),
            yaxis=dict(title="Renewal $", tickprefix="$", tickformat="~s"))
        st.plotly_chart(sfig, width="stretch")
        st.caption(
            "Bars: total optimiser-committed renewal cost in that year. Dashed line: the "
            "annual budget. Years above the line are oversubscribed; years below have headroom "
            "(the optimiser smooths against the per-year budget constraint).")

    comp = panels.get("arfr_compliance")
    if comp:
        st.markdown('<div style="height:0.9rem"></div>', unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown("**Asset Renewal Funding Ratio — staying inside the regulated corridor**")
            _compliance_corridor(st, go, comp, panels.get("min_budget"))

    st.markdown('<div style="height:0.9rem"></div>', unsafe_allow_html=True)

    funded_tab, deferred_tab = st.tabs(
        [f"Funded ({len(funded)})", f"Deferred ({len(deferred)})"])

    with funded_tab:
        if not funded:
            st.info("No buildings in the funded programme at this scenario and budget.")
        else:
            gantt_rows = [
                {"Building": f["building"], "Suburb": f["suburb"] or "—",
                 "Start": dt.date(int(f["renewal_year"]), 1, 1),
                 "End": dt.date(int(f["renewal_year"]), 12, 31),
                 "Cost": _fmt_money(f["mean_cost"] or 0.0),
                 "Robust": "yes" if f["robust"] else "no"}
                for f in funded if f["renewal_year"]
            ]
            gdf = pd.DataFrame(gantt_rows)
            with st.container(border=True):
                st.markdown("**Funded programme — when each building is scheduled**")
                st.caption(
                    "Each bar marks the year the optimiser commits to renew that "
                    "building; colour groups nearby suburbs. Hover a bar for its "
                    "suburb, cost and robustness — the table below lists them in full.")
                gfig = px.timeline(
                    gdf, x_start="Start", x_end="End", y="Building",
                    color="Suburb", hover_data=["Cost", "Robust"])
                gfig.update_yaxes(autorange="reversed", title_text="Building")
                gfig.update_xaxes(title_text="Renewal year", showgrid=True)
                gfig.update_layout(
                    margin=dict(l=8, r=8, t=8, b=8),
                    height=max(280, 26 * len(gdf)),
                    showlegend=False)
                st.plotly_chart(gfig, width="stretch")

            with st.container(border=True):
                st.markdown("**Funded programme — enriched table**")
                table_rows = [
                    {"Renewal year": int(f["renewal_year"]) if f["renewal_year"] else "—",
                     "Building": f["building"], "Suburb": f["suburb"] or "—",
                     "Asset class": f["asset_type"] or "—",
                     "Current grade": condition_grade_number(float(f["condition_now"] or 1.0)),
                     "Value": _fmt_money(f["value"] or 0.0),
                     "Criticality": f"{float(f['criticality'] or 0.5):.2f}",
                     "Robust priority": "✓" if f["robust"] else "",
                     "Est. cost": _fmt_money(f["mean_cost"] or 0.0)}
                    for f in funded
                ]
                st.dataframe(pd.DataFrame(table_rows), width="stretch", hide_index=True)
                st.caption(
                    "Sortable. Current grade is a 1–5 condition rating (1 = near-new, "
                    "5 = failed / renewal backlog). \"Robust priority ✓\" marks buildings "
                    "the optimiser renews across nearly every simulated future — the "
                    "schedule is least sensitive to the climate realisation drawn.")

    with deferred_tab:
        if not deferred:
            st.success(
                "No buildings deferred at this scenario and budget — every building in the "
                "register is funded within the horizon.")
        else:
            horizon_end = spend["years"][-1]
            rows = []
            for d in deferred:
                first_breach = d["first_breach"]
                yrs_degraded = (
                    max(0, horizon_end - first_breach + 1) if first_breach is not None else 0
                )
                rows.append({
                    "Building": d["building"], "Suburb": d["suburb"] or "—",
                    "Asset class": d["asset_type"] or "—",
                    "Current grade": condition_grade_number(float(d["condition_now"] or 1.0)),
                    "Value": _fmt_money(d["value"] or 0.0),
                    "Criticality": f"{float(d['criticality'] or 0.5):.2f}",
                    "Reaches intervention": (
                        str(first_breach) if first_breach is not None else "after horizon"),
                    "Years degraded by end": yrs_degraded,
                })
            with st.container(border=True):
                st.markdown(
                    f"**{len(deferred)} buildings deferred this horizon** — the optimiser has "
                    "not committed to renew these within the chosen budget and scenario. "
                    "Their condition continues to drift; \"Reaches intervention\" is the first "
                    "year each building's expected condition crosses the 4.0 service-failure "
                    "threshold.")
                st.dataframe(
                    pd.DataFrame(rows).sort_values(
                        ["Reaches intervention", "Value"], ascending=[True, False]),
                    width="stretch", hide_index=True)
                st.caption(
                    "For the present-value cost of deferring a specific building, use the "
                    "Optimal-timing panel under THE TRADE-OFFS — and the per-building card on "
                    "THE PORTFOLIO once it ships (phase 3).")

    st.markdown('<div style="height:1.2rem"></div>', unsafe_allow_html=True)
    _optimiser_override_panel(
        st, go, pd, conn, funded, deferred, spend, scenario, budget, horizon)


def _tab_works(st, pd, panels: dict) -> None:
    """The prioritised works programme table."""
    works = panels["works"]
    if not works:
        st.info("No optimised works programme yet — run `lga-amp optimise`.")
        return
    df = pd.DataFrame(works)
    if "mean_cost" in df:
        df["mean_cost"] = df["mean_cost"].map(_fmt_money)
    if "renew_share" in df:
        df["renew_share"] = df["renew_share"].map(lambda v: f"{v * 100:.0f}%")
    if "renewal_year" in df:
        df["renewal_year"] = df["renewal_year"].map(
            lambda v: "—" if pd.isna(v) else str(int(v)))
    if "robust" in df:
        df["robust"] = df["robust"].map(lambda v: "✓" if v else "")
    df = df.rename(columns={
        "building": "Building", "suburb": "Suburb", "renewal_year": "Renewal year",
        "renew_share": "Share renewed", "robust": "Robust priority", "mean_cost": "Est. cost",
    })
    st.caption(
        "The optimiser's prioritised renewal schedule under the chosen scenario and budget — "
        "highest-priority buildings first. “Share renewed” is the portion of the building's "
        "components funded in the window; “robust priority” (✓) flags buildings the optimiser "
        "renews across nearly every simulated future, not just the central case.")
    st.dataframe(df, width="stretch", hide_index=True)


# --------------------------------------------------------------------------- #
# THE WHAT-IF — add a proposed / missing building, watch the picture move.
# All recompute lands on a session shadow; the canonical DB is never written.
# --------------------------------------------------------------------------- #

# The buildings-fixture asset-type domain (S.4.2). Imported lazily so headless
# mode never touches the fixture loader at import time.
def _asset_types() -> list[str]:
    from engine.ingest.mitcham_public import _load_fixture
    return list(_load_fixture()["unit_rate_per_m2_by_type"].keys())


def _suburbs(conn) -> list[str]:
    """Distinct suburbs in the canonical register (for the proposal form)."""
    rows = conn.execute(
        "SELECT DISTINCT suburb FROM assets WHERE council = 'mitcham' "
        "AND suburb IS NOT NULL ORDER BY suburb"
    ).fetchall()
    return [r[0] for r in rows]


def _suburb_centroid(conn, suburb: str) -> tuple[float | None, float | None]:
    """Mean lat/lon of existing assets in ``suburb`` (the map prefill, S.4.2)."""
    row = conn.execute(
        "SELECT AVG(lat), AVG(lon) FROM assets WHERE council = 'mitcham' AND suburb = ?",
        [suburb],
    ).fetchone()
    if not row or row[0] is None:
        return None, None
    return float(row[0]), float(row[1])


def _next_proposal_bid(proposals: list[dict], status: str) -> str:
    """Mint the next ``pNNNN`` / ``uNNNN`` building id for a new proposal (S.3.4)."""
    prefix = "p" if status == "proposed" else "u"
    used = {
        int(p["building_id"][1:])
        for p in proposals
        if p["building_id"].startswith(prefix) and p["building_id"][1:].isdigit()
    }
    n = 1
    while n in used:
        n += 1
    return f"{prefix}{n:04d}"


def _whatif_banner(st, n_enabled: int) -> None:
    """Persistent gold-bordered hypothetical banner + context pill (S.4.5)."""
    st.markdown(
        f"""
        <div style="border:2px solid {GOLD}; background:#fffdf0; border-radius:14px;
             padding:0.85rem 1.1rem; display:flex; justify-content:space-between;
             align-items:center; gap:1rem; margin:0.2rem 0 1.1rem;">
          <div style="font-weight:600; color:{CHARCOAL};">
            🧪 Hypothetical workspace — <b>nothing here is saved</b>. Every recompute
            runs on a throwaway copy; the council register is never touched.
          </div>
          <span style="background:{GOLD}; color:{CHARCOAL}; border-radius:999px;
                padding:0.3rem 0.8rem; font-size:0.8rem; font-weight:700; white-space:nowrap;">
            WHAT-IF · {n_enabled} proposal{'s' if n_enabled != 1 else ''}
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _whatif_explainer(st) -> None:
    """Dismissible, context-aware honesty-thesis explainer (S.4.7, S.1.2).

    No-op once the session opts out, or when there is no pending add/remove
    event. Renders from a pending-event slot in ``session_state``; a Dismiss
    button clears the event and a "Don't show these tips again" checkbox
    suppresses it for the session. Never blocks interaction.
    """
    if st.session_state.get("_whatif_tips_off"):
        return
    ev = st.session_state.get("_whatif_last_event")
    if not ev:
        return
    if ev["event"] == "add" and ev["status"] == "proposed":
        msg = (
            f"💡 **{ev['label']}** is a *proposed build* — it flatters the near-term "
            "ratios (ACR ↑, backlog diluted, grade ↑) but adds depreciation now and "
            "contributes **nothing** to the 25-year ARFR. Its renewal need lands ~20 yr "
            "after commission (HVAC). Watch the **50-year corridor** and the "
            "**net-new-liability badge** below."
        )
    elif ev["event"] == "add":
        msg = (
            f"💡 **{ev['label']}** is a *missing asset* — registering it surfaces need "
            "that under-registration was hiding: ACR ↓, backlog ↑, ARFR pulled toward "
            "(or through) the 80% floor. This is the honest gap."
        )
    else:
        msg = (
            f"💡 Removed **{ev['label']}** — the diff now reflects the remaining "
            f"{ev['n_enabled']} enabled proposal(s)."
        )
    with st.container(border=True):
        st.markdown(msg)
        c1, c2 = st.columns([1, 3])
        if c1.button("Dismiss", key="_whatif_tip_dismiss"):
            st.session_state["_whatif_last_event"] = None
            st.rerun()
        c2.checkbox("Don't show these tips again", key="_whatif_tips_off")


def _proposal_form(st, conn, next_bid_proposed: str, next_bid_missing: str):
    """Building-level proposal form -> a proposal dict (or None) (S.4.2)."""
    from engine.ingest.proposed import build_proposed_building, min_component_life, unit_rate_for

    try:
        from pydantic import ValidationError
    except Exception:  # pragma: no cover - pydantic is a hard dep
        ValidationError = ValueError  # type: ignore

    suburbs = _suburbs(conn) or ["Mitcham"]
    asset_types = _asset_types()

    with st.expander("➕ Add a proposed or missing building", expanded=True):
        with st.form("whatif_add", clear_on_submit=False):
            status = st.radio(
                "This building is", ["proposed", "missing"], horizontal=True,
                format_func=lambda s: {
                    "proposed": "Proposed (new build)",
                    "missing": "Missing from the register",
                }[s],
            )
            next_bid = next_bid_proposed if status == "proposed" else next_bid_missing
            # Name leads full-width; the short fields pair up in balanced two-column
            # rows so neither column leaves a tall void beside the other.
            name = st.text_input("Name", value="Proposed —")
            r1c1, r1c2 = st.columns(2)
            with r1c1:
                atype = st.selectbox("Asset type", asset_types)
            with r1c2:
                suburb = st.selectbox("Suburb", suburbs)
            r2c1, r2c2 = st.columns(2)
            with r2c1:
                if status == "proposed":
                    year = st.number_input(
                        "Commission year", CURRENT_YEAR, 2051, CURRENT_YEAR + 2, step=1)
                else:
                    year = st.number_input(
                        "Install year", 1950, CURRENT_YEAR, 1995, step=1)
            with r2c2:
                crit = st.slider("Criticality", 0.0, 1.0, 0.6, 0.05)
            if status == "proposed":
                cond = 1.0
            else:
                cond = st.slider(
                    "Current condition (1 new – 5 failed)", 1.0, 5.0, 3.5, 0.1)
            mode = st.radio("Value", ["Lump sum", "Footprint × rate"], horizontal=True)
            grc_total = None
            extent = None
            if mode == "Lump sum":
                grc_total = 1_000.0 * st.number_input(
                    "Replacement cost ($k)", 100.0, 50_000.0, 4_000.0, 100.0)
            else:
                fa, rc = st.columns(2)
                extent = fa.number_input("Footprint (m²)", 50.0, 20_000.0, 1_200.0, 50.0)
                try:
                    default_rate = unit_rate_for(atype)
                except KeyError:
                    default_rate = 4_800.0
                rate = rc.number_input("$/m²", 500.0, 12_000.0, float(default_rate), 100.0)
                grc_total = extent * rate
            st.caption(
                f"Replacement cost: **{_fmt_money(grc_total)}** · expands to 7 components "
                f"under `{next_bid}`. Deterioration is component-typical (not asset-type "
                "specific); climate exposure is council-average (not suburb-specific).")
            if st.form_submit_button("Add to proposals", type="primary"):
                try:
                    lat, lon = _suburb_centroid(conn, suburb)
                    rows = build_proposed_building(
                        building_id=next_bid, status=status, name=name,
                        asset_type=atype, suburb=suburb, install_year=int(year),
                        condition=float(cond), criticality=float(crit),
                        grc_total=(grc_total if mode == "Lump sum" else None),
                        extent=(extent if mode != "Lump sum" else None),
                        lat=lat, lon=lon,
                    )
                except (ValueError, ValidationError) as e:    # [FIX-G8] inline validation
                    st.error(f"Could not add proposal: {e}")
                    return None
                return {
                    "building_id": next_bid, "status": status, "rows": rows,
                    "enabled": True, "label": name,
                    "commission_year": int(year),
                    "first_renewal_year": int(year) + int(min_component_life(rows)),
                }
    return None


def _proposal_ledger(st, proposals: list[dict]) -> bool:
    """Render the proposal ledger with enable/disable + remove (S.4.4).

    Returns True if the ledger changed (a rerun is warranted). Mutates the
    ``proposals`` list in place. [FIX-G11] netting caption when 2+ are enabled.
    """
    if not proposals:
        return False
    changed = False
    n_enabled = sum(1 for p in proposals if p["enabled"])
    st.markdown("#### Proposal ledger")
    if n_enabled >= 2:
        st.caption(
            f"Combined effect of {n_enabled} enabled proposals — individual "
            "contributions may offset (a flattering new build can net against a "
            "worsening missing asset). Disable proposals to isolate a single story.")
    for i, p in enumerate(list(proposals)):
        c1, c2, c3, c4 = st.columns([0.5, 3, 1.2, 0.8])
        with c1:
            new_enabled = st.checkbox(
                "on", value=p["enabled"], key=f"_wi_en_{p['building_id']}",
                label_visibility="collapsed")
            if new_enabled != p["enabled"]:
                p["enabled"] = new_enabled
                changed = True
        with c2:
            tag = "Proposed" if p["status"] == "proposed" else "Missing"
            st.markdown(
                f"**{p['label']}** · `{p['building_id']}` · {tag} · "
                f"renews ~{p['first_renewal_year']}")
        with c3:
            st.caption(f"{len(p['rows'])} components")
        with c4:
            if st.button("Remove", key=f"_wi_rm_{p['building_id']}"):
                proposals.remove(p)
                st.session_state["_whatif_last_event"] = {
                    "event": "remove", "status": p["status"], "label": p["label"],
                    "n_enabled": sum(1 for q in proposals if q["enabled"]),
                }
                changed = True
    return changed


def _delta_kpi_card(st, label, value_str, delta, *, good_when="down",
                    fmt=None, unit="") -> None:
    """A KPI card with a vs-today change chip (S.4.3). Reuses ``_kpi_card``."""
    if fmt is None:
        fmt = lambda d: f"{d:+,.0f}"  # noqa: E731
    if delta is None:
        _kpi_card(st, label, value_str, "needs recompute", "")
        return
    if abs(delta) < 1e-9:
        _kpi_card(st, label, value_str, "no change", "")
        return
    improving = (delta < 0) if good_when == "down" else (delta > 0)
    tone = "good" if improving else "bad"
    _kpi_card(st, label, value_str, f"{fmt(delta)}{unit} vs today", tone)


def _channel_a_cards(st, ca: dict) -> None:
    """The four exact Channel-A delta cards (S.2.1, S.4.3)."""
    acr = ca["acr"]
    dep = ca["depreciation"]
    backlog = ca["backlog"]

    def _pct(v):
        return f"{v * 100:.0f}%" if v is not None else "—"

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _delta_kpi_card(
            st, "Asset Consumption Ratio", _pct(acr["proposed"]),
            (acr["delta"] * 100 if acr["delta"] is not None else None),
            good_when="up", fmt=lambda d: f"{d:+.1f}", unit="pp")
    with c2:
        _delta_kpi_card(
            st, "Annual depreciation (ASR denom)", _fmt_money(dep["proposed"]),
            dep["delta"], good_when="down", fmt=lambda d: _fmt_money(abs(d)).replace("$", ("+$" if d > 0 else "-$")))
    with c3:
        _delta_kpi_card(
            st, "Renewal backlog", _pct(backlog["proposed"]),
            (backlog["delta"] * 100 if backlog["delta"] is not None else None),
            good_when="down", fmt=lambda d: f"{d:+.1f}", unit="pp")
    with c4:
        grade = ca.get("portfolio_grade")
        if grade:
            _kpi_card(st, "Portfolio grade",
                      f"{GRADE_NUMBER[grade['base']]} → {GRADE_NUMBER[grade['proposed']]}"
                      if grade['base'] != grade['proposed']
                      else str(GRADE_NUMBER[grade['base']]),
                      "value-weighted · 1–5", "")
        else:
            _kpi_card(st, "Portfolio grade", "—", "needs assets", "")


def _channel_b_dimmed(st) -> None:
    """Dimmed Channel-B cards shown before a recompute (S.2.2)."""
    st.markdown(
        '<div style="opacity:0.5;">', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        _kpi_card(st, "Asset Renewal Funding Ratio", "—",
                  "Recompute to model funding impact", "")
    with c2:
        _kpi_card(st, "Funding gap p50", "—",
                  "Recompute to model funding impact", "")
    st.markdown("</div>", unsafe_allow_html=True)


def _tab_whatif(st, go, px, pd, conn, db_path, scenario: str, budget: float, horizon: int) -> None:
    """THE WHAT-IF — add a proposed / missing building and see the impact (S.4).

    Channel A (instant, exact, no solve) renders on every add/toggle. Channel B
    (ARFR / gap / corridor) and B' (programme displacement) require a shadow
    recompute behind the "Recompute impact" / "Re-solve programme" buttons. The
    canonical DB is never written — all recompute lands on a file-copy shadow.
    """
    from engine.render.metrics import asset_economics, report_card
    from engine.whatif.ui_logic import (
        baseline_cache_key,
        channel_a_metrics,
        displacement_diff,
        net_new_liability,
        whatif_cache_key,
    )

    proposals: list[dict] = st.session_state.setdefault("_whatif_proposals", [])
    n_enabled = sum(1 for p in proposals if p["enabled"])

    _whatif_banner(st, n_enabled)
    _whatif_explainer(st)

    # [FIX-G12] Degraded-state guard — Channel A only on an un-simulated DB.
    mc_empty = _table_is_empty(conn, "mc_paths")
    if mc_empty:
        st.warning(
            "Run the full pipeline first to model funding-gap impact — only the exact "
            "register-derived ratios (ACR, ASR denominator, backlog, grade) are available "
            "on this database.")

    # The proposal form.
    new = _proposal_form(
        st, conn,
        _next_proposal_bid(proposals, "proposed"),
        _next_proposal_bid(proposals, "missing"),
    )
    if new is not None:
        proposals.append(new)
        st.session_state["_whatif_last_event"] = {
            "event": "add", "status": new["status"], "label": new["label"],
            "n_enabled": sum(1 for p in proposals if p["enabled"]),
        }
        st.rerun()

    if _proposal_ledger(st, proposals):
        st.rerun()

    enabled = [p for p in proposals if p["enabled"]]
    if not enabled:
        st.info("Add a proposal above to preview its impact.")
        return

    enabled_rows: list = []
    for p in enabled:
        enabled_rows.extend(p["rows"])

    # ---- Channel A: instant, exact (no solve) ---------------------------- #
    # Register-derived, so it works even on a degraded (un-simulated) DB — no
    # mc_paths needed; ACR / depreciation / backlog come straight off the
    # assets table via asset_economics + report_card (S.2.1).
    econ = asset_economics(conn)
    rc = report_card(conn)
    ca = channel_a_metrics(econ, rc, enabled_rows)
    # Portfolio-grade diff (value-weighted, with the proposals folded in).
    ca["portfolio_grade"] = _grade_with_proposals(rc, enabled_rows)

    st.markdown("#### Instant impact — exact register-derived ratios (Channel A)")
    st.caption(
        "These four move the instant a proposal is added — they are computed per "
        "component, in closed form, with no simulation. Exact, not modelled.")
    _channel_a_cards(st, ca)

    # Net-new-liability badge (S.5) — fires per proposed build whose earliest
    # component breach lands beyond the horizon.
    for p in enabled:
        if p["status"] == "proposed" and net_new_liability(
            p["rows"], p["commission_year"], horizon
        ):
            st.markdown(
                f"""
                <div style="border:2px solid {GOLD}; background:#fffdf0; border-radius:12px;
                     padding:0.7rem 1rem; margin:0.6rem 0; color:{CHARCOAL}; font-weight:600;">
                  ⚠️ Net-new liability — <b>{p['label']}</b> embeds a renewal need at
                  ~{p['first_renewal_year']}, beyond the {horizon}-year horizon
                  (ends {CURRENT_YEAR + horizon}). It flatters today's picture but the
                  liability lands later — visible in the 50-year corridor.
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown('<div style="height:1.1rem"></div>', unsafe_allow_html=True)
    st.markdown("#### Modelled funding impact (Channel B)")

    recompute_disabled = mc_empty or st.session_state.get("_whatif_running", False)
    cset = st.columns([1.4, 1.4, 2])
    with cset[0]:
        precision = st.radio(
            "Fidelity", ["Fast preview", "Precise"], horizontal=True,
            help="Fast: N≈40 realisations / 40 scenarios. Precise: 200 / 60.")
    # Fast-preview N=30: an N-sweep on the full 1505-component register showed the
    # portfolio-aggregate ARFR / gap / corridor are stable to <1% from N=20 up to
    # N=200 (the large portfolio averages out per-asset MC noise), so 30 is a safe
    # directional default and ~halves the original N=80 pass. Precise stays at 200.
    n_real = 30 if precision == "Fast preview" else 200
    n_scen = 40 if precision == "Fast preview" else 60
    with cset[1]:
        do_resolve = st.checkbox(
            "Also re-solve the programme", value=False,
            help="Tier B′ — runs Stage 6 to show which existing building is displaced.")
    with cset[2]:
        recompute = st.button(
            "Recompute impact", type="primary", disabled=recompute_disabled,
            key="_whatif_recompute")
        if mc_empty:
            st.caption("Run the full pipeline first to model funding-gap impact.")

    cache_key = whatif_cache_key(scenario, budget, horizon, do_resolve, proposals)
    cached = st.session_state.get("_whatif_result")
    if cached is not None and cached.get("key") != cache_key:
        # A proposal edit / scenario / budget change invalidated the result — show
        # the stale diff greyed rather than blanking (S.4.6).
        st.caption(
            "Inputs changed since the last recompute — the diff below is **stale**. "
            "Re-run *Recompute impact* under the current scenario/budget.")

    # Baseline cache — the canonical-only baseline pass does NOT depend on which
    # proposals are added, so it is keyed on EXACTLY the baseline-affecting inputs
    # (no proposal digest / count). Reusing it across proposal edits is parity-
    # neutral (same N / SHADOW_SEED as the proposal pass). This speeds the common
    # loop of tweak-a-proposal -> recompute: only the proposal pass re-runs.
    # Quantise the budget ONCE ($10k — matches whatif_cache_key's rounding and is
    # finer than the sidebar's $100k slider step) and thread budget_q through the
    # key AND both passes below, so the key can never gate a baseline computed at a
    # different budget. n_scen is in the key only under do_resolve, because the
    # cached base_units come from baseline_solve(..., n_scenarios=n_scen).
    budget_q = round(float(budget), -4)
    baseline_key = baseline_cache_key(
        scenario, budget, horizon, n_real, n_scen, do_resolve)
    baseline_cache: dict = st.session_state.setdefault("_whatif_baseline_cache", {})
    baseline_from_cache = False

    if recompute and not recompute_disabled:
        st.session_state["_whatif_running"] = True
        try:
            baseline = baseline_cache.get(baseline_key)
            baseline_from_cache = baseline is not None
            spinner_msg = (
                f"Re-running the proposal pass at N={n_real} (baseline reused)…"
                if baseline_from_cache
                else f"Building shadow and re-running the model at N={n_real}…"
            )
            with st.spinner(spinner_msg):
                if baseline is None:
                    baseline = _compute_whatif_baseline(
                        db_path, scenario, budget_q, horizon,
                        n_real, n_scen, do_resolve,
                    )
                    baseline_cache[baseline_key] = baseline
                    # Cap session-cache growth over a long slider-dragging demo:
                    # keep only the most-recent 8 baseline keys.
                    if len(baseline_cache) > 8:
                        for k in list(baseline_cache)[:-8]:
                            del baseline_cache[k]
                result = _run_whatif_recompute(
                    db_path, enabled_rows, scenario, budget_q, horizon,
                    n_real, n_scen, do_resolve, cache_key, baseline=baseline,
                )
            result["baseline_from_cache"] = baseline_from_cache
            st.session_state["_whatif_result"] = result
        except Exception as e:  # surface, never crash the view
            st.error(f"Recompute failed: {e}")
        finally:
            st.session_state["_whatif_running"] = False

    result = st.session_state.get("_whatif_result")
    if result is None:
        _channel_b_dimmed(st)
        st.caption(
            "ARFR, funding gap, and the compliance corridor are *modelled*, not exact — "
            "they need a Monte-Carlo re-run. Click **Recompute impact**.")
        return

    if result.get("baseline_from_cache"):
        st.caption("Baseline reused from cache — only the proposal pass was re-run.")
    _render_whatif_result(st, go, pd, result, horizon, displacement_diff)


def _grade_with_proposals(rc: dict, new_rows: list) -> dict | None:
    """Value-weighted portfolio grade before/after folding in the proposal rows."""
    if not rc:
        return None
    base_grade = rc["portfolio_grade"]
    total = float(rc["total_value"] or 0.0)
    wcond = rc["portfolio_condition"] * total
    # Per building: a building's condition is its worst component (MAX).
    by_bid: dict[str, dict] = {}
    for r in new_rows:
        aid = r.asset_id if hasattr(r, "asset_id") else r["asset_id"]
        bid = aid.split("-", 1)[0]
        cond = float(r.condition if hasattr(r, "condition") else r["condition"])
        grc = float(r.grc if hasattr(r, "grc") else r["grc"])
        b = by_bid.setdefault(bid, {"cond": cond, "value": 0.0})
        b["cond"] = max(b["cond"], cond)
        b["value"] += grc
    for b in by_bid.values():
        wcond += b["cond"] * b["value"]
        total += b["value"]
    new_cond = wcond / total if total else 0.0
    return {"base": base_grade, "proposed": condition_grade(new_cond)}


def _compute_whatif_baseline(db_path, scenario, budget, horizon,
                             n_real, n_scen, do_resolve) -> dict:
    """Canonical-only baseline pass on its OWN shadow (S.2.5).

    The baseline does NOT depend on which proposals are added — it runs over the
    canonical register only, before any proposal rows are inserted. So it is fully
    determined by ``(scenario, budget, horizon, n_real, do_resolve)`` and is safe
    to cache and reuse across proposal edits (caching is parity-neutral: identical
    to recomputing it inline at the same N/SHADOW_SEED).

    Extracts the baseline metrics as plain values (numbers / dicts / lists) so the
    shadow can be torn down immediately — nothing here holds the connection.
    """
    from engine.render.metrics import financial_indicators as _fi
    from engine.whatif.recompute import baseline_solve, run_stage_2_5
    from engine.whatif.shadow import close_shadow, open_shadow

    conn, _path = open_shadow(db_path)
    try:
        # [H6] Run the MC to the stress horizon so the 50-yr corridor/metrics read
        # real simulated data, not a table truncated at the 25-yr decision horizon.
        run_stage_2_5(conn, horizon=max(horizon, HORIZON_STRESS), n_realisations=n_real)
        base_fi = _fi(conn, scenario, horizon=horizon, annual_budget=budget)
        base_fi50 = _fi(conn, scenario, horizon=HORIZON_STRESS, annual_budget=budget)
        base_summary = _fetch_dicts(
            conn,
            "SELECT year, avg_condition, breach_share FROM mc_summary "
            "WHERE scenario = ? ORDER BY year", [scenario])
        base_units = None
        if do_resolve:
            base_units = baseline_solve(
                conn, horizon=horizon, annual_budget=budget, n_scenarios=n_scen)
        return {
            "base_fi": base_fi, "base_fi50": base_fi50,
            "base_summary": base_summary, "base_units": base_units,
        }
    finally:
        close_shadow(conn)


def _run_whatif_recompute(db_path, new_rows, scenario, budget, horizon,
                          n_real, n_scen, do_resolve, cache_key,
                          baseline: dict | None = None) -> dict:
    """Open a shadow, run baseline + proposed passes at the same N/seed (S.2.5).

    Returns a plain result dict (never the connection — unpicklable, [FIX-D8]).
    The shadow is read while open, all derived panels snapshotted, then torn down.

    ``baseline`` — when supplied (from the session-state baseline cache), the
    canonical-only baseline pass is SKIPPED and its already-extracted metrics are
    reused. This is parity-neutral: the cached baseline was computed at the same
    ``n_real``/``SHADOW_SEED`` as the proposal pass below, so the displacement diff
    (cached ``base_units`` vs the proposal solve units) still compares like for like.
    On a miss the baseline is computed here on its own shadow first.
    """
    from engine.render.metrics import financial_indicators as _fi
    from engine.render.programme import funded_programme, spend_by_year
    from engine.whatif.recompute import recompute_with_new_assets
    from engine.whatif.shadow import close_shadow, open_shadow

    if baseline is None:
        baseline = _compute_whatif_baseline(
            db_path, scenario, budget, horizon, n_real, n_scen, do_resolve)
    base_fi = baseline["base_fi"]
    base_fi50 = baseline["base_fi50"]
    base_summary = baseline["base_summary"]
    base_units = baseline["base_units"]

    # Proposal pass opens its OWN fresh shadow (canonical + inserted proposals).
    # open_shadow is single-slot: this tears down any prior shadow, which is fine —
    # the baseline metrics above are already extracted as plain values.
    conn, _path = open_shadow(db_path)
    try:
        # Insert proposals, re-run (and re-solve if Tier B′).
        prop_units = recompute_with_new_assets(
            conn, new_rows, scenario=scenario, horizon=horizon, annual_budget=budget,
            n_realisations=n_real, n_scenarios=n_scen, resolve=do_resolve,
            baseline_units=base_units,
            mc_horizon=max(horizon, HORIZON_STRESS))  # [H6] MC to the stress horizon
        prop_fi = _fi(conn, scenario, horizon=horizon, annual_budget=budget)
        prop_fi50 = _fi(conn, scenario, horizon=HORIZON_STRESS, annual_budget=budget)
        prop_summary = _fetch_dicts(
            conn,
            "SELECT year, avg_condition, breach_share FROM mc_summary "
            "WHERE scenario = ? ORDER BY year", [scenario])
        # Dual-horizon corridor (25 & 50 yr) on the proposed shadow.
        comp25 = arfr_compliance(conn, scenario, horizon=horizon, annual_budget=budget)
        comp50 = arfr_compliance(conn, scenario, horizon=HORIZON_STRESS, annual_budget=budget)
        minb = minimum_sustainable_budget(conn, scenario, horizon=horizon)
        # Per-component breach for the proposed buildings.
        comp_breach = _fetch_dicts(
            conn,
            "SELECT asset_id, MIN(year) AS first_breach "
            "FROM mc_paths WHERE scenario = ? AND condition >= 4.0 "
            "AND (asset_id LIKE 'p%' OR asset_id LIKE 'u%') GROUP BY asset_id "
            "ORDER BY asset_id", [scenario])
        # Assigned climate exposure (council-average) for the proposals.
        expo = _fetch_dicts(
            conn,
            "SELECT hazard, AVG(intensity) AS intensity FROM climate_exposure "
            "WHERE scenario = ? AND (asset_id LIKE 'p%' OR asset_id LIKE 'u%') "
            "GROUP BY hazard ORDER BY hazard", [scenario])
        # Need-by-year (proposed shadow) from cumulative need.
        from engine.render.metrics import cumulative_need
        need25 = cumulative_need(conn, scenario, horizon, 0.50)
        prog = None
        if do_resolve:
            prog = {
                "funded": funded_programme(conn, horizon=horizon),
                "spend": spend_by_year(conn, horizon=horizon),
                "base_units": base_units,
                "prop_units": prop_units,
            }
        return {
            "key": cache_key,
            "n_real": n_real,
            "n_scen": n_scen,
            "scenario": scenario,
            "base_fi": base_fi, "prop_fi": prop_fi,
            "base_fi50": base_fi50, "prop_fi50": prop_fi50,
            "base_summary": base_summary, "prop_summary": prop_summary,
            "comp25": comp25, "comp50": comp50, "minb": minb,
            "comp_breach": comp_breach, "expo": expo,
            "need25_years": list(range(CURRENT_YEAR, CURRENT_YEAR + horizon)),
            "need25": need25,
            "programme": prog,
        }
    finally:
        close_shadow(conn)


def _render_whatif_result(st, go, pd, result, horizon, displacement_diff) -> None:
    """Render the six before/after cards, impact sub-panel, dual corridor, programme."""
    base = result["base_fi"]
    prop = result["prop_fi"]
    n = result["n_real"]

    def _d(key):
        b = base.get(key)
        p = prop.get(key)
        if b is None or p is None:
            return p, None
        return p, p - b

    n_scen = result.get("n_scen", n)
    st.success(
        f"Modelled at N={n} realisations over {n_scen} cost scenarios — fast preview "
        f"is directional; the p95 band settles under **Precise**. Numbers below are "
        f"**modelled**, not exact.")
    st.markdown("##### Before / after — six headline indicators")
    r1 = st.columns(4)
    arfr_v, arfr_d = _d("arfr")
    gap_v, gap_d = _d("gap_p50")
    acr_v, acr_d = _d("acr")
    asr_v, asr_d = _d("asr")
    with r1[0]:
        _delta_kpi_card(
            st, "ARFR (modelled)",
            f"{arfr_v * 100:.0f}%" if arfr_v else "—",
            (arfr_d * 100 if arfr_d is not None else None),
            good_when="up", fmt=lambda x: f"{x:+.0f}", unit="pp")
    with r1[1]:
        _delta_kpi_card(
            st, "Funding gap p50", _fmt_money(gap_v or 0.0), gap_d,
            good_when="down",
            fmt=lambda x: ("+" if x > 0 else "-") + _fmt_money(abs(x)))
    with r1[2]:
        _delta_kpi_card(
            st, "Asset Consumption Ratio",
            f"{acr_v * 100:.0f}%" if acr_v else "—",
            (acr_d * 100 if acr_d is not None else None),
            good_when="up", fmt=lambda x: f"{x:+.1f}", unit="pp")
    with r1[3]:
        _delta_kpi_card(
            st, "Asset Sustainability Ratio",
            f"{asr_v * 100:.0f}%" if asr_v else "—",
            (asr_d * 100 if asr_d is not None else None),
            good_when="up", fmt=lambda x: f"{x:+.0f}", unit="pp")
    r2 = st.columns(4)
    bl_v, bl_d = _d("backlog")
    with r2[0]:
        _delta_kpi_card(
            st, "Renewal backlog",
            f"{bl_v * 100:.0f}%" if bl_v is not None else "—",
            (bl_d * 100 if bl_d is not None else None),
            good_when="down", fmt=lambda x: f"{x:+.1f}", unit="pp")
    with r2[1]:
        gb = base.get("portfolio_grade")
        gp = prop.get("portfolio_grade")
        gb_n = GRADE_NUMBER.get(gb) if gb else None
        gp_n = GRADE_NUMBER.get(gp) if gp else None
        _kpi_card(st, "Portfolio grade",
                  f"{gb_n} → {gp_n}" if gb != gp else (str(gp_n) if gp_n else "—"),
                  "value-weighted · 1–5", "")

    # Budget-raise answer (S.4.3 [FIX-G3]) — framed against ASR sustainability.
    minb = result.get("minb") or {}
    p50 = minb.get("p50", {})
    with st.container(border=True):
        st.markdown("**Budget to keep this portfolio sustainable**")
        bb1, bb2 = st.columns(2)
        with bb1:
            _kpi_card(st, "Budget to average ARFR 100%",
                      f"{_fmt_money(p50.get('target_budget', 0.0))}/yr",
                      "with the proposals folded in")
        with bb2:
            _kpi_card(st, "Budget to never drop below 80%",
                      f"{_fmt_money(p50.get('floor_budget', 0.0))}/yr",
                      "the flat budget that holds the floor", "warn")

    # Impact sub-panel (S.4.3 [FIX-G1]).
    st.markdown('<div style="height:0.9rem"></div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("**Impact detail — condition, need timing, and the proposed asset's breach**")
        bs = pd.DataFrame(result["base_summary"])
        ps = pd.DataFrame(result["prop_summary"])
        if not bs.empty and not ps.empty:
            cfig = go.Figure()
            cfig.add_trace(go.Scatter(
                x=bs["year"], y=bs["avg_condition"], name="avg condition (today)",
                line=dict(color="#6b7568", width=2, dash="dot")))
            cfig.add_trace(go.Scatter(
                x=ps["year"], y=ps["avg_condition"], name="avg condition (with proposal)",
                line=dict(color=BRIGHT_GREEN, width=3)))
            cfig.update_layout(margin=dict(l=8, r=8, t=8, b=8), height=240,
                               legend=dict(orientation="h", y=-0.25),
                               yaxis=dict(title="avg condition (1 new – 5 failed)"))
            st.plotly_chart(cfig, width="stretch")

        nfig = go.Figure(go.Scatter(
            x=result["need25_years"], y=result["need25"], fill="tozeroy",
            line=dict(color="#e8590c", width=2), name="cumulative renewal need (p50)"))
        nfig.update_layout(margin=dict(l=8, r=8, t=8, b=8), height=220,
                           yaxis=dict(title="cumulative renewal need $", tickprefix="$",
                                      tickformat="~s"))
        st.plotly_chart(nfig, width="stretch")

        cbreach = result.get("comp_breach") or []
        if cbreach:
            st.markdown("**Per-component first breach (the proposed building)**")
            st.dataframe(pd.DataFrame(cbreach).rename(
                columns={"asset_id": "Component", "first_breach": "First breach year"}),
                width="stretch", hide_index=True)
        expo = result.get("expo") or []
        if expo:
            expo_txt = " · ".join(
                f"{e['hazard']}: {float(e['intensity']):.2f}" for e in expo)
            st.caption(
                f"Assigned climate exposure (council-average, not suburb-specific): {expo_txt}. "
                "The proposed asset inherits the fixture-average hazard curve.")

    # Dual-horizon compliance corridor (25 & 50 yr, S.4.3).
    st.markdown('<div style="height:0.9rem"></div>', unsafe_allow_html=True)
    cc1, cc2 = st.columns(2)
    with cc1:
        with st.container(border=True):
            st.markdown("**Compliance corridor — 25-year**")
            _compliance_corridor(st, go, result.get("comp25"), None)
    with cc2:
        with st.container(border=True):
            st.markdown("**Compliance corridor — 50-year (embedded liability shows here)**")
            _compliance_corridor(st, go, result.get("comp50"), None)

    # Programme displacement (Tier B′).
    prog = result.get("programme")
    if prog is not None:
        st.markdown('<div style="height:0.9rem"></div>', unsafe_allow_html=True)
        diff = displacement_diff(prog["base_units"], prog["prop_units"], horizon)
        with st.container(border=True):
            st.markdown("**Programme displacement — who moves to make room**")
            st.caption(
                "Gold: a proposal that won a funded slot. Red: an existing building funded "
                "in the baseline solve but pushed out by the proposal (same N/seed, so this "
                "is real re-prioritisation, not sampling churn).")
            if not diff["new_in"] and not diff["displaced"] and not diff["shifted"]:
                st.info("No programme displacement at this budget — the proposal fits without "
                        "pushing any existing renewal out.")
            else:
                rows = []
                for d in diff["new_in"]:
                    rows.append({"Building": d["asset_id"], "Change": "brought in (proposal)",
                                 "Year": d["renew_year"], "_tone": "gold"})
                for d in diff["displaced"]:
                    rows.append({"Building": d["asset_id"], "Change": "displaced",
                                 "Year": d["original_year"], "_tone": "red"})
                for d in diff["shifted"]:
                    rows.append({"Building": d["asset_id"],
                                 "Change": f"shifted {d['original_year']}→{d['new_year']}",
                                 "Year": d["new_year"], "_tone": ""})
                ddf = pd.DataFrame(rows)

                def _style(row):
                    tone = row["_tone"]
                    color = ("background-color:#fffced" if tone == "gold"
                             else "background-color:#fbe3e3" if tone == "red" else "")
                    return [color] * len(row)

                show = ddf.drop(columns=["_tone"])
                st.dataframe(ddf.style.apply(_style, axis=1).hide(["_tone"], axis=1)
                             if hasattr(ddf.style, "hide") else show,
                             width="stretch", hide_index=True)


def _tab_engagement(st, go, pd, conn, scenario, budget, horizon) -> None:
    """Two-lens view with a live engagement-strength slider: engineering optimum
    vs community/elected engagement-weighted, re-solved on demand."""
    from engine.engage.scores import building_multipliers, load_engagement
    from engine.optimise.solver import solve_all

    st.markdown(
        "Council capital decisions are made through a community and political lens, not on "
        "engineering grounds alone. This view weights renewal priority by **community "
        "significance** and **elected-member priority**, then shows — transparently — how the "
        "works programme shifts, and what that adjustment costs against the engineering optimum."
    )

    buildings = _building_frame(conn)
    if not buildings:
        st.info("No buildings yet — run the pipeline.")
        return
    eng = building_multipliers(conn)
    if not eng:
        st.info("No engagement scores available.")
        return

    strength = st.slider(
        "Engagement weighting", min_value=0.0, max_value=1.0, value=1.0, step=0.25,
        help="0 = pure engineering optimum · 1 = full community / elected-member weighting",
    )

    cache = st.session_state.setdefault("_eng_cache", {})

    def _renewed(s: float) -> set:
        key = (scenario, round(float(budget)), round(float(s), 3))
        if key not in cache:
            r = solve_all(
                conn, horizon=horizon, annual_budget=budget, n_scenarios=40,
                engagement=eng, engagement_strength=s, persist=False, robust_pass=False,
            )
            cache[key] = {u["asset_id"] for u in r["units"] if u["renewed"]}
        return cache[key]

    with st.spinner("Optimising the engineering and engagement lenses…"):
        base_set = _renewed(0.0)
        adj_set = _renewed(strength)

    bdf = pd.DataFrame(buildings)
    bdf["engagement"] = bdf["building_id"].map(eng).fillna(0.5)
    moved_in = bdf[bdf["building_id"].isin(adj_set - base_set)].sort_values("engagement", ascending=False)
    moved_out = bdf[bdf["building_id"].isin(base_set - adj_set)].sort_values("value", ascending=False)

    c1, c2, c3 = st.columns(3)
    with c1:
        _kpi_card(st, "Facilities reprioritised", str(len(moved_in)),
                  f"{len(base_set)} funded in each lens")
    with c2:
        _kpi_card(st, "Brought in by engagement",
                  _fmt_money(moved_in["value"].sum()) if not moved_in.empty else "$0",
                  f"avg engagement {moved_in['engagement'].mean():.2f}" if not moved_in.empty else None)
    with c3:
        _kpi_card(st, "Deferred for engagement",
                  _fmt_money(moved_out["value"].sum()) if not moved_out.empty else "$0",
                  f"avg engagement {moved_out['engagement'].mean():.2f}" if not moved_out.empty else None,
                  tone="warn")

    st.markdown('<div style="height:0.8rem"></div>', unsafe_allow_html=True)

    scores = load_engagement()
    sdf = pd.DataFrame(
        [{"suburb": s, "score": (v["community"] + v["elected"]) / 2.0} for s, v in scores.items()]
    ).sort_values("score")
    with st.container(border=True):
        st.markdown("**Community + elected-member engagement, by suburb** (illustrative)")
        fig = go.Figure(go.Bar(x=sdf["score"], y=sdf["suburb"], orientation="h", marker_color="#097556"))
        fig.update_layout(margin=dict(l=8, r=8, t=8, b=8), height=560, xaxis_title="engagement (1–5)")
        st.plotly_chart(fig, width="stretch")

    def _show(col, title, frame):
        with col:
            with st.container(border=True):
                st.markdown(title)
                if frame.empty:
                    st.caption("No change at this weighting.")
                else:
                    t = frame[["name", "suburb", "grade", "value", "engagement"]].copy()
                    t["grade"] = t["grade"].map(GRADE_NUMBER)
                    t["value"] = t["value"].map(_fmt_money)
                    t["engagement"] = t["engagement"].map(lambda v: f"{v:.2f}")
                    st.dataframe(t, width="stretch", hide_index=True)

    left, right = st.columns(2)
    _show(left, "**Moved up** — community / elected priority", moved_in)
    _show(right, "**Deferred** — engineering priority, lower engagement", moved_out)

    st.caption(
        "The trade-off, made explicit: honouring community and elected priorities brings the "
        "facilities on the left into the funded programme and defers those on the right — the "
        "latter are higher engineering priority but lower community/political salience. Weights "
        "are illustrative; in practice they are set transparently and recorded, with an equity "
        "overlay so the highest-need assets are not crowded out by the loudest voice. Decision "
        "support, not decision authority."
    )


def _portfolio_set_strip(st, conn, scenario: str, budget: float, horizon: int) -> None:
    """The portfolio set — one recommended portfolio plus four credible options,
    each made by moving a single lever and compared on the same data. Each card
    leads with the metric its lever moves; the levers themselves sit in the tabs
    below (budget A/B, community weighting, present-value timing)."""
    try:
        base = unfunded_liability(conn, scenario, horizon, budget)
        fi = financial_indicators(conn, scenario, horizon=horizon, annual_budget=budget)
        stretch_budget = budget * 1.5
        stretch = unfunded_liability(conn, scenario, horizon, stretch_budget)
    except Exception:
        return
    if not base:
        return
    arfr = fi.get("arfr") if fi else None
    gap0 = base.get("gap_p50") or 0.0
    gap_s = (stretch.get("gap_p50") or 0.0) if stretch else 0.0
    closed = max(0.0, gap0 - gap_s)
    arfr_disp = f"{arfr * 100:.0f}%" if arfr else "—"

    cards = [
        ("reco", "Recommended", "Balanced",
         "The balanced optimum at the council's set budget.",
         [("Budget", f"{_fmt_money(budget)}/yr", ""),
          ("Gap (p50)", _fmt_money(gap0), "bad"),
          ("ARFR by end", arfr_disp, "")]),
        ("", "", "Budget-stretch",
         "A higher annual spend, and what it buys down of the gap.",
         [("Budget", f"{_fmt_money(stretch_budget)}/yr", ""),
          ("Gap (p50)", _fmt_money(gap_s), "good"),
          ("Closes", _fmt_money(closed), "good")]),
        ("", "", "Risk-first",
         "Guards the worst-case tail hardest (CVaR) — fewer surprises.",
         [("Budget", f"{_fmt_money(budget)}/yr", ""),
          ("Worst case (p95)", _fmt_money(base.get("gap_p95") or 0.0), ""),
          ("Lens", "tail-guarded", "")]),
        ("", "", "Community-led",
         "Community + elected priorities weighted up — recorded, auditable.",
         [("Budget", f"{_fmt_money(budget)}/yr", ""),
          ("Gap (p50)", _fmt_money(gap0), ""),
          ("Lens", "Engagement ↓", "")]),
        ("", "", "Defer-and-save",
         "Leans into deferrals where the present-value maths genuinely saves.",
         [("Budget", f"{_fmt_money(budget)}/yr", ""),
          ("Gap (p50)", _fmt_money(gap0), ""),
          ("Lens", "Timing ↓", "")]),
    ]
    for col, (extra, tag, name, desc, rows) in zip(st.columns(5), cards):
        rows_html = "".join(
            f'<div class="orow"><span class="l">{label}</span>'
            f'<span class="v {tone}">{v}</span></div>'
            for label, v, tone in rows)
        tag_html = f'<span class="tag">{tag}</span>' if tag else ""
        with col:
            st.markdown(
                f'<div class="opt {extra}">{tag_html}<div class="onm">{name}</div>'
                f'<div class="odesc">{desc}</div>{rows_html}</div>',
                unsafe_allow_html=True)
    st.caption(
        "Every option is solved on the same data and scenario, so members compare "
        "coherent programmes — not a single number to accept or reject. The levers "
        "that generate each option are below: budget A/B, community weighting, and "
        "present-value timing.")


def _render_streamlit(
    conn: duckdb.DuckDBPyConnection,
    panels: dict,
    scenario: str,
    annual_budget: float,
    horizon: int,
    db_path: str | Path = DEFAULT_DB,
) -> None:
    """Draw the interactive report. Streamlit + Plotly imported lazily here.

    ``db_path`` is the canonical DuckDB PATH (the connection is read-only); the
    What-If view needs the path to build a writable file-copy shadow from it.
    """
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.io as pio
    import streamlit as st

    # "Bloom" Plotly template — restrained official-statistics feel: pastel
    # lavender colourway, faint gridlines, no plot fill, tabular figures.
    pio.templates["mitcham"] = go.layout.Template(
        layout=dict(
            font=dict(family="Inter, sans-serif", color=INK, size=13),
            title=dict(font=dict(family="Inter, sans-serif", size=16, color=INK)),
            colorway=[ACCENT, DEEP, "#7a9a5e", ACCENT_2, "#b8930a"],
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(gridcolor="#e6e0ee", zerolinecolor="#e6e0ee", linecolor="#d9d1e4"),
            yaxis=dict(gridcolor="#e6e0ee", zerolinecolor="#e6e0ee", linecolor="#d9d1e4"),
            colorscale=dict(sequential=[[0, "#ece5f4"], [0.5, ACCENT_2], [1, ACCENT_DEEP]]),
            hoverlabel=dict(font=dict(family="Inter, sans-serif")),
        )
    )
    pio.templates.default = "mitcham"

    st.set_page_config(
        page_title="Buildings Renewal Outlook", page_icon="🏛", layout="wide"
    )
    _inject_brand_css(st)

    views = ["The Call", "The Plan", "The Portfolio", "The Trade-offs", "The What-If"]
    subtitles = {
        "The Call": "Where the portfolio stands today, what we recommend, and the 25-year picture",
        "The Plan": "The actionable renewal programme — what's funded, what's deferred, the spend by year",
        "The Portfolio": "Every council building — by suburb, condition, and climate exposure",
        "The Trade-offs": "Compare funding strategies, weight engagement, time renewals on present value",
        "The What-If": "Add a proposed or missing building and watch the whole financial picture move",
    }

    # Branded top bar (identity only); nav + controls render beneath it.
    st.markdown(
        '<div class="appbar">'
        '<div><div class="nm">Buildings Renewal Outlook'
        '<span class="demo-badge">Demo</span></div>'
        '<div class="sub">City of Mitcham &middot; Asset Management</div></div>'
        '<div class="prep">Prepared by <b>Social Capital Advisory</b><br>'
        '<span class="motto">Posteris Aedificemus</span></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Horizontal nav (styled as a tab strip) + the persistent context controls.
    # Punchy names carry a plain functional tag so the strip is legible cold; the
    # returned value stays the punchy key used for routing and the page title.
    nav_labels = {
        "The Call": "The Call · Overview",
        "The Plan": "The Plan · Programme",
        "The Portfolio": "The Portfolio · Map",
        "The Trade-offs": "The Trade-offs · Options",
        "The What-If": "The What-If",
    }
    view = st.radio(
        "Navigate", views, label_visibility="collapsed", horizontal=True,
        format_func=lambda v: nav_labels.get(v, v))
    cc1, cc2, cc3 = st.columns([1.5, 2.4, 0.9], vertical_alignment="bottom")
    with cc1:
        sel_scenario = st.selectbox(
            "Climate scenario", SCENARIOS,
            index=SCENARIOS.index(scenario) if scenario in SCENARIOS else 1,
            format_func=lambda s: SCENARIO_LABELS.get(s, s))
    with cc2:
        sel_budget = 1_000_000.0 * st.slider(
            "Annual renewal budget", min_value=0.2, max_value=8.0,
            value=float(annual_budget) / 1_000_000, step=0.1, format="$%.1fM")
    with cc3:
        st.markdown(
            '<div style="padding-bottom:0.5rem; font-size:0.78rem; color:var(--muted);">'
            'As at <b style="color:var(--ink-2);">30 Jun 2026</b></div>',
            unsafe_allow_html=True)

    panels = _gather_panels(conn, sel_scenario, sel_budget, horizon)

    st.markdown(
        f"""
        <div class="page-h">
          <div><p class="ttl">{view}</p><p class="sub">{subtitles.get(view, '')}</p></div>
          <div class="ctx-pills">
            <span class="ctx">{SCENARIO_LABELS.get(sel_scenario, sel_scenario)}</span>
            <span class="ctx">{_fmt_money(sel_budget)} / year</span>
            <span class="ctx">2026&ndash;{2026 + horizon}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if view == "The Call":
        _tab_call(st, go, pd, conn, panels, sel_scenario, sel_budget, horizon)
    elif view == "The Plan":
        _tab_plan(st, go, px, pd, conn, panels, sel_scenario, sel_budget, horizon)
    elif view == "The Portfolio":
        _tab_portfolio(st, go, px, pd, conn, panels, sel_scenario, horizon)
    elif view == "The What-If":
        _tab_whatif(st, go, px, pd, conn, db_path, sel_scenario, sel_budget, horizon)
    else:  # The Trade-offs
        st.markdown("##### The portfolio set — one recommendation, four options")
        st.caption(
            "The optimiser hands back one recommended portfolio plus four credible "
            "options, each made by moving a single lever. Choose between coherent "
            "programmes, not a single number.")
        _portfolio_set_strip(st, conn, sel_scenario, sel_budget, horizon)
        st.markdown('<div style="height:1.1rem"></div>', unsafe_allow_html=True)
        st.markdown("##### The levers behind the options")
        engage_tab, scen_tab, time_tab = st.tabs(
            ["Engagement weighting", "Scenarios A/B", "Timing & deferral"])
        with engage_tab:
            st.caption(
                "Weight the optimiser by community + elected priority alongside the "
                "engineering lens. Move the slider to see which facilities are brought in "
                "by engagement and which are deferred — the trade-off, made explicit.")
            _tab_engagement(st, go, pd, conn, sel_scenario, sel_budget, horizon)
        with scen_tab:
            st.caption(
                "Compare two funding strategies side by side at different budgets and "
                "scenarios; then watch the portfolio age under a do-nothing trajectory.")
            _tab_scenarios(st, px, pd, conn, panels, sel_scenario, sel_budget, horizon)
        with time_tab:
            st.caption(
                "Two views on deferral economics: above, the nominal unfunded gap as a "
                "function of the annual budget (does deferral cost?); below, the present-"
                "value optimal renewal year per asset (when does deferral genuinely save?).")
            _deferral_panel(st, conn, sel_scenario, sel_budget, horizon)
            st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)
            _timing_panel(st, conn, sel_scenario, horizon)

    # Provenance footer — the trust signal an officer defends to ESCOSA. Shown
    # on every view so the as-at date, source and method travel with the figures.
    st.markdown(
        '<div class="provfoot">'
        '<span><b>As at</b> 30 Jun 2026</span>'
        '<span><b>Source</b> Mitcham public data (synthesised)</span>'
        f'<span><b>Model run</b> {SCENARIO_LABELS.get(sel_scenario, sel_scenario)} &middot; '
        f'{horizon}-yr &middot; stochastic MILP + CVaR&#8329;&#8325;</span>'
        '<span><b>Method</b> decision support, not decision authority — '
        'chartered-engineer sign-off before any capital decision</span>'
        '</div>',
        unsafe_allow_html=True,
    )


def _parse_db_arg(argv: list[str]) -> Path:
    """Pull a `.duckdb` path out of argv (Streamlit passes script args after `--`)."""
    for i, tok in enumerate(argv):
        if tok == "--db" and i + 1 < len(argv):
            return Path(argv[i + 1])
        if tok.endswith(".duckdb"):
            return Path(tok)
    return Path(DEFAULT_DB)


if __name__ == "__main__":
    render_dashboard(db_path=_parse_db_arg(sys.argv[1:]), headless=False)
