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

from engine.config import DEFAULT_DB
from engine.render.metrics import (
    climate_exposure_value,
    condition_grade,
    gap_trajectory,
    report_card,
    unfunded_liability,
)
from engine.render.timing import deferral_timing

# Mitcham brand palette + motto (SCA on the prepared-by line).
MITCHAM_GREEN = "#01310c"
BRIGHT_GREEN = "#097556"
GOLD = "#fbec30"
CHARCOAL = "#1c2316"
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
    conn = duckdb.connect(str(db_path))
    try:
        panels = _gather_panels(conn, scenario, annual_budget, horizon)
        if headless:
            return panels
        _render_streamlit(conn, panels, scenario, annual_budget, horizon)
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
    """City of Mitcham brand in a modern, full-width dashboard style."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        :root{
          --green:#01310c; --green-b:#097556; --green-2:#0a5a30; --gold:#fbec30;
          --ink:#17231a; --muted:#6b7568; --bg:#f5f7f4; --card:#ffffff;
          --line:#e8ece8; --mint:#e9f3ee;
        }
        .stApp{ background:var(--bg); color:var(--ink); }
        html, body, [class*="css"], .stMarkdown, p, li, button, input, label, select, textarea{ font-family:'Inter', sans-serif; }
        h1,h2,h3,h4,h5{ font-family:'Inter', sans-serif !important; color:var(--ink) !important; font-weight:700; letter-spacing:-0.015em; }
        .block-container{ max-width:100% !important; padding:1.3rem 2.4rem 3rem !important; }
        header[data-testid="stHeader"]{ background:transparent; }

        /* ---- sidebar ---- */
        [data-testid="stSidebar"]{ background:#ffffff; border-right:1px solid var(--line); min-width:340px !important; width:340px !important; }
        [data-testid="stSidebar"] .block-container{ padding-top:1.1rem !important; }
        .sb-brand{ display:flex; align-items:center; gap:0.85rem; padding:0.1rem 0.1rem 1rem; border-bottom:1px solid var(--line); margin-bottom:1rem; }
        .sb-brand img{ height:96px; width:auto; }
        .sb-brand .nm{ font-weight:700; font-size:0.95rem; color:var(--green); line-height:1.1; }
        .sb-brand .sub{ font-size:0.68rem; color:var(--muted); letter-spacing:0.12em; }
        .sb-foot{ font-size:0.7rem; color:var(--muted); line-height:1.55; border-top:1px solid var(--line); padding-top:0.8rem; margin-top:0.6rem; }
        .sb-foot b{ color:var(--green); font-weight:600; }

        [data-testid="stSidebar"] [role="radiogroup"]{ gap:0.15rem; }
        [data-testid="stSidebar"] [role="radiogroup"] > label{ padding:0.45rem 0.6rem; border-radius:10px; }
        [data-testid="stSidebar"] [role="radiogroup"] > label:hover{ background:var(--mint); }
        [data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked){ background:var(--mint); }
        [data-testid="stSidebar"] [role="radiogroup"] label p{ font-size:0.93rem !important; font-weight:600; color:var(--ink); }

        /* ---- page header ---- */
        .page-h{ display:flex; justify-content:space-between; align-items:flex-end; gap:1rem; margin:0.1rem 0 1.2rem; }
        .page-h .ttl{ font-size:1.55rem; font-weight:700; color:var(--ink); margin:0; letter-spacing:-0.02em; }
        .page-h .sub{ color:var(--muted); font-size:0.9rem; margin:0.15rem 0 0; }
        .ctx-pills{ display:flex; gap:0.4rem; flex-wrap:wrap; }
        .ctx{ background:#fff; border:1px solid var(--line); border-radius:999px; padding:0.3rem 0.75rem; font-size:0.78rem; font-weight:600; color:var(--green); white-space:nowrap; }

        /* ---- hero feature card ---- */
        .hero-card{ position:relative; overflow:hidden; border-radius:20px; padding:1.7rem 1.9rem;
          background:linear-gradient(120deg,#01310c 0%,#0a5a30 58%,#097556 100%); color:#fff;
          box-shadow:0 20px 44px -26px rgba(1,49,12,0.65); }
        .hero-card::after{ content:""; position:absolute; right:-40px; top:-70px; width:280px; height:280px;
          background:radial-gradient(circle,rgba(251,236,48,0.16),transparent 70%); }
        .hc-label{ font-size:0.78rem; letter-spacing:0.08em; text-transform:uppercase; color:rgba(255,255,255,0.82); font-weight:600; margin:0; position:relative; z-index:1; }
        .hc-val{ font-size:3.1rem; font-weight:800; line-height:1; margin:0.35rem 0 0.1rem; letter-spacing:-0.03em; position:relative; z-index:1; }
        .hc-sub{ color:rgba(255,255,255,0.85); font-size:0.9rem; margin:0.2rem 0 0; position:relative; z-index:1; }
        .hc-pills{ margin-top:1rem; display:flex; gap:0.5rem; flex-wrap:wrap; position:relative; z-index:1; }
        .pill{ background:rgba(255,255,255,0.15); color:#fff; border:1px solid rgba(255,255,255,0.28); padding:0.3rem 0.75rem; border-radius:999px; font-size:0.78rem; font-weight:600; }
        .pill.gold{ background:var(--gold); color:#1c2316; border-color:transparent; }

        /* ---- kpi cards ---- */
        .kpi{ background:var(--card); border:1px solid var(--line); border-radius:16px; padding:1.05rem 1.2rem 1.1rem;
          box-shadow:0 1px 2px rgba(16,40,30,0.05); height:100%; }
        .kpi-label{ font-size:0.78rem; color:var(--muted); font-weight:500; margin:0 0 0.4rem; }
        .kpi-val{ font-size:1.85rem; font-weight:700; color:var(--ink); line-height:1.05; letter-spacing:-0.02em; }
        .kpi-chip{ display:inline-block; margin-top:0.6rem; font-size:0.73rem; font-weight:600; padding:0.2rem 0.6rem; border-radius:999px; background:var(--mint); color:#0a5a3f; }
        .kpi-chip.warn{ background:#fdeedd; color:#b5500a; }
        .kpi-chip.bad{ background:#fbe3e3; color:#b3231f; }

        /* ---- chart card frame ---- */
        [data-testid="stVerticalBlockBorderWrapper"]{ background:var(--card); border:1px solid var(--line) !important;
          border-radius:16px; box-shadow:0 1px 2px rgba(16,40,30,0.05); padding:0.5rem 0.7rem; }

        /* ---- native metrics (scenarios tab) ---- */
        [data-testid="stMetric"]{ background:var(--card); border:1px solid var(--line); border-radius:14px;
          padding:0.9rem 1.05rem; box-shadow:0 1px 2px rgba(16,40,30,0.05); }
        [data-testid="stMetricLabel"] p{ font-size:0.74rem !important; color:var(--muted) !important; font-weight:500; }
        [data-testid="stMetricValue"]{ font-weight:700; color:var(--ink); font-size:1.55rem; }

        /* ---- misc ---- */
        [data-testid="stCaptionContainer"] p{ color:var(--muted) !important; font-size:0.8rem; }
        [data-testid="stDataFrame"]{ border:1px solid var(--line); border-radius:12px; overflow:hidden; }
        hr{ border-color:var(--line); }
        ::selection{ background:rgba(9,117,86,0.18); }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _sidebar_brand(st) -> None:
    """City of Mitcham crest + wordmark at the top of the sidebar."""
    import base64

    lockup = Path(__file__).parent / "assets" / "mitcham-lockup.png"
    try:
        b64 = base64.b64encode(lockup.read_bytes()).decode("ascii")
        img = f'<img src="data:image/png;base64,{b64}" alt="City of Mitcham"/>'
    except OSError:
        img = ""
    st.markdown(
        f"""
        <div class="sb-brand">
          {img}
          <div>
            <div class="nm">City of Mitcham</div>
            <div class="sub">ASSET MANAGEMENT</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# Traffic-light colours for service grades and hazard series (legibility beats
# brand restraint for a risk read-out; this is the Mitcham dashboard, not SCA).
GRADE_COLORS = {"A": "#1a7a3a", "B": "#6bbf59", "C": "#f0b400", "D": "#e8590c", "F": "#d6212b"}
HAZARD_COLORS = {"heat": "#e8590c", "flood": "#1f77b4", "bushfire": "#b5170f"}
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


def _hero_card(st, headline: dict, scenario: str, budget: float, grade: str | None, horizon: int) -> None:
    """The green gradient feature card carrying the headline unfunded-gap figure."""
    band = f"{_fmt_money(headline['gap_p05'])} – {_fmt_money(headline['gap_p95'])}"
    grade_pill = f'<span class="pill gold">Grade {grade}</span>' if grade else ""
    st.markdown(
        f"""
        <div class="hero-card">
          <p class="hc-label">{horizon}-year unfunded renewal gap</p>
          <div class="hc-val">{_fmt_money(headline['gap_p50'])}</div>
          <p class="hc-sub">Renewal demand {_fmt_money(headline['demand_p50'])} against funded
          capacity {_fmt_money(headline['capacity'])} &middot; uncertainty band {band}</p>
          <div class="hc-pills">
            <span class="pill">{SCENARIO_LABELS.get(scenario, scenario)}</span>
            <span class="pill">{_fmt_money(budget)} / year</span>
            {grade_pill}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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

    c1, c2, c3 = st.columns(3)
    with c1:
        r = st.slider("Real discount rate (%/yr)", 0.0, 10.0, 5.0, 0.5, key="tim_r") / 100.0
    with c2:
        g = st.slider("Cost escalation if deferred (%/yr)", 0.0, 10.0, 3.0, 0.5,
                      key="tim_g") / 100.0
    with c3:
        carry = st.slider("Carrying cost past intervention (%/yr of value)", 0.0, 10.0,
                          3.0, 0.5, key="tim_carry") / 100.0

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

    st.markdown('<div style="height:0.8rem"></div>', unsafe_allow_html=True)
    _deferral_panel(st, conn, scenario, budget, horizon)

    st.markdown('<div style="height:0.8rem"></div>', unsafe_allow_html=True)
    _timing_panel(st, conn, scenario, horizon)

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
        color_continuous_scale="RdYlGn_r", range_color=(1.0, 5.0), size_max=20,
        hover_name="name", zoom=10.4, center=_MAP_CENTER, map_style="carto-positron",
        height=560, labels={"cond": "condition"})
    fig.update_layout(margin=dict(l=0, r=0, t=8, b=0))
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Condition under {SCENARIO_LABELS.get(scenario, scenario)} — green (good) to red "
               "(needs renewal). No intervention modelled: this is the do-nothing trajectory.")


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
                color=mdf["ex"], colorscale="YlOrRd", cmin=0.0, cmax=1.0,
                showscale=True, colorbar_title="exposure"),
            text=mdf["name"], hoverinfo="text"))
        fig2.update_layout(
            map_style="carto-positron", map_zoom=10.5, map_center=_MAP_CENTER,
            margin=dict(l=0, r=0, t=8, b=0), height=520)
        st.plotly_chart(fig2, width="stretch")


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


def _render_streamlit(
    conn: duckdb.DuckDBPyConnection,
    panels: dict,
    scenario: str,
    annual_budget: float,
    horizon: int,
) -> None:
    """Draw the interactive report. Streamlit + Plotly imported lazily here."""
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.io as pio
    import streamlit as st

    pio.templates["mitcham"] = go.layout.Template(
        layout=dict(
            font=dict(family="Inter, sans-serif", color="#17231a", size=13),
            title=dict(font=dict(family="Inter, sans-serif", size=16, color="#17231a")),
            colorway=["#097556", "#01310c", "#a8ab12", "#496516", "#b8930a"],
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#ffffff",
            xaxis=dict(gridcolor="#eef1ee", zerolinecolor="#eef1ee"),
            yaxis=dict(gridcolor="#eef1ee", zerolinecolor="#eef1ee"),
        )
    )
    pio.templates.default = "mitcham"

    st.set_page_config(
        page_title="City of Mitcham — Buildings Renewal Outlook", page_icon="🏛", layout="wide"
    )
    _inject_brand_css(st)

    views = ["Overview", "Map", "Scenarios", "Engagement", "Climate", "Works programme"]
    subtitles = {
        "Overview": "Renewal need, funding gap and priorities across the building portfolio",
        "Map": "Every council building, by condition and replacement value",
        "Scenarios": "Compare funding strategies and watch the portfolio age",
        "Engagement": "Weighting renewal by community significance and elected-member priority",
        "Climate": "Asset value under rising climate hazard, to 2050",
        "Works programme": "The prioritised, value-weighted renewal schedule",
    }

    with st.sidebar:
        _sidebar_brand(st)
        view = st.radio("Navigate", views, label_visibility="collapsed")
        st.markdown('<div style="height:0.5rem"></div>', unsafe_allow_html=True)
        sel_scenario = st.selectbox(
            "Climate scenario", SCENARIOS,
            index=SCENARIOS.index(scenario) if scenario in SCENARIOS else 1,
            format_func=lambda s: SCENARIO_LABELS.get(s, s))
        sel_budget = 1_000_000.0 * st.slider(
            "Annual renewal budget", min_value=0.2, max_value=8.0,
            value=float(annual_budget) / 1_000_000, step=0.1, format="$%.1fM")
        st.caption(
            "What the council commits to building renewal each year. The model directs it "
            "to the highest-priority buildings; the Overview shows the funding gap that remains.")
        st.markdown(
            '<div class="sb-foot">Prepared by <b>Social Capital Advisory</b>.<br>'
            "Decision support, not decision authority. Illustrative data from public sources.<br><br>"
            "<i>POSTERIS AEDIFICEMUS</i> &mdash; &ldquo;we build for posterity&rdquo;.</div>",
            unsafe_allow_html=True,
        )

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

    if view == "Overview":
        _tab_overview(st, go, pd, panels, sel_scenario, sel_budget, horizon)
    elif view == "Map":
        _tab_map(st, go, pd, panels)
    elif view == "Scenarios":
        _tab_scenarios(st, px, pd, conn, panels, sel_scenario, sel_budget, horizon)
    elif view == "Engagement":
        _tab_engagement(st, go, pd, conn, sel_scenario, sel_budget, horizon)
    elif view == "Climate":
        _tab_climate(st, go, conn, panels, sel_scenario)
    else:
        _tab_works(st, pd, panels)


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
