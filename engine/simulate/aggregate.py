"""Stage 5 — aggregate Monte Carlo paths into per-scenario/year summaries.

Collapses the high-volume ``mc_paths`` table into ``mc_summary``. The
``funding_gap_pXX`` columns are the 5th / 50th / 95th percentiles, *across
realisations*, of the **per-realisation annual renewal need** (the portfolio
sum of cost-if-renewed for assets breaching that year) — NOT a pooled quantile
over individual asset-year cells, and NOT the unfunded gap (which is owned by
``metrics.cumulative_need`` / ``unfunded_liability``: cumulative, first-breach-
once, net of capacity). ``avg_condition`` and ``breach_share`` are the portfolio
means for the (scenario, year), unchanged by the per-realisation grouping below
because every asset appears in every realisation. [M3]
"""
from __future__ import annotations

import duckdb

# Inner query forms each realisation's portfolio total for the year; the outer
# query takes the cross-realisation quantiles. Grouping condition/breach by
# realisation first and averaging is algebraically identical to the flat mean
# (balanced panel), so those two columns are preserved exactly.
AGG_SQL = """
INSERT INTO mc_summary
SELECT
  scenario, year,
  quantile_cont(realisation_need, 0.05),
  quantile_cont(realisation_need, 0.50),
  quantile_cont(realisation_need, 0.95),
  AVG(realisation_condition),
  AVG(realisation_breach)
FROM (
  SELECT scenario, year, realisation,
    SUM(cost_if_renewed * CAST(renew_need AS DOUBLE)) AS realisation_need,
    AVG(condition) AS realisation_condition,
    AVG(CAST(renew_need AS DOUBLE)) AS realisation_breach
  FROM mc_paths
  GROUP BY scenario, year, realisation
)
GROUP BY scenario, year
"""


def aggregate(conn: duckdb.DuckDBPyConnection) -> int:
    conn.execute("DELETE FROM mc_summary")
    conn.execute(AGG_SQL)
    return conn.execute("SELECT count(*) FROM mc_summary").fetchone()[0]
