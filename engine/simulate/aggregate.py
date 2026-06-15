"""Stage 5 — aggregate Monte Carlo paths into per-scenario/year summaries.

Collapses the high-volume ``mc_paths`` table into ``mc_summary``: the renewal
funding gap (cost-if-renewed weighted by the renewal-need flag) is summarised at
the 5th / 50th / 95th percentiles across realisations, alongside the mean
condition and the share of asset-years that breach the renewal threshold.
"""
from __future__ import annotations

import duckdb

AGG_SQL = """
INSERT INTO mc_summary
SELECT
  scenario, year,
  quantile_cont(cost_if_renewed * CAST(renew_need AS DOUBLE), 0.05),
  quantile_cont(cost_if_renewed * CAST(renew_need AS DOUBLE), 0.50),
  quantile_cont(cost_if_renewed * CAST(renew_need AS DOUBLE), 0.95),
  AVG(condition),
  AVG(CAST(renew_need AS DOUBLE))
FROM mc_paths GROUP BY scenario, year
"""


def aggregate(conn: duckdb.DuckDBPyConnection) -> int:
    conn.execute("DELETE FROM mc_summary")
    conn.execute(AGG_SQL)
    return conn.execute("SELECT count(*) FROM mc_summary").fetchone()[0]
