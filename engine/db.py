"""DuckDB connection helper and schema bootstrap for the LGA-AMP engine."""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import duckdb

TABLES_DDL: dict[str, str] = {
    "assets": """
        CREATE TABLE IF NOT EXISTS assets (
            asset_id VARCHAR PRIMARY KEY,
            name VARCHAR,
            suburb VARCHAR,
            council VARCHAR,
            asset_type VARCHAR,
            asset_class VARCHAR,
            component VARCHAR,
            install_year INTEGER,
            useful_life_years DOUBLE,
            condition DOUBLE,
            grc DOUBLE,
            extent DOUBLE,
            lat DOUBLE,
            lon DOUBLE,
            criticality_seed DOUBLE
        )
    """,
    "climate_exposure": """
        CREATE TABLE IF NOT EXISTS climate_exposure (
            asset_id VARCHAR,
            scenario VARCHAR,
            hazard VARCHAR,
            year INTEGER,
            intensity DOUBLE,
            PRIMARY KEY (asset_id, scenario, hazard, year)
        )
    """,
    "priors": """
        CREATE TABLE IF NOT EXISTS priors (
            asset_type VARCHAR,
            component VARCHAR,
            useful_life_mu DOUBLE,
            useful_life_sigma DOUBLE,
            unit_rate DOUBLE,
            unit_rate_cv DOUBLE,
            PRIMARY KEY (asset_type, component)
        )
    """,
    "climate_factors": """
        CREATE TABLE IF NOT EXISTS climate_factors (
            component VARCHAR,
            hazard VARCHAR,
            k DOUBLE,
            PRIMARY KEY (component, hazard)
        )
    """,
    "criteria_scores": """
        CREATE TABLE IF NOT EXISTS criteria_scores (
            asset_id VARCHAR,
            axis VARCHAR,
            score DOUBLE,
            provenance VARCHAR,
            PRIMARY KEY (asset_id, axis)
        )
    """,
    "mc_paths": """
        CREATE TABLE IF NOT EXISTS mc_paths (
            realisation INTEGER,
            asset_id VARCHAR,
            scenario VARCHAR,
            year INTEGER,
            condition DOUBLE,
            renew_need BOOLEAN,
            cost_if_renewed DOUBLE,
            PRIMARY KEY (realisation, asset_id, scenario, year)
        )
    """,
    "mc_summary": """
        CREATE TABLE IF NOT EXISTS mc_summary (
            scenario VARCHAR,
            year INTEGER,
            funding_gap_p05 DOUBLE,
            funding_gap_p50 DOUBLE,
            funding_gap_p95 DOUBLE,
            avg_condition DOUBLE,
            breach_share DOUBLE,
            PRIMARY KEY (scenario, year)
        )
    """,
    "opt_solutions": """
        CREATE TABLE IF NOT EXISTS opt_solutions (
            realisation INTEGER,
            asset_id VARCHAR,
            scenario VARCHAR,
            renew_year INTEGER,
            cost DOUBLE,
            PRIMARY KEY (realisation, asset_id, scenario)
        )
    """,
    "opt_summary": """
        CREATE TABLE IF NOT EXISTS opt_summary (
            asset_id VARCHAR,
            scenario VARCHAR,
            renew_year_p50 INTEGER,
            renew_share DOUBLE,
            robust BOOLEAN,
            mean_cost DOUBLE,
            PRIMARY KEY (asset_id, scenario)
        )
    """,
}


def connect(db_path: str | Path) -> duckdb.DuckDBPyConnection:
    """Open (or create) a DuckDB database file, creating parent dirs as needed."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(path))


def bootstrap_schema(conn: duckdb.DuckDBPyConnection) -> None:
    """Create all engine tables if they do not already exist."""
    for ddl in TABLES_DDL.values():
        conn.execute(ddl)


def insert_models(
    conn: duckdb.DuckDBPyConnection,
    table: str,
    columns: Sequence[str],
    rows: Sequence[object],
) -> None:
    """Bulk-insert pydantic model rows into ``table`` via a registered DataFrame.

    Vectorised INSERT ... SELECT off a registered DataFrame — orders of
    magnitude faster than row-by-row ``executemany`` for the large batches the
    simulation stages produce, and it avoids the per-row round trips that stall
    when the database file is sync-watched.
    """
    if not rows:
        return
    import pandas as pd

    cols = list(columns)
    df = pd.DataFrame([r.model_dump() for r in rows], columns=cols)  # noqa: F841 (used by DuckDB scan)
    col_list = ", ".join(cols)
    conn.register("_bulk_insert_df", df)
    try:
        conn.execute(
            f"INSERT INTO {table} ({col_list}) SELECT {col_list} FROM _bulk_insert_df"
        )
    finally:
        conn.unregister("_bulk_insert_df")


def insert_dataframe(conn: duckdb.DuckDBPyConnection, table: str, df) -> None:
    """Bulk-insert a pandas DataFrame whose columns match ``table`` order."""
    if df is None or len(df) == 0:
        return
    cols = ", ".join(df.columns)
    conn.register("_bulk_df", df)
    try:
        conn.execute(f"INSERT INTO {table} ({cols}) SELECT {cols} FROM _bulk_df")
    finally:
        conn.unregister("_bulk_df")
