"""Defaults and settings for the engine."""
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

# The working DuckDB lives OUTSIDE OneDrive. A sync-watched .duckdb/.wal pair
# stalls under write load (OneDrive locks the file mid-insert), so the default
# DB sits on a local, un-synced path. Override per-run with --db. The file is
# disposable and fully reproducible from `lga-amp pipeline`.
DEFAULT_DB = Path.home() / ".lga-amp" / "lga_amp.duckdb"
HORIZON_PRIMARY = 25
HORIZON_STRESS = 50
N_REALISATIONS = 5_000
SCENARIOS = ("no_climate", "rcp45", "rcp85")
RANDOM_SEED = 20260527

# Demo renewal budget: ~2% of the $48.1M Mitcham buildings portfolio, a
# defensible sustainable-renewal rate. Low enough that climate opens a real
# funding gap (the demo's point); override per run with --annual-budget.
DEMO_ANNUAL_BUDGET = 1_000_000.0


class EngineConfig(BaseModel):
    db_path: Path = DEFAULT_DB
    horizon: int = HORIZON_PRIMARY
    n_realisations: int = N_REALISATIONS
    seed: int = RANDOM_SEED
