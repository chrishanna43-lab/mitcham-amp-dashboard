"""Regenerate the demonstration DuckDB for the dashboard.

Reproduces the City of Mitcham buildings renewal demo from the public-data
fixtures bundled in the engine. Fully deterministic (fixed seeds), so it
reproduces the same headline figures on every run.

    python scripts/build_demo_db.py              # full demo (n=200 realisations, ~500 MB)
    python scripts/build_demo_db.py --n 80       # smaller, faster DB for cloud hosting

The optimiser uses MOSEK if it is installed and licensed, otherwise it falls
back automatically to the open-source HiGHS solver — no licence required.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from engine.pipeline import run_pipeline

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=str(ROOT / "data" / "lga_amp.duckdb"),
                    help="output DuckDB path (default: data/lga_amp.duckdb)")
    ap.add_argument("--n", type=int, default=200, help="Monte Carlo realisations")
    ap.add_argument("--horizon", type=int, default=25)
    ap.add_argument("--annual-budget", type=float, default=1_000_000.0)
    ap.add_argument("--n-scenarios", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20260527)
    args = ap.parse_args()

    db = Path(args.db)
    db.parent.mkdir(parents=True, exist_ok=True)

    out = run_pipeline(
        db_path=db,
        horizon=args.horizon,
        n_realisations=args.n,
        annual_budget=args.annual_budget,
        n_scenarios=args.n_scenarios,
        granularity="building",
        seed=args.seed,
    )

    h = out["headline"]
    size_mb = db.stat().st_size / 1e6
    print(f"\nBuilt {db}  ({size_mb:.0f} MB)")
    print(
        f"Unfunded gap p50 ~${h['gap_p50'] / 1e6:.1f}M  "
        f"(demand ${h['demand_p50'] / 1e6:.1f}M vs capacity ${h['capacity'] / 1e6:.0f}M)"
    )


if __name__ == "__main__":
    main()
