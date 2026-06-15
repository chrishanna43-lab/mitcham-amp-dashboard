"""Stage 2 — build per-asset climate hazard exposure paths.

For every asset, every scenario (no_climate / rcp45 / rcp85) and every hazard
(heat / flood / bushfire) we project a normalised intensity index across the
planning horizon. Intensity is linearly interpolated in time between the
fixture's baseline (baseline_year) and endpoint (endpoint_year) values, clamped
so years at or before the baseline read the baseline value and years at or after
the endpoint read the endpoint value.
"""
from __future__ import annotations

import json
from pathlib import Path

from engine.schemas import Asset, ClimateExposure, Hazard

_FIXTURE = Path(__file__).parent / "_fixtures" / "mitcham_climate.json"

_HAZARDS: tuple[Hazard, ...] = ("heat", "flood", "bushfire")

# Simulation start year. Exposure rows must span the SAME window the Monte Carlo
# stage reads ([CURRENT_YEAR, CURRENT_YEAR + horizon)); the climate projection is
# still interpolated from the fixture's (earlier) baseline_year.
CURRENT_YEAR = 2026


def _load_fixture() -> dict:
    return json.loads(_FIXTURE.read_text(encoding="utf-8"))


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def build_exposure(assets: list[Asset], horizon: int) -> list[ClimateExposure]:
    """Project climate exposure for every asset over ``horizon`` years.

    Returns a list of validated :class:`ClimateExposure` rows — one per
    (asset, scenario, hazard, year) — where ``year`` runs from ``CURRENT_YEAR``
    (the simulation start, inclusive) for ``horizon`` years, matching the window
    the Monte Carlo stage reads. Intensity is the linear interpolation
    ``b + (e - b) * t`` with ``t`` the clamped fraction of the
    baseline→endpoint span (from the fixture's ``baseline_year``) elapsed by
    that year.
    """
    fx = _load_fixture()
    baseline_year = int(fx["baseline_year"])
    endpoint_year = int(fx["endpoint_year"])
    scenarios: dict = fx["scenarios"]
    span = float(endpoint_year - baseline_year)

    years = range(CURRENT_YEAR, CURRENT_YEAR + horizon)

    expo: list[ClimateExposure] = []
    for asset in assets:
        for scenario, hazards in scenarios.items():
            for hazard in _HAZARDS:
                band = hazards[hazard]
                b = float(band["baseline"])
                e = float(band["endpoint"])
                for year in years:
                    t = _clip((year - baseline_year) / span, 0.0, 1.0) if span else 0.0
                    intensity = b + (e - b) * t
                    expo.append(
                        ClimateExposure(
                            asset_id=asset.asset_id,
                            scenario=scenario,
                            hazard=hazard,
                            year=year,
                            intensity=intensity,
                        )
                    )
    return expo


_COLUMNS = ("asset_id", "scenario", "hazard", "year", "intensity")


def write_exposure(conn, expo: list[ClimateExposure]) -> None:
    """Replace all rows in ``climate_exposure`` with ``expo``."""
    from engine.db import insert_models

    conn.execute("DELETE FROM climate_exposure")
    insert_models(conn, "climate_exposure", _COLUMNS, expo)
