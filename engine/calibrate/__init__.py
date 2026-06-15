"""Stage 3 — calibrate deterioration / cost / climate-factor priors.

Loads the IPWEA useful-lives fixture and writes two reference tables: ``priors``
(one row per building component, useful-life and unit-rate distribution
parameters) and ``climate_factors`` (component x hazard acceleration
coefficients). In v1 the building priors do not vary by building type, so every
component prior is stored against ``asset_type = 'any'``.
"""
from __future__ import annotations

import json
from pathlib import Path

_FIXTURE = Path(__file__).parent / "_fixtures" / "ipwea_useful_lives.json"

_PRIOR_COLUMNS = (
    "asset_type",
    "component",
    "useful_life_mu",
    "useful_life_sigma",
    "unit_rate",
    "unit_rate_cv",
)


def _load_fixture() -> dict:
    return json.loads(_FIXTURE.read_text(encoding="utf-8"))


def write_priors(conn) -> None:
    """Replace the ``priors`` and ``climate_factors`` tables from the fixture."""
    data = _load_fixture()

    conn.execute("DELETE FROM priors")
    conn.execute("DELETE FROM climate_factors")

    for component, p in data["components"].items():
        conn.execute(
            f"INSERT INTO priors ({', '.join(_PRIOR_COLUMNS)}) VALUES (?, ?, ?, ?, ?, ?)",
            [
                "any",
                component,
                p["useful_life_mu"],
                p["useful_life_sigma"],
                p["unit_rate"],
                p["unit_rate_cv"],
            ],
        )

    for factor in data["climate_factors"]:
        conn.execute(
            "INSERT INTO climate_factors (component, hazard, k) VALUES (?, ?, ?)",
            [factor["component"], factor["hazard"], factor["k"]],
        )
