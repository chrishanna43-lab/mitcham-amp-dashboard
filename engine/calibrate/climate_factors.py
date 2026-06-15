"""Stage 3 — climate acceleration factors.

Each factor pairs a component with a hazard and an acceleration coefficient
``k`` (additional condition-units lost per year per unit hazard intensity).
Loaded from the IPWEA useful-lives fixture and validated through the shared
:class:`~engine.schemas.ClimateFactor` model.
"""
from __future__ import annotations

import json
from pathlib import Path

from engine.schemas import ClimateFactor

_FIXTURE = Path(__file__).parent / "_fixtures" / "ipwea_useful_lives.json"


def load_climate_factors() -> list[ClimateFactor]:
    """Return every climate acceleration factor from the fixture."""
    data = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    return [ClimateFactor(**row) for row in data["climate_factors"]]
