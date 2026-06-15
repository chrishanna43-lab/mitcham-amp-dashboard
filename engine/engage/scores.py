"""Community + elected-member engagement weighting (illustrative).

Turns per-suburb community-significance and elected-member-priority scores into
a per-building engagement multiplier the optimiser can fold into its breach
weight. ALL VALUES ARE ILLUSTRATIVE for the demonstration — in a real
engagement these come from structured councillor input, Your Say / consultation
data, petitions, service requests, facility usage and an equity overlay.
"""
from __future__ import annotations

import json
from pathlib import Path

import duckdb

_FIXTURE = (
    Path(__file__).resolve().parents[1] / "ingest" / "_fixtures" / "mitcham_engagement.json"
)


def load_engagement(path: str | Path = _FIXTURE) -> dict[str, dict]:
    """Return ``{suburb: {"community": int, "elected": int}}`` from the fixture."""
    return json.loads(Path(path).read_text(encoding="utf-8"))["suburbs"]


def engagement_index(community: float, elected: float) -> float:
    """Combine the two 1–5 scores into a 0..1 engagement index."""
    return (float(community) + float(elected)) / 10.0


def suburb_index(path: str | Path = _FIXTURE) -> dict[str, float]:
    """Return ``{suburb: engagement index 0..1}``."""
    return {
        suburb: engagement_index(v["community"], v["elected"])
        for suburb, v in load_engagement(path).items()
    }


def building_multipliers(
    conn: duckdb.DuckDBPyConnection, default: float = 0.5
) -> dict[str, float]:
    """Return ``{building_id: engagement index}``.

    Joins each building's suburb to the engagement scores; buildings whose
    suburb has no score (or no suburb) get ``default``.
    """
    idx = suburb_index()
    rows = conn.execute(
        "SELECT DISTINCT split_part(asset_id, '-', 1) AS bid, suburb "
        "FROM assets WHERE council = 'mitcham'"
    ).fetchall()
    return {bid: idx.get(suburb, default) for bid, suburb in rows}
