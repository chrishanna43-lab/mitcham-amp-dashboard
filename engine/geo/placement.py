"""Place synthesised buildings at real coordinates within their Mitcham suburb.

The ingest stage gives every building a name like ``Blackwood Library`` and a
crude centroid-jittered coordinate. This module replaces that coordinate with a
deterministic point inside the building's *actual* suburb polygon (ABS ASGS 2021
boundaries) and records the suburb, so the dashboard map shows buildings spread
across the real City of Mitcham. The condition data stays synthetic — only the
location is made geographically faithful.
"""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np

# Suburb boundaries shipped with the dashboard (ABS ASGS 2021, EPSG:4326).
_DEFAULT_GEO = (
    Path(__file__).resolve().parents[1]
    / "render" / "dashboard" / "geo" / "mitcham-suburbs.geojson"
)

Ring = list[tuple[float, float]]  # exterior ring as (lon, lat) vertices


def load_suburb_polygons(path: str | Path = _DEFAULT_GEO) -> dict[str, list[Ring]]:
    """Return ``{suburb_name: [exterior ring, ...]}`` from the suburbs GeoJSON.

    Each polygon part contributes its exterior ring; holes are ignored (suburb
    boundaries effectively have none, and a stray point in a hole is harmless
    for placement).
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    polys: dict[str, list[Ring]] = {}
    for feat in data.get("features", []):
        name = (feat.get("properties") or {}).get("name")
        geom = feat.get("geometry") or {}
        if not name or not geom:
            continue
        rings: list[Ring] = []
        if geom["type"] == "Polygon":
            rings.append([(float(x), float(y)) for x, y in geom["coordinates"][0]])
        elif geom["type"] == "MultiPolygon":
            for part in geom["coordinates"]:
                rings.append([(float(x), float(y)) for x, y in part[0]])
        if rings:
            polys[name] = rings
    return polys


def _point_in_ring(x: float, y: float, ring: Ring) -> bool:
    """Even-odd ray-casting test for a point against one ring."""
    inside = False
    n = len(ring)
    j = n - 1
    for i in range(n):
        xi, yi = ring[i]
        xj, yj = ring[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def _point_in_polygon(x: float, y: float, rings: list[Ring]) -> bool:
    return any(_point_in_ring(x, y, r) for r in rings)


def _bbox(rings: list[Ring]) -> tuple[float, float, float, float]:
    xs = [p[0] for r in rings for p in r]
    ys = [p[1] for r in rings for p in r]
    return min(xs), min(ys), max(xs), max(ys)


def _sample_point(
    rings: list[Ring], rng: np.random.Generator, attempts: int = 200
) -> tuple[float, float]:
    """Rejection-sample a point inside the polygon (bbox centre if it fails)."""
    minx, miny, maxx, maxy = _bbox(rings)
    for _ in range(attempts):
        x = float(rng.uniform(minx, maxx))
        y = float(rng.uniform(miny, maxy))
        if _point_in_polygon(x, y, rings):
            return x, y
    return (minx + maxx) / 2.0, (miny + maxy) / 2.0  # numeric guard for thin polygons


def match_suburb(name: str, suburb_names_desc: list[str]) -> str | None:
    """Longest suburb name that the building name starts with (e.g. 'Lower
    Mitcham Library' → 'Lower Mitcham', not 'Mitcham'). ``suburb_names_desc``
    must be sorted longest-first.
    """
    for s in suburb_names_desc:
        if name == s or name.startswith(s + " "):
            return s
    return None


def place_buildings(
    conn: duckdb.DuckDBPyConnection,
    geo_path: str | Path = _DEFAULT_GEO,
    seed: int = 20260528,
) -> dict:
    """Assign each Mitcham building a suburb and a point inside that suburb.

    Updates ``assets.lat``/``lon`` and a ``suburb`` column (added if missing) for
    every component of each building. Placement is deterministic per building id.
    Returns a summary dict (n_buildings, n_unmatched, unmatched names, per-suburb
    counts).
    """
    import pandas as pd

    polys = load_suburb_polygons(geo_path)
    if not polys:
        raise ValueError(f"no suburb polygons loaded from {geo_path}")
    names_desc = sorted(polys.keys(), key=len, reverse=True)
    all_rings = [r for rings in polys.values() for r in rings]
    gminx, gminy, gmaxx, gmaxy = _bbox(all_rings)
    fallback_xy = ((gminx + gmaxx) / 2.0, (gminy + gmaxy) / 2.0)

    rows = conn.execute(
        """
        SELECT split_part(asset_id, '-', 1) AS building_id, MAX(name) AS name
        FROM assets WHERE council = 'mitcham'
        GROUP BY building_id
        """
    ).fetchall()

    recs: list[dict] = []
    unmatched: list[str] = []
    counts: dict[str, int] = {}
    for building_id, name in rows:
        idx = int(building_id[1:]) if building_id[1:].isdigit() else 0
        rng = np.random.default_rng(seed + idx)
        suburb = match_suburb(name or "", names_desc)
        if suburb is None:
            unmatched.append(name)
            lon, lat = fallback_xy
            suburb_value = None
        else:
            lon, lat = _sample_point(polys[suburb], rng)
            suburb_value = suburb
            counts[suburb] = counts.get(suburb, 0) + 1
        recs.append({"building_id": building_id, "lat": lat, "lon": lon, "suburb": suburb_value})

    df = pd.DataFrame(recs)  # noqa: F841 (scanned by DuckDB below)
    conn.execute("ALTER TABLE assets ADD COLUMN IF NOT EXISTS suburb VARCHAR")
    conn.register("_placements", df)
    try:
        conn.execute(
            """
            UPDATE assets AS a
            SET lat = m.lat, lon = m.lon, suburb = m.suburb
            FROM _placements m
            WHERE split_part(a.asset_id, '-', 1) = m.building_id
            """
        )
    finally:
        conn.unregister("_placements")

    return {
        "n_buildings": len(recs),
        "n_unmatched": len(unmatched),
        "unmatched": unmatched,
        "suburb_counts": counts,
    }
