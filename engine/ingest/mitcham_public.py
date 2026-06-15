"""Stage 1 — synthesise the Mitcham buildings asset register.

Generates a row-level component asset register from the published City of
Mitcham Buildings AMP distributions (counts, condition mix, component shares
and useful lives, unit rates). Individual building records are synthesised —
not actual — but the portfolio totals and value-weighted condition mix are
anchored to the published figures.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from engine.schemas import Asset

_FIXTURE = Path(__file__).parent / "_fixtures" / "mitcham_buildings_public.json"

# Condition band → midpoint on the IPWEA 1 (very good) … 5 (very poor) scale.
_BAND_MIDPOINT: dict[str, float] = {
    "very_good": 1.25,
    "good": 2.5,
    "fair": 3.75,
    "poor_or_very_poor": 4.75,
}

# Reference year for age → install_year conversion.
_REF_YEAR = 2026
_MIN_EXTENT_M2 = 5.0
_AGE_SIGMA_YEARS = 6.0
_CONDITION_SIGMA = 0.25

# Real City of Mitcham suburbs — used to give synthesised buildings plausible,
# human-readable names (e.g. "Blackwood Library") instead of opaque ids.
_SUBURBS = (
    "Bedford Park", "Belair", "Bellevue Heights", "Blackwood", "Brown Hill Creek",
    "Clapham", "Clarence Gardens", "Colonel Light Gardens", "Coromandel Valley",
    "Craigburn Farm", "Cumberland Park", "Daw Park", "Eden Hills", "Glenalta",
    "Hawthorndene", "Kingswood", "Lower Mitcham", "Lynton", "Melrose Park",
    "Mitcham", "Netherby", "Panorama", "Pasadena", "Springfield", "St Marys",
    "Torrens Park", "Urrbrae", "Westbourne Park",
)

# Asset-type → facility label used in the building name.
_FACILITY_LABEL: dict[str, str] = {
    "library": "Library",
    "community_centre": "Community Centre",
    "aquatic_centre": "Aquatic Centre",
    "sports_pavilion": "Pavilion",
    "civic_centre": "Civic Centre",
    "depot": "Depot",
    "public_toilet": "Public Toilet",
    "kiosk_shelter": "Kiosk",
    "heritage_building": "Heritage Hall",
}


def _building_name(building_idx: int, btype: str, used: dict[str, int]) -> str:
    """A plausible, unique facility name: '<suburb> <facility label>' (+ counter)."""
    suburb = _SUBURBS[(building_idx - 1) % len(_SUBURBS)]
    base = f"{suburb} {_FACILITY_LABEL.get(btype, btype.replace('_', ' ').title())}"
    used[base] = used.get(base, 0) + 1
    return base if used[base] == 1 else f"{base} {used[base]}"


def _load_fixture() -> dict:
    return json.loads(_FIXTURE.read_text(encoding="utf-8"))


def generate_register(seed: int = 42) -> list[Asset]:
    """Synthesise the row-level Mitcham buildings register.

    Returns a list of validated :class:`Asset` rows, one per building component.
    Raw component GRC is uniformly scaled so the portfolio total equals the
    published ``total_grc_aud``; because condition bands are sampled
    independently of value, uniform scaling preserves the value-weighted
    condition mix.
    """
    fx = _load_fixture()
    rng = np.random.default_rng(seed)

    centroid = fx["geo_centroid"]
    jitter_deg = fx["geo_jitter_km"] / 111.0
    unit_rates = fx["unit_rate_per_m2_by_type"]
    components = fx["components"]
    cond_dist = fx["condition_distribution_by_value"]
    band_names = list(cond_dist.keys())
    band_probs = np.array([cond_dist[b] for b in band_names], dtype=float)

    rows: list[Asset] = []
    building_idx = 0
    used_names: dict[str, int] = {}

    for bt in fx["building_types"]:
        btype = bt["type"]
        unit_rate = unit_rates[btype]
        log_median = math.log(bt["median_extent_m2"])
        sigma = bt["extent_cv"]
        median_age = bt["median_age"]

        for _ in range(bt["count"]):
            building_idx += 1
            building_id = f"b{building_idx:04d}"
            building_name = _building_name(building_idx, btype, used_names)

            extent = float(np.clip(rng.lognormal(mean=log_median, sigma=sigma), _MIN_EXTENT_M2, None))
            age = max(1, int(rng.normal(median_age, _AGE_SIGMA_YEARS)))
            install_year = _REF_YEAR - age
            lat = centroid["lat"] + float(rng.uniform(-jitter_deg, jitter_deg))
            lon = centroid["lon"] + float(rng.uniform(-jitter_deg, jitter_deg))
            criticality_seed = float(rng.uniform(0.3, 1.0))

            building_value = extent * unit_rate

            for comp in components:
                comp_grc = building_value * comp["share_of_grc"]
                band = band_names[int(rng.choice(len(band_names), p=band_probs))]
                midpoint = _BAND_MIDPOINT[band]
                condition = float(np.clip(rng.normal(midpoint, _CONDITION_SIGMA), 1.0, 5.0))

                rows.append(
                    Asset(
                        asset_id=f"{building_id}-{comp['component']}",
                        name=building_name,
                        council="mitcham",
                        asset_type=btype,
                        asset_class=fx["asset_class"],
                        component=comp["component"],
                        install_year=install_year,
                        useful_life_years=float(comp["useful_life_years"]),
                        condition=condition,
                        grc=comp_grc,
                        extent=extent,
                        lat=lat,
                        lon=lon,
                        criticality_seed=criticality_seed,
                    )
                )

    # Anchor the synthesised portfolio total to the published figure.
    raw_sum = sum(r.grc for r in rows)
    scale = fx["total_grc_aud"] / raw_sum
    for r in rows:
        r.grc = r.grc * scale

    return rows


_COLUMNS = (
    "asset_id",
    "name",
    "council",
    "asset_type",
    "asset_class",
    "component",
    "install_year",
    "useful_life_years",
    "condition",
    "grc",
    "extent",
    "lat",
    "lon",
    "criticality_seed",
)


def write_register(conn, rows: list[Asset]) -> None:
    """Replace the Mitcham asset rows in ``assets`` with ``rows``."""
    from engine.db import insert_models

    conn.execute("DELETE FROM assets WHERE council = 'mitcham'")
    insert_models(conn, "assets", _COLUMNS, rows)
