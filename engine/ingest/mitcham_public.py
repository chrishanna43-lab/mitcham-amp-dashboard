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


# Component fan-out shape, shared by ``generate_register`` and the proposal
# builder. Sourced from the fixture so the two paths can never drift; falls back
# to the fixture on disk so callers (e.g. the proposal builder) need no fixture
# handle of their own.
def _component_specs() -> list[dict]:
    """The seven (component, share_of_grc, useful_life_years) fixture rows."""
    return _load_fixture()["components"]


def _components_for_building(
    *,
    building_id: str,
    name: str | None,
    asset_type: str,
    suburb: str | None,
    council: str,
    asset_class: str,
    install_year: int,
    condition,
    building_value: float,
    criticality_seed: float | None,
    extent: float | None = None,
    lat: float | None = None,
    lon: float | None = None,
    components: list[dict] | None = None,
) -> list[Asset]:
    """Fan a single building over its seven components into validated ``Asset`` rows.

    The shared inner fan-out behind both ``generate_register`` and the What-If
    proposal builder (``engine.ingest.proposed.build_proposed_building``).

    Contract (deliberately narrow so the proposal schema can never drift from the
    canonical ingest schema):

    - ``building_value`` is **final** — each component's GRC is
      ``building_value * share_of_grc``; no portfolio-wide rescale happens here
      (``generate_register`` applies its published-total anchor afterwards).
    - ``condition`` is either a **scalar** applied to every component, or a
      sequence of one condition per component (in fixture order). The proposal
      builder passes a scalar; ``generate_register`` passes its per-component
      resampled values so the refactor is byte-identical.
    - ``asset_class`` is threaded as a parameter (the ``Asset`` model forbids
      extras, so an absent/extra field is a ``ValidationError``).
    """
    comps = components if components is not None else _component_specs()
    if isinstance(condition, (int, float)):
        conditions = [float(condition)] * len(comps)
    else:
        conditions = [float(c) for c in condition]
        if len(conditions) != len(comps):
            raise ValueError(
                f"condition sequence length {len(conditions)} != {len(comps)} components"
            )

    rows: list[Asset] = []
    for comp, cond in zip(comps, conditions):
        comp_grc = building_value * comp["share_of_grc"]
        rows.append(
            Asset(
                asset_id=f"{building_id}-{comp['component']}",
                name=name,
                suburb=suburb,
                council=council,
                asset_type=asset_type,
                asset_class=asset_class,
                component=comp["component"],
                install_year=install_year,
                useful_life_years=float(comp["useful_life_years"]),
                condition=cond,
                grc=comp_grc,
                extent=extent,
                lat=lat,
                lon=lon,
                criticality_seed=criticality_seed,
            )
        )
    return rows


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

            # Resample condition per component (preserving the exact rng draw
            # order: band then condition, component by component) and hand the
            # full condition vector to the shared fan-out helper.
            conditions: list[float] = []
            for _comp in components:
                band = band_names[int(rng.choice(len(band_names), p=band_probs))]
                midpoint = _BAND_MIDPOINT[band]
                conditions.append(
                    float(np.clip(rng.normal(midpoint, _CONDITION_SIGMA), 1.0, 5.0))
                )

            rows.extend(
                _components_for_building(
                    building_id=building_id,
                    name=building_name,
                    asset_type=btype,
                    suburb=None,
                    council="mitcham",
                    asset_class=fx["asset_class"],
                    install_year=install_year,
                    condition=conditions,
                    building_value=building_value,
                    criticality_seed=criticality_seed,
                    extent=extent,
                    lat=lat,
                    lon=lon,
                    components=components,
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
