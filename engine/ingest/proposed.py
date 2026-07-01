"""Construct a proposed / missing building from minimal council-officer input.

The What-If feature's data-model entry point (S.3.1). One building-level form
expands to <=7 validated :class:`Asset` component rows via the shared ingest
fan-out (``engine.ingest.mitcham_public._components_for_building``), so the
proposal schema can never drift from the canonical register schema.

Two paths share this builder:
  * a **proposed new build** (future/current ``commission_year``, condition ~1.0);
  * a **missing / under-registered** existing asset (past ``install_year``,
    mid/poor condition).

For a proposed build ``install_year == commission_year`` (the divergence from
ingest, which derives ``install_year = _REF_YEAR - age``).
"""
from __future__ import annotations

from engine.ingest.mitcham_public import (
    _component_specs,
    _components_for_building,
    _load_fixture,
)
from engine.schemas import Asset

# The seven canonical building components. The ``Asset`` pydantic model leaves
# ``component`` a free VARCHAR, so an unknown component would silently default
# to cv=0.2 / k=0 in the Monte Carlo (S.3.1, [FIX-G10/D13]); enforce the list
# here, at the only construction site for proposal rows.
_ALLOWED_COMPONENTS = {
    "roof", "envelope", "structure", "internal", "hvac", "electrical", "hydraulic",
}

# Canonical council — hard-pinned on every proposal row ([FIX-B1], S.3.4). Five
# engine paths hard-filter WHERE council='mitcham'; a non-Mitcham proposal would
# count in the headline grade yet vanish from the map beneath it.
_COUNCIL = "mitcham"


def unit_rate_for(asset_type: str) -> float:
    """Footprint unit rate ($/m^2) for ``asset_type`` from the buildings fixture.

    Raises ``KeyError`` for an unknown asset type so a typo surfaces rather than
    silently costing a building at $0/m^2.
    """
    return float(_load_fixture()["unit_rate_per_m2_by_type"][asset_type])


def build_proposed_building(
    *,
    building_id: str,
    status: str,
    name: str,
    asset_type: str,
    suburb: str | None,
    install_year: int,
    condition: float,
    criticality: float,
    grc_total: float | None = None,
    extent: float | None = None,
    unit_rate: float | None = None,
    lat: float | None = None,
    lon: float | None = None,
) -> list[Asset]:
    """One form -> <=7 validated :class:`Asset` component rows (S.3.1).

    Value derivation (exactly one of two paths):
      * **Lump sum:** pass ``grc_total`` directly; ``extent`` is cosmetic (cost
        sampling reads ``grc``, not ``unit_rate x extent``).
      * **Footprint x rate:** pass ``extent`` (and optionally ``unit_rate``; it
        defaults to the fixture rate for ``asset_type``); the building value is
        ``extent * unit_rate``.

    Every row is validated through the strict ``Asset`` model (bounds:
    ``install_year`` 1800-2100, ``condition`` 1-5), carries ``council='mitcham'``
    and ``asset_class='buildings'``, and its ``component`` is asserted to be in
    the seven-component allow-list.
    """
    if grc_total is not None:
        building_value = float(grc_total)
    elif extent is not None:
        rate = float(unit_rate) if unit_rate is not None else unit_rate_for(asset_type)
        building_value = float(extent) * rate
    else:
        raise ValueError(
            "Provide either grc_total (lump sum) or extent (footprint x rate)."
        )

    rows = _components_for_building(
        building_id=building_id,
        name=name,
        asset_type=asset_type,
        suburb=suburb,
        council=_COUNCIL,                       # [FIX-B1] hard-pin
        asset_class="buildings",                # threaded param; Asset forbids extras
        install_year=int(install_year),         # == commission_year for proposed
        condition=float(condition),             # scalar -> applied to all components
        building_value=building_value,          # FINAL value, no rescale
        criticality_seed=float(criticality),
        extent=(float(extent) if extent is not None else None),
        lat=lat,
        lon=lon,
        components=_component_specs(),
    )

    unknown = [r.component for r in rows if r.component not in _ALLOWED_COMPONENTS]
    if unknown:
        raise ValueError(f"Unknown component(s) not in allow-list: {sorted(set(unknown))}")
    assert all(r.council == _COUNCIL for r in rows)
    return rows


def min_component_life(rows: list[Asset]) -> float:
    """Shortest useful life across a proposal's component rows.

    The net-new-liability badge keys off this (the earliest breach): for a
    standard build HVAC at 20 years is the binding component (S.5, [FIX-MINOR1]).
    """
    return min(r.useful_life_years for r in rows)
