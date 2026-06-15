"""Stage 4 — multi-criteria proxy scorers.

Each scorer maps an :class:`Asset` to a 1..5 condition-style index where
1 = excellent and 5 = poor, consistent with the IPWEA condition convention.
The proxies are deliberately transparent: capacity benchmarks floor-area per
resident against per-type norms, while functionality, accessibility and
sustainability blend a per-type base level with an age penalty. They give a
defensible first-pass score where no survey signal exists; a later survey pass
can overwrite the ``proxy`` provenance with observed values.
"""
from __future__ import annotations

from engine.schemas import Asset

CURRENT_YEAR = 2026
RESIDENTS = 67000

# m2-per-resident provision benchmark by asset type (a "good" provision level).
_CAPACITY_BENCHMARK: dict[str, float] = {
    "library": 0.045,
    "community_centre": 0.030,
    "aquatic_centre": 0.060,
    "sports_pavilion": 0.020,
    "civic_centre": 0.040,
    "depot": 0.040,
    "public_toilet": 0.0008,
    "kiosk_shelter": 0.0006,
    "heritage_building": 0.030,
}
_CAPACITY_BENCHMARK_DEFAULT = 0.030

# Functionality age-sensitivity weight by type (higher = degrades faster).
_FUNCTIONALITY_WEIGHT: dict[str, float] = {
    "heritage_building": 0.5,
    "civic_centre": 0.7,
    "library": 0.9,
    "community_centre": 0.9,
    "aquatic_centre": 1.1,
    "sports_pavilion": 1.0,
    "depot": 1.2,
    "public_toilet": 1.1,
    "kiosk_shelter": 1.1,
}
_FUNCTIONALITY_WEIGHT_DEFAULT = 1.0

# Accessibility base level by type (higher = inherently less accessible).
_ACCESSIBILITY_BASE: dict[str, float] = {
    "library": 1.5,
    "community_centre": 2.0,
    "aquatic_centre": 2.2,
    "sports_pavilion": 2.8,
    "civic_centre": 1.8,
    "depot": 4.0,
    "public_toilet": 3.0,
    "kiosk_shelter": 3.4,
    "heritage_building": 3.6,
}
_ACCESSIBILITY_BASE_DEFAULT = 3.0

# Sustainability base level by type (higher = inherently less sustainable).
_SUSTAINABILITY_BASE: dict[str, float] = {
    "aquatic_centre": 4.0,
    "depot": 3.5,
    "civic_centre": 3.0,
    "library": 2.5,
    "community_centre": 2.5,
    "sports_pavilion": 3.0,
    "public_toilet": 3.0,
    "kiosk_shelter": 2.5,
    "heritage_building": 4.0,
}
_SUSTAINABILITY_BASE_DEFAULT = 3.0


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _age(a: Asset) -> int:
    return CURRENT_YEAR - a.install_year


def capacity_score(a: Asset) -> float:
    """Score floor-area provision per resident against a per-type benchmark.

    Returns 3.0 (neutral) when extent is unknown. Otherwise compares the
    asset's m2-per-resident to the type benchmark: ratio >= 2.0 scores 1.0
    (ample), ratio <= 0.2 scores 5.0 (under-provided), with a linear map
    between.
    """
    if a.extent is None:
        return 3.0
    benchmark = _CAPACITY_BENCHMARK.get(a.asset_type, _CAPACITY_BENCHMARK_DEFAULT)
    ratio = (a.extent / RESIDENTS) / benchmark
    if ratio >= 2.0:
        return 1.0
    if ratio <= 0.2:
        return 5.0
    return 5.0 - 4.0 * (ratio - 0.2) / (2.0 - 0.2)


def functionality_score(a: Asset) -> float:
    """Score functional fitness, worsening with age at a per-type rate."""
    age = _age(a)
    age_norm = min(1.0, age / 80)
    w = _FUNCTIONALITY_WEIGHT.get(a.asset_type, _FUNCTIONALITY_WEIGHT_DEFAULT)
    raw = 1.0 + 4.0 * w * age_norm
    return _clamp(raw, 1.0, 5.0)


def accessibility_score(a: Asset) -> float:
    """Score accessibility from a per-type base plus an age penalty."""
    age = _age(a)
    base = _ACCESSIBILITY_BASE.get(a.asset_type, _ACCESSIBILITY_BASE_DEFAULT)
    age_pen = min(1.0, age / 50)
    return _clamp(base + age_pen, 1.0, 5.0)


def sustainability_score(a: Asset) -> float:
    """Score environmental sustainability from a per-type base plus age penalty."""
    age = _age(a)
    base = _SUSTAINABILITY_BASE.get(a.asset_type, _SUSTAINABILITY_BASE_DEFAULT)
    age_pen = min(1.0, age / 60)
    return _clamp(base + age_pen, 1.0, 5.0)
