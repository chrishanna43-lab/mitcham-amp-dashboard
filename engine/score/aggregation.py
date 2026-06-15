"""Stage 4 — assemble and persist the four-axis criterion scores.

Reads the asset register, applies each proxy scorer across the four criteria
axes (capacity, functionality, accessibility, sustainability), and replaces the
``criteria_scores`` table with one ``proxy`` row per asset per axis.
"""
from __future__ import annotations

from engine.schemas import Asset, Axis, CriterionScore
from engine.score.proxies import (
    accessibility_score,
    capacity_score,
    functionality_score,
    sustainability_score,
)

# (axis, scorer) pairs evaluated for every asset, in stable axis order.
_AXES: tuple[tuple[Axis, object], ...] = (
    ("capacity", capacity_score),
    ("functionality", functionality_score),
    ("accessibility", accessibility_score),
    ("sustainability", sustainability_score),
)

_COLUMNS = ("asset_id", "axis", "score", "provenance")


def score_all(conn) -> int:
    """Derive and persist proxy criterion scores for every asset.

    Rebuilds :class:`Asset` rows from ``assets``, replaces all rows in
    ``criteria_scores`` and bulk-writes four ``proxy`` rows per asset.
    Returns the number of criterion-score rows written.
    """
    from engine.db import insert_models

    rows = conn.execute("SELECT * FROM assets").fetchall()
    cols = [d[0] for d in conn.description]
    assets = [Asset(**dict(zip(cols, r))) for r in rows]

    conn.execute("DELETE FROM criteria_scores")

    out: list[CriterionScore] = []
    for a in assets:
        for axis, fn in _AXES:
            out.append(
                CriterionScore(
                    asset_id=a.asset_id,
                    axis=axis,
                    score=fn(a),
                    provenance="proxy",
                )
            )

    insert_models(conn, "criteria_scores", _COLUMNS, out)
    return len(out)


def write_scores(conn) -> int:
    """Alias for :func:`score_all` — derive and persist all criterion scores."""
    return score_all(conn)
