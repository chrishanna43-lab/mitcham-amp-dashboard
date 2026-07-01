"""Session-scoped shadow DB for the What-If feature (S.2.3).

The canonical DuckDB is **never** written to. A What-If recompute runs against a
file-copy *shadow* that holds **only the input tables** (assets, priors,
climate_factors, criteria_scores); the heavy derived tables (mc_paths /
mc_summary / opt_*) are rebuilt on the shadow, not copied. The canonical is
attached READ_ONLY purely to seed the input tables, then detached, and is opened
``read_only=True`` everywhere else ([FIX-D1]) so any mis-wired write raises
loudly instead of corrupting it.

The shadow is a **single session slot** with explicit teardown ([FIX-D10]):
``open_shadow`` closes and removes the previous shadow before minting a new one;
``atexit`` is a backstop only.
"""
from __future__ import annotations

import atexit
import shutil
import tempfile
from pathlib import Path

from engine import db as _db
from engine.db import bootstrap_schema

# Input tables seeded from canonical. NOT mc_paths / mc_summary / opt_* — those
# are rebuilt on the shadow by run_stage_2_5 / solve_all ([FIX-D4]).
_INPUT_TABLES = ("assets", "priors", "climate_factors", "criteria_scores")

# Sentinel table written into every shadow so ``_is_shadow`` can recognise the
# connection regardless of how it is threaded through the caller.
_SENTINEL_TABLE = "_whatif_shadow_marker"

# Single-slot teardown state (module-level so a new open tears down the prior).
_active_shadow_dir: Path | None = None


def _teardown_active() -> None:
    """Remove the currently-tracked shadow directory, if any."""
    global _active_shadow_dir
    if _active_shadow_dir is not None:
        shutil.rmtree(_active_shadow_dir, ignore_errors=True)
        _active_shadow_dir = None


def open_shadow(canonical_path: str | Path) -> tuple[object, Path]:
    """Build a session shadow DB holding ONLY the input tables; open it read-write.

    The canonical is attached READ_ONLY, the four input tables are copied across,
    then it is detached. Returns ``(conn, shadow_path)``. The previous shadow (if
    any) is torn down first — there is only ever one live slot ([FIX-D10]).
    """
    global _active_shadow_dir
    canonical_path = Path(canonical_path)

    # Single slot: drop the previous shadow before minting a new one.
    _teardown_active()

    shadow_dir = Path(tempfile.mkdtemp(prefix="whatif_"))
    shadow_path = shadow_dir / "whatif.duckdb"

    conn = _db.connect(shadow_path)          # read-write target
    bootstrap_schema(conn)                   # full schema, empty tables
    conn.execute(f"CREATE TABLE {_SENTINEL_TABLE} (ok BOOLEAN)")
    conn.execute(f"INSERT INTO {_SENTINEL_TABLE} VALUES (TRUE)")

    conn.execute(f"ATTACH '{canonical_path.as_posix()}' AS canon (READ_ONLY)")
    try:
        for t in _INPUT_TABLES:
            # Seed by EXPLICIT column name, never positional `SELECT *`: the live
            # canonical's column order can drift from the fresh bootstrap schema
            # (e.g. `suburb` was appended last by a later stage), and a positional
            # copy then casts a string column into a DOUBLE one and blows up. Listing
            # the freshly-bootstrapped table's own columns aligns by name regardless.
            cols = [r[1] for r in conn.execute(f"PRAGMA table_info('{t}')").fetchall()]
            collist = ", ".join(f'"{c}"' for c in cols)
            conn.execute(f"INSERT INTO {t} ({collist}) SELECT {collist} FROM canon.{t}")
    finally:
        conn.execute("DETACH canon")

    assert shadow_path != canonical_path
    assert str(shadow_path).startswith(tempfile.gettempdir())

    _active_shadow_dir = shadow_dir
    atexit.register(lambda: shutil.rmtree(shadow_dir, ignore_errors=True))  # backstop
    return conn, shadow_path


def _is_shadow(conn) -> bool:
    """True iff ``conn`` is a What-If shadow (carries the sentinel table).

    Defence-in-depth (S.2.3): ``recompute_with_new_assets`` asserts this before
    any write, so a canonical (read-only) handle can never be mutated by mistake.
    """
    try:
        row = conn.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_name = ?",
            [_SENTINEL_TABLE],
        ).fetchone()
        return bool(row and row[0])
    except Exception:
        return False


def close_shadow(conn=None) -> None:
    """Close ``conn`` (if given) and tear down the single shadow slot ([FIX-D10])."""
    if conn is not None:
        try:
            conn.close()
        except Exception:
            pass
    _teardown_active()
