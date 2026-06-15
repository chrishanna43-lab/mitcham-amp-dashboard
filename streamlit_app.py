"""Streamlit entry point — City of Mitcham Buildings Renewal Outlook (LGA-AMP demo).

This is the file a host (e.g. Streamlit Community Cloud) runs. It shows a simple
password screen, locates the demonstration DuckDB, then hands off to the
dashboard renderer in ``engine.render.dashboard.app``.

Access:
  Set an ``APP_PASSWORD`` secret/env var to require a shared password before the
  dashboard loads. If it is unset, the dashboard is open (handy for local dev).

Database resolution order:
  1. ``LGA_AMP_DB``      env var pointing at a local .duckdb file
  2. ``data/lga_amp.duckdb`` committed/built into the repo
  3. ``LGA_AMP_DB_URL``  env var or secret — downloaded once and cached
  4. ``~/.lga-amp/lga_amp.duckdb`` (local-dev default)
"""
from __future__ import annotations

import hmac
import os
import urllib.request
from pathlib import Path

import streamlit as st

from engine.render.dashboard.app import render_dashboard

ROOT = Path(__file__).resolve().parent
REPO_DB = ROOT / "data" / "lga_amp.duckdb"

# Must be the first Streamlit call. render_dashboard() calls it again, so we
# no-op it just before handing off (see bottom of file).
st.set_page_config(
    page_title="City of Mitcham — Buildings Renewal Outlook",
    page_icon="🏛",
    layout="wide",
)


def _secret(name: str) -> str | None:
    """Read a Streamlit secret without exploding when no secrets file exists."""
    try:
        return st.secrets.get(name)  # type: ignore[no-any-return]
    except Exception:
        return None


def _check_password() -> None:
    """Gate the app behind a single shared password (skipped if none is set)."""
    expected = os.environ.get("APP_PASSWORD") or _secret("APP_PASSWORD")
    if not expected:
        return  # no password configured → open access
    if st.session_state.get("_authed"):
        return

    st.markdown("## City of Mitcham — Buildings Renewal Outlook")
    st.caption("Prepared by Social Capital Advisory. Please enter the access password.")
    pw = st.text_input(
        "Password", type="password", label_visibility="collapsed",
        placeholder="Access password",
    )
    if pw:
        if hmac.compare_digest(pw, str(expected)):
            st.session_state["_authed"] = True
            st.rerun()
        st.error("Incorrect password.")
    st.stop()


@st.cache_resource(show_spinner="Fetching the demonstration database…")
def _download_db(url: str, dest: str) -> str:
    """Download the DB once per running instance and cache the path."""
    p = Path(dest)
    if not p.is_file():
        p.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, dest)
    return dest


def _resolve_db() -> Path | None:
    env = os.environ.get("LGA_AMP_DB")
    if env and Path(env).is_file():
        return Path(env)

    if REPO_DB.is_file():
        return REPO_DB

    url = os.environ.get("LGA_AMP_DB_URL") or _secret("LGA_AMP_DB_URL")
    if url:
        return Path(_download_db(url, str(REPO_DB)))

    default = Path.home() / ".lga-amp" / "lga_amp.duckdb"
    if default.is_file():
        return default

    return None


_check_password()

db = _resolve_db()
if db is None:
    st.title("City of Mitcham — Buildings Renewal Outlook")
    st.error(
        "No demonstration database found. Provide one of:\n\n"
        "- a built `data/lga_amp.duckdb` — run `python scripts/build_demo_db.py`, or\n"
        "- an `LGA_AMP_DB_URL` secret/env var pointing at a hosted copy, or\n"
        "- an `LGA_AMP_DB` env var with a local path.\n\n"
        "See the README for deployment options."
    )
    st.stop()

# render_dashboard() calls st.set_page_config() again; we already called it
# above, so neutralise the second call to avoid Streamlit's "can only be called
# once" error.
st.set_page_config = lambda *a, **k: None  # type: ignore[assignment]
render_dashboard(db_path=db, headless=False)
