# City of Mitcham — Buildings Renewal Outlook (LGA-AMP demonstration)

A live, branded decision-support dashboard for council building-renewal planning.
It reads a modelled outlook — climate-aware deterioration, a Monte Carlo funding
range, and a budget-constrained renewal optimiser — and presents it across four
views: **The Call**, **The Plan**, **The Portfolio**, and **The Trade-offs**.

Built by [Social Capital Advisory](https://socialcapitaladvisory.com.au) on the
City of Mitcham's buildings portfolio using **synthesised public data**. It is a
demonstration: decision support, not decision authority. Figures are a realistic
stand-in, not the council's actual asset records, and require chartered-engineer
sign-off before any real capital decision.

This repository is the **deployment package only** — the runnable engine and
dashboard. It deliberately excludes SCA's methodology, sales positioning,
research corpus, and any client-confidential material.

---

## Run it locally

```bash
python -m venv .venv && .venv\Scripts\activate     # Windows
pip install -r requirements.txt

# Build the demonstration database (writes data/lga_amp.duckdb)
python scripts/build_demo_db.py

# Launch
streamlit run streamlit_app.py
```

Open http://localhost:8501. Use the sidebar to change the climate scenario and
annual renewal budget; every view updates live.

---

## The database

The dashboard needs a DuckDB file. The full demo (`--n 200`) is **~500 MB**, which
exceeds GitHub's 100 MB per-file limit — so it is **not committed** (`.gitignore`
excludes `*.duckdb`). `streamlit_app.py` looks for it in this order:

1. `LGA_AMP_DB` — env var with a local `.duckdb` path
2. `data/lga_amp.duckdb` — built or committed into the repo
3. `LGA_AMP_DB_URL` — env var / secret; downloaded once and cached
4. `~/.lga-amp/lga_amp.duckdb` — the local-dev default

Pick whichever fits your host:

| Option | How | Best for |
|---|---|---|
| **Build it** | `python scripts/build_demo_db.py` | local use; CI build steps |
| **Host it + URL** | upload the `.duckdb` to a GitHub Release asset / object store, set `LGA_AMP_DB_URL` as a secret | **Streamlit Community Cloud** (keeps the repo small, exact demo figures) |
| **Git LFS** | `git lfs install`, build a slim DB (`--n 80`), commit it (`.gitattributes` already tracks `*.duckdb`) | self-contained repo, smaller demo |

> The hosted-URL route is recommended for cloud: it keeps the repo lean and
> preserves the exact headline figures from the full 200-realisation run.

---

## The solver

The renewal optimiser is a two-stage stochastic CVaR program (cvxpy). It uses
**MOSEK** when installed and licensed; otherwise it falls back **automatically**
to the open-source **HiGHS** solver bundled with cvxpy
(`engine/optimise/stochastic.py::_solve`). `mosek` is intentionally left out of
`requirements.txt`, so cloud deployments use HiGHS with no licence — the live
"Engagement weighting" and "Force into programme" re-solves work, just a little
slower (a re-solve takes a few seconds to ~30 s).

---

## Deploy to Streamlit Community Cloud

1. Push this repo to GitHub (a **private** repo is fine — Streamlit Cloud deploys
   from private repos, and keeps the engine source off the public web).
2. Host the demo DB (build it, then upload to a GitHub Release or object store).
3. On https://share.streamlit.io → **New app**, point it at this repo and
   `streamlit_app.py`.
4. In the app's **Settings → Secrets**, add:
   ```toml
   LGA_AMP_DB_URL = "https://…/lga_amp.duckdb"
   ```
5. **Restrict access** (this is a client demo): in **Settings → Sharing**, turn
   off public access and add a viewer allowlist of the specific emails who may
   open it. (Alternatively add a password gate in `streamlit_app.py`.)

The first load downloads and caches the DB, then serves the dashboard.

---

## Layout

```
streamlit_app.py        # entry point — resolves the DB, runs the dashboard
requirements.txt        # runtime deps (HiGHS solver; no MOSEK)
scripts/build_demo_db.py# regenerate the demo DuckDB from bundled fixtures
engine/                 # the renewal engine + dashboard
  render/dashboard/app.py   # the four-view Streamlit dashboard
  ingest/ climate/ calibrate/ score/ simulate/ optimise/ engage/ geo/ render/
.streamlit/config.toml  # Streamlit chrome theme + server settings
```

---

*Prepared by Social Capital Advisory. Demonstration on synthesised public data —
decision support, not decision authority.*
