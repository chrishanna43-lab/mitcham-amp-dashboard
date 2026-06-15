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
dashboard, plus a bundled demo database. It deliberately excludes SCA's
methodology, sales positioning, research corpus, and any client-confidential
material.

---

## Sharing it (the short version)

1. Deploy this repo to **Streamlit Community Cloud** (see below) → you get a link
   like `https://your-app.streamlit.app`.
2. Set an **`APP_PASSWORD`** secret on the app.
3. Email the **link + password** to whoever you like. They click it, type the
   password, and the dashboard opens — no GitHub, no account, no install.

---

## Deploy to Streamlit Community Cloud

1. Push this repo to GitHub (a **private** repo is fine).
2. Go to https://share.streamlit.io → **Create app** → **Deploy a public app
   from GitHub**.
3. Pick this repo, branch `main`, main file **`streamlit_app.py`**. Click **Deploy**.
4. Open **⚙ Settings → Secrets** and add:
   ```toml
   APP_PASSWORD = "choose-a-password"
   ```
   Save. The app reboots with the password screen enabled.
5. Copy the app URL and email it with the password.

The demo database is bundled in the repo (`data/lga_amp.duckdb`, ~95 MB), so the
app loads immediately — nothing else to host.

> **Note on access:** this is a single shared password suitable for a demo, not
> hardened authentication. For per-person access instead, use Streamlit Cloud's
> built-in viewer allowlist (Settings → Sharing) and you can leave `APP_PASSWORD`
> unset.

---

## Run it locally

```bash
python -m venv .venv && .venv\Scripts\activate     # Windows
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Open http://localhost:8501. With no `APP_PASSWORD` set, it runs unlocked. The
bundled `data/lga_amp.duckdb` is used automatically.

---

## The database

The bundled `data/lga_amp.duckdb` is a **right-sized build** (16 Monte Carlo
realisations) kept under GitHub's 100 MB file limit and committed as a normal
file, so the hosted app loads instantly. The headline figures (renewal demand,
the ~$2.8M unfunded gap at $0.8M/yr, the works programme) match the full demo;
only the Monte Carlo uncertainty band is coarser than a full-resolution run.

To regenerate it — or build a fuller-resolution version locally:

```bash
python scripts/build_demo_db.py            # default (n=16, ~95 MB)
python scripts/build_demo_db.py --n 200    # full resolution (~500 MB; host separately)
```

`streamlit_app.py` resolves the database in this order: `LGA_AMP_DB` env var →
`data/lga_amp.duckdb` → `LGA_AMP_DB_URL` (downloaded + cached) →
`~/.lga-amp/lga_amp.duckdb`.

---

## The solver

The renewal optimiser is a two-stage stochastic CVaR program (cvxpy). It uses
**MOSEK** when installed and licensed; otherwise it falls back **automatically**
to the open-source **HiGHS** solver bundled with cvxpy. `mosek` is intentionally
left out of `requirements.txt`, so cloud deployments use HiGHS with no licence —
the live "Engagement weighting" and "Force into programme" re-solves work, just a
little slower.

---

## Layout

```
streamlit_app.py        # entry point — password gate, DB resolution, runs the dashboard
requirements.txt        # runtime deps (HiGHS solver; no MOSEK)
data/lga_amp.duckdb      # bundled demo database (~95 MB)
scripts/build_demo_db.py# regenerate the demo DuckDB from bundled fixtures
engine/                 # the renewal engine + dashboard
  render/dashboard/app.py   # the four-view Streamlit dashboard
.streamlit/config.toml  # Streamlit chrome theme + server settings
```

---

*Prepared by Social Capital Advisory. Demonstration on synthesised public data —
decision support, not decision authority.*
