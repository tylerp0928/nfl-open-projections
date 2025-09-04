# nfl-open-projections

All-open starter kit to compile NFL data (nflverse) and build baseline team & player projections.

**Key ideas**
- Use `nfl_data_py` (open-source) as the spine for play-by-play (1999+), schedules, rosters, and ID mappings.
- Store raw tables as Parquet, compute features (rolling EPA, player usage), and train simple baseline models.
- Everything here is **open** (no paid vendors).

## Quickstart

```bash
# 1) Create env & install
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2) Run the full pipeline (seasons 2019-2024 as example)
python -m src.pipelines.run_all --seasons 2019-2024
```

This will:
1. **ETL**: Download play-by-play, rosters, weekly, schedules, and ID crosswalks via `nfl_data_py` (saves to `data/raw/`).
2. **Features**: Build team rolling EPA ratings and player usage shares (saves to `data/processed/`).
3. **Models**: Train a baseline game win model and produce simple player projections (saves to `data/artifacts/`).

> Tip: If you can't install locally, run the same commands on Colab or any cloud notebook.

## Sources (open)
- nflverse / nflfastR play-by-play (1999+) and models (EPA/WP)
- nflreadr player/ID tables and participation data
- nfl_data_py Python interface wrapping the above

## Notes
- This is a minimal baseline meant for extension.
- Respect data source terms. This kit avoids scraping sites that disallow it.

---

## Extended pipeline (betting + context + season sims)

```bash
python -m src.pipelines.run_extended --seasons 2019-2024
```

**Adds**:
- Optional **betting** features (closing spread/total) if your `nfl_data_py` version exposes `import_betting_lines`. Falls back gracefully if not.
- **Context** features: rest days, travel distance (stadium-to-stadium), and dome/indoor indicator.
- **Extended game model** using all features (calibrated logistic).
- **Player stat projections** (targets, receiving yards/TDs; carries, rush yards/TDs) with simple empirical-Bayes shrinkage.
- **Season Monte Carlo** (win totals + naive playoff odds) for the last season in your range.

**Artifacts** land in `data/artifacts/`:
- `game_win_extended.joblib`, `game_win_extended_metrics.json`
- `player_stat_projections.csv`
- `season_<YEAR>_sim_summary.csv`


## CI/CD & Hosting

### Option A — GitHub + Actions (recommended)
1. Create a private GitHub repo and push this project.
2. Actions will run **daily at 13:00 UTC** (9am ET) by default. Trigger manually via “Run workflow” with a custom seasons range.
3. (Optional) To push artifacts back to the repo on an `artifacts` branch, add a repo secret named **ARTIFACTS_PAT** (a Personal Access Token with `repo` scope) and set a workflow variable **PUSH_ARTIFACTS=true**.

### Option B — GitHub Codespaces
- Open the repo in Codespaces. The `.devcontainer` installs deps automatically. Run:
  ```bash
  make run-ext SEASONS=2019-2025
  ```

### Option C — Docker
- Build & run anywhere:
  ```bash
  make docker-build
  make docker-run
  ```

---

## How to collaborate with ChatGPT
- Share the GitHub repo URL in chat. Ask for changes; I’ll provide patch diffs or file replacements you can paste into PRs.
- I can also generate new workflow files, notebooks, or modules; you then commit them.
- Want reminders? Ask me to set weekly/daily check-ins and I’ll ping you with notes and suggested PRs.


## Injuries & Weekly HTML Slate Report
- The pipeline tries to pull injuries via `nfl_data_py` (`import_injuries` / `import_weekly_injuries` / `import_injury_reports`) and stores them to `data/raw/`.
- We compute **injury multipliers** per player (`Active`=1.00, `Questionable`=0.80, `Doubtful`=0.25, `Out/IR/PUP/Suspended`=0.00, etc.) and apply them to usage projections.
- Generate a **Weekly Slate Report** (HTML) combining game probabilities and top projected players (injury-adjusted).

**Run**
```bash
python -m src.pipelines.run_extended --seasons 2019-2025
# Then create a week report (if not made automatically):
python -m src.models.predict_game_week --season 2025 --week 1
python -m src.reports.slate_report -- (module has build_weekly_slate_report; see run_extended)
```
Artifacts:
- `player_usage_projections_injury_adj.csv`
- `slate_report_<SEASON>_wk<week>.html`


## Web App (Streamlit) — Free UI
You can run a full UI locally or on **Streamlit Community Cloud**.

### Run locally
```bash
streamlit run streamlit_app.py
```

### Deploy on Streamlit Cloud (free)
1. Push this project to a **private GitHub repo**.
2. Go to https://streamlit.io/cloud and deploy from that repo.
3. (Optional) In **App → Settings → Secrets**, set:
   ```
   GH_REPO = yourname/yourrepo
   GH_TOKEN = <personal access token with repo scope>
   ```
   Then use the **Admin** tab to trigger the GitHub Actions workflow to refresh data/artifacts remotely.

**The app reads artifacts from the repo workspace at runtime** (`data/processed`, `data/artifacts`). Your scheduled GitHub Action will refresh those files daily, and the app shows the newest outputs.
