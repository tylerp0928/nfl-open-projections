from __future__ import annotations
import os
from pathlib import Path
import pandas as pd

RAW_DIR = Path(os.getenv("RAW_DIR", "data/raw"))
PROC_DIR = Path(os.getenv("PROC_DIR", "data/processed"))

# Simple usage multipliers by status (tunable)
INJURY_MULTIPLIERS = {
    "IR": 0.00,
    "PUP": 0.00,
    "Suspended": 0.00,
    "Out": 0.00,
    "Doubtful": 0.25,
    "Questionable": 0.80,
    "Probable": 0.95,
    "Active": 1.00,
    "Rest": 0.85,
}

def build_injury_adjustments() -> str|None:
    files = [p for p in RAW_DIR.glob("injuries_*.parquet")]
    if not files:
        print("No injuries parquet found; skipping injury adjustments.")
        return None
    inj = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)

    # Harmonize key columns
    cols = inj.columns.str.lower()
    inj.columns = cols
    # Try to find consistent keys
    candidates_id = [c for c in inj.columns if "player_id" in c]
    pid = candidates_id[0] if candidates_id else None
    team_col = "team" if "team" in inj.columns else ("posteam" if "posteam" in inj.columns else None)
    season_col = "season" if "season" in inj.columns else None
    week_col = "week" if "week" in inj.columns else ("game_week" if "game_week" in inj.columns else None)
    status_col = "status_norm" if "status_norm" in inj.columns else ("status" if "status" in inj.columns else None)

    if not all([pid, team_col, season_col, week_col, status_col]):
        print("Injury table missing required keys; skipping.")
        return None

    adj = inj[[season_col, week_col, team_col, pid, status_col]].copy()
    adj = adj.rename(columns={season_col:"season", week_col:"week", team_col:"team", pid:"player_id", status_col:"status"})
    adj["status"] = adj["status"].fillna("Active")
    adj["inj_multiplier"] = adj["status"].map(INJURY_MULTIPLIERS).fillna(0.9)

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    p = PROC_DIR / "injury_adjustments.parquet"
    adj.to_parquet(p, index=False)
    return str(p)
