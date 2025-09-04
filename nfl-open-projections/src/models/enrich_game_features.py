from __future__ import annotations
import os
from pathlib import Path
import pandas as pd

PROC_DIR = Path(os.getenv("PROC_DIR", "data/processed"))
RAW_DIR = Path(os.getenv("RAW_DIR", "data/raw"))

def build_game_model_table() -> str:
    ratings = pd.read_parquet(PROC_DIR / "team_ratings.parquet")
    context = pd.read_parquet(PROC_DIR / "context_features.parquet")
    sched_files = [p for p in RAW_DIR.glob("schedules_*.parquet")]
    schedules = pd.read_parquet(sched_files[0]) if len(sched_files)==1 else pd.concat([pd.read_parquet(p) for p in sched_files], ignore_index=True)
    # Optional betting
    bet_path = PROC_DIR / "betting_features.parquet"
    betting = pd.read_parquet(bet_path) if bet_path.exists() else None

    # Base ratings for home & away
    base = ratings[["season","week","game_id","team","off_epa_pp_roll","def_epa_pp_roll","net_epa_rating"]]
    home = schedules[["game_id","season","week","home_team","away_team","home_score","away_score"]].merge(
        base.rename(columns={"team":"home_team"}),
        on=["game_id","season","week","home_team"], how="left"
    )
    full = home.merge(
        base.rename(columns={"team":"away_team",
                             "off_epa_pp_roll":"away_off_roll",
                             "def_epa_pp_roll":"away_def_roll",
                             "net_epa_rating":"away_net"}),
        on=["game_id","season","week","away_team"], how="left"
    )

    # Context for both teams (pre-game values)
    ctx = context.rename(columns={"team":"home_team","rest_days":"home_rest","travel_km":"home_travel","is_dome_like":"home_dome"})
    full = full.merge(ctx[["game_id","home_team","home_rest","home_travel","home_dome"]], on=["game_id","home_team"], how="left")

    ctx2 = context.rename(columns={"team":"away_team","rest_days":"away_rest","travel_km":"away_travel","is_dome_like":"away_dome"})
    full = full.merge(ctx2[["game_id","away_team","away_rest","away_travel","away_dome"]], on=["game_id","away_team"], how="left")

    # Feature diffs
    full["net_diff"] = full["net_epa_rating"] - full["away_net"]
    full["off_diff"] = full["off_epa_pp_roll"] - full["away_off_roll"]
    full["def_diff"] = full["def_epa_pp_roll"] - full["away_def_roll"]
    full["rest_diff"] = full["home_rest"] - full["away_rest"]
    full["travel_diff_km"] = full["home_travel"] - full["away_travel"]
    full["dome_any"] = ((full["home_dome"]==1) | (full["away_dome"]==1)).astype(int)

    if betting is not None:
        full = full.merge(betting, on="game_id", how="left")

    out = full
    out["home_win"] = (out["home_score"] > out["away_score"]).astype(int)

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    p = PROC_DIR / "game_model_table.parquet"
    out.to_parquet(p, index=False)
    return str(p)
