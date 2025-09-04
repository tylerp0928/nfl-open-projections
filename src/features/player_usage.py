from __future__ import annotations
import os
from pathlib import Path
import pandas as pd

RAW_DIR = Path(os.getenv("RAW_DIR", "data/raw"))
PROC_DIR = Path(os.getenv("PROC_DIR", "data/processed"))

def _save(df: pd.DataFrame, name: str) -> str:
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    out = PROC_DIR / name
    df.to_parquet(out, index=False)
    return str(out)

def build_player_usage() -> str:
    pbp_files = [p for p in RAW_DIR.glob("pbp_*.parquet")]
    assert pbp_files, "No PBP parquet found. Run ETL first."
    pbp = pd.read_parquet(pbp_files[0]) if len(pbp_files)==1 else pd.concat([pd.read_parquet(p) for p in pbp_files], ignore_index=True)

    # Filter to pass & rush plays
    pbp = pbp.loc[(pbp.get("rush_attempt",0)==1) | (pbp.get("pass_attempt",0)==1)].copy()

    # --- Carries (rusher on rush attempts)
    rush = pbp.loc[pbp.get("rush_attempt",0)==1, ["season","week","game_id","posteam","rusher_player_id","rusher_player_name"]].copy()
    rush["carries"] = 1
    carries = (rush.groupby(["season","week","game_id","posteam","rusher_player_id","rusher_player_name"], dropna=False)["carries"]
                    .sum().reset_index())

    team_carries = (rush.groupby(["season","week","game_id","posteam"], dropna=False)["carries"]
                        .sum().reset_index().rename(columns={"carries":"team_carries"}))

    # --- Targets (receiver_player_id on pass attempts)
    pas = pbp.loc[pbp.get("pass_attempt",0)==1, ["season","week","game_id","posteam","receiver_player_id","receiver_player_name","complete_pass"]].copy()
    pas["targets"] = 1
    pas["receptions"] = pas["complete_pass"].fillna(0).astype(int)
    tgts = (pas.groupby(["season","week","game_id","posteam","receiver_player_id","receiver_player_name"], dropna=False)
                .agg(targets=("targets","sum"), receptions=("receptions","sum"))
                .reset_index())

    team_targets = (pas.groupby(["season","week","game_id","posteam"], dropna=False)["targets"]
                       .sum().reset_index().rename(columns={"targets":"team_targets"}))

    # Merge and compute shares
    usage = pd.merge(carries, team_carries, on=["season","week","game_id","posteam"], how="outer")
    usage = usage.merge(tgts, left_on=["season","week","game_id","posteam","rusher_player_id","rusher_player_name"],
                        right_on=["season","week","game_id","posteam","receiver_player_id","receiver_player_name"],
                        how="outer", suffixes=("_rush","_rec"))

    # Unify player id/name columns
    usage["player_id"] = usage["rusher_player_id"].fillna(usage["receiver_player_id"])
    usage["player_name"] = usage["rusher_player_name"].fillna(usage["receiver_player_name"])

    # Fill team totals for share calc
    usage["team_carries"] = usage["team_carries"].fillna(0)
    usage["team_targets"] = usage["team_targets"].fillna(0)
    usage["carries"] = usage["carries"].fillna(0)
    usage["targets"] = usage["targets"].fillna(0)
    usage["receptions"] = usage["receptions"].fillna(0)

    usage["carry_share"] = usage.apply(lambda r: r["carries"]/r["team_carries"] if r["team_carries"]>0 else 0, axis=1)
    usage["target_share"] = usage.apply(lambda r: r["targets"]/r["team_targets"] if r["team_targets"]>0 else 0, axis=1)

    # Rolling usage (per player across weeks)
    usage = usage.sort_values(["player_id","season","week"])
    usage["carry_share_roll4"] = usage.groupby("player_id")["carry_share"].transform(lambda s: s.shift(1).rolling(4, min_periods=1).mean())
    usage["target_share_roll4"] = usage.groupby("player_id")["target_share"].transform(lambda s: s.shift(1).rolling(4, min_periods=1).mean())

    return _save(usage, "player_usage.parquet")
