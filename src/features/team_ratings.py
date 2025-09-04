from __future__ import annotations
import os
from pathlib import Path
import pandas as pd

RAW_DIR = Path(os.getenv("RAW_DIR", "data/raw"))
PROC_DIR = Path(os.getenv("PROC_DIR", "data/processed"))

def _read_parquet(name: str) -> pd.DataFrame:
    return pd.read_parquet(RAW_DIR / name)

def _save(df: pd.DataFrame, name: str) -> str:
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    out = PROC_DIR / name
    df.to_parquet(out, index=False)
    return str(out)

def build_team_epa_rolling(window:int=8) -> str:
    pbp_files = [p for p in RAW_DIR.glob("pbp_*.parquet")]
    assert pbp_files, "No PBP parquet found. Run ETL first."
    pbp = pd.read_parquet(pbp_files[0]) if len(pbp_files)==1 else pd.concat([pd.read_parquet(p) for p in pbp_files], ignore_index=True)

    # Keep scrimmage plays only
    pbp = pbp.loc[pbp["play_type"].isin(["pass","run"]) | ((pbp.get("rush_attempt",0)==1) | (pbp.get("pass_attempt",0)==1))].copy()

    # Build per-game EPA/play for offense and defense
    # Offense
    off = (pbp.groupby(["season","week","game_id","posteam"], dropna=False)
              .agg(plays=("epa","size"), epa_sum=("epa","sum"))
              .reset_index())
    off = off.rename(columns={"posteam":"team"})
    off["epa_per_play"] = off["epa_sum"] / off["plays"]

    # Defense (by defteam)
    deff = (pbp.groupby(["season","week","game_id","defteam"], dropna=False)
              .agg(d_plays=("epa","size"), d_epa_sum=("epa","sum"))
              .reset_index()
              .rename(columns={"defteam":"team"}))
    deff["def_epa_per_play_allowed"] = deff["d_epa_sum"] / deff["d_plays"]

    df = off.merge(deff[["season","week","game_id","team","def_epa_per_play_allowed"]],
                   on=["season","week","game_id","team"], how="left")

    # Sort chronologically then rolling by team
    df = df.sort_values(["team","season","week"])
    df["off_epa_pp_roll"] = (df.groupby("team")["epa_per_play"]
                               .transform(lambda s: s.shift(1).rolling(window, min_periods=3).mean()))
    df["def_epa_pp_roll"] = (df.groupby("team")["def_epa_per_play_allowed"]
                               .transform(lambda s: s.shift(1).rolling(window, min_periods=3).mean()))

    # A simple net rating (lower defensive EPA allowed is better; subtract)
    df["net_epa_rating"] = df["off_epa_pp_roll"] - df["def_epa_pp_roll"]

    return _save(df, "team_ratings.parquet")
