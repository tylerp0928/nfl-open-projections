from __future__ import annotations
import os
from pathlib import Path
import pandas as pd

PROC_DIR = Path(os.getenv("PROC_DIR", "data/processed"))
ART_DIR = Path(os.getenv("ART_DIR", "data/artifacts"))

def _ensure_dirs():
    ART_DIR.mkdir(parents=True, exist_ok=True)

def build_simple_usage_projections() -> str:
    _ensure_dirs()
    usage = pd.read_parquet(PROC_DIR / "player_usage.parquet")

    # Forecast next game's usage as EWMA of last 4 (carry/target shares)
    usage = usage.sort_values(["player_id","season","week"])
    usage["carry_share_fcast"] = usage.groupby("player_id")["carry_share"].transform(lambda s: s.ewm(span=4, adjust=False).mean())
    usage["target_share_fcast"] = usage.groupby("player_id")["target_share"].transform(lambda s: s.ewm(span=4, adjust=False).mean())

    # Take most recent obs per player-season-week as "projection for next week"
    latest = usage.groupby(["player_id","season"]).tail(1).copy()
    out = latest[["player_id","player_name","posteam","season","week","carry_share_fcast","target_share_fcast"]].copy()
    out.rename(columns={
        "carry_share_fcast":"proj_carry_share_next",
        "target_share_fcast":"proj_target_share_next"
    }, inplace=True)

    out_path = ART_DIR / "player_usage_projections.csv"
    out.to_csv(out_path, index=False)
    return str(out_path)
