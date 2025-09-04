from __future__ import annotations
import os
from pathlib import Path
import pandas as pd
import numpy as np

RAW_DIR = Path(os.getenv("RAW_DIR", "data/raw"))
PROC_DIR = Path(os.getenv("PROC_DIR", "data/processed"))
ART_DIR = Path(os.getenv("ART_DIR", "data/artifacts"))

def _col(df: pd.DataFrame, candidates: list[str], default: str | None = None) -> str | None:
    """Return the first column name that exists (case-insensitive)."""
    cols = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in cols:
            return cols[c.lower()]
    return default

def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    out = a.astype(float)
    out = out / b.replace({0: np.nan})
    return out.fillna(0.0)

def build_player_usage() -> str:
    # ---- load weekly parquet (created by fetch_nflverse) ----
    weekly_parqs = sorted(RAW_DIR.glob("weekly_*.parquet"))
    if not weekly_parqs:
        raise FileNotFoundError("weekly parquet not found in data/raw; run ETL first.")
    wk = pd.read_parquet(weekly_parqs[-1])  # most recent combined file

    # ---- normalize column names & pick keys ----
    # handle different nfl_data_py versions
    pid = _col(wk, ["player_id", "gsis_id", "pfr_id"])
    pname = _col(wk, ["player_name", "player", "name"])
    team = _col(wk, ["recent_team", "team", "posteam"])
    season = _col(wk, ["season"])
    week = _col(wk, ["week", "game_week"])

    targets = _col(wk, ["targets", "target"])
    rush_att = _col(wk, ["rush]()
