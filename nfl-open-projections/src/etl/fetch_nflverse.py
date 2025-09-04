from __future__ import annotations
import os
from pathlib import Path
from typing import Iterable
import pandas as pd
import nfl_data_py as nfl

RAW_DIR = Path(os.getenv("RAW_DIR", "data/raw"))

def _save_parquet(df: pd.DataFrame, name: str) -> str:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out = RAW_DIR / f"{name}.parquet"
    df.to_parquet(out, index=False)
    return str(out)

def fetch_pbp(years: Iterable[int]) -> str:
    # Pull core play-by-play (1999+). Columns omitted for simplicity to keep schema wide.
    df = nfl.import_pbp_data(list(years), downcast=False)
    return _save_parquet(df, f"pbp_{min(years)}_{max(years)}")

def fetch_weekly(years: Iterable[int]) -> str:
    df = nfl.import_weekly_data(list(years), downcast=False)
    return _save_parquet(df, f"weekly_{min(years)}_{max(years)}")

def fetch_rosters(years: Iterable[int]) -> str:
    df = nfl.import_rosters(list(years))
    return _save_parquet(df, f"rosters_{min(years)}_{max(years)}")

def fetch_schedules(years: Iterable[int]) -> str:
    df = nfl.import_schedules(list(years))
    return _save_parquet(df, f"schedules_{min(years)}_{max(years)}")

def fetch_ids() -> str:
    # Crosswalk: ids across sites (gsis_id, pfr_id, pff_id, sportradar_id, etc.)
    df = nfl.import_ids()
    return _save_parquet(df, "id_map")

def run(years: Iterable[int]) -> dict:
    return {
        "pbp": fetch_pbp(years),
        "weekly": fetch_weekly(years),
        "rosters": fetch_rosters(years),
        "schedules": fetch_schedules(years),
        "ids": fetch_ids(),
    }
