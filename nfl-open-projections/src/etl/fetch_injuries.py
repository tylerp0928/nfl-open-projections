from __future__ import annotations
import os
from pathlib import Path
import pandas as pd

RAW_DIR = Path(os.getenv("RAW_DIR", "data/raw"))

STATUS_NORMALIZE = {
    "IR":"IR","PUP":"PUP","QUESTIONABLE":"Questionable","DOUBTFUL":"Doubtful",
    "OUT":"Out","SUSPENDED":"Suspended","PROBABLE":"Probable",
    "ACTIVE":"Active","REST":"Rest"
}

def fetch_injuries(years) -> str|None:
    """Try multiple nfl_data_py entry points for injuries and return parquet path or None."""
    try:
        import nfl_data_py as nfl
    except Exception as e:
        print("nfl_data_py import failed:", e)
        return None

    df = None
    # Try common functions that exist across versions
    for fn in ("import_injuries","import_weekly_injuries","import_injury_reports"):
        if hasattr(nfl, fn):
            try:
                df = getattr(nfl, fn)(years)
                break
            except Exception as e:
                print(f"{fn} failed: {e}")
                df = None

    if df is None or len(df)==0:
        print("No injuries data available from nfl_data_py for these seasons.")
        return None

    # Normalize status text
    for c in ("status","player_status","injury_status"):
        if c in df.columns:
            df["status_norm"] = df[c].astype(str).str.upper().map(STATUS_NORMALIZE).fillna(df[c].astype(str))
            break
    if "status_norm" not in df.columns:
        df["status_norm"] = "Active"

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    p = RAW_DIR / f"injuries_{min(years)}_{max(years)}.parquet"
    df.to_parquet(p, index=False)
    return str(p)
