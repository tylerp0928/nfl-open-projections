from __future__ import annotations
import os
from pathlib import Path
import pandas as pd

RAW_DIR = Path(os.getenv("RAW_DIR", "data/raw"))
PROC_DIR = Path(os.getenv("PROC_DIR", "data/processed"))

def fetch_betting_lines(years) -> str|None:
    try:
        import nfl_data_py as nfl
    except Exception as e:
        print("nfl_data_py not available:", e)
        return None
    try:
        df = nfl.import_betting_lines(years)  # may not be available in older versions
    except Exception as e:
        print("import_betting_lines failed:", e)
        return None
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    p = RAW_DIR / f"betting_{min(years)}_{max(years)}.parquet"
    df.to_parquet(p, index=False)
    return str(p)

def build_betting_game_features() -> str|None:
    # Reduce betting to pregame closing spread/total per game_id
    files = [p for p in RAW_DIR.glob("betting_*.parquet")]
    if not files:
        return None
    df = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    # Try to standardize column names (book, spread_close, total_close, team)
    cols = {c.lower():c for c in df.columns}
    # Heuristics
    if "spread_close" in cols or "closing_spread" in cols or "spread" in cols:
        pass
    # Keep one line per game: home closing spread and total
    # Attempt to identify home team columns
    candidates = [c for c in df.columns if "home" in c.lower() and "team" in c.lower()]
    if candidates:
        home_col = candidates[0]
    else:
        home_col = "home_team" if "home_team" in df.columns else None

    # If multiple books, choose consensus if available else median
    group = df.groupby("game_id", dropna=False).agg({
        "spread_close":"median" if "spread_close" in df.columns else "min",
        "total_close":"median" if "total_close" in df.columns else "min"
    })
    out = group.reset_index().rename(columns={"spread_close":"closing_spread","total_close":"closing_total"})

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    p = PROC_DIR / "betting_features.parquet"
    out.to_parquet(p, index=False)
    return str(p)
