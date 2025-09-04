from __future__ import annotations
import os
from pathlib import Path
import pandas as pd

RAW_DIR = Path(os.getenv("RAW_DIR", "data/raw"))
RAW_DIR.mkdir(parents=True, exist_ok=True)

def _safe_import(fn, years, name):
    """Call an nfl_data_py import function and tolerate per-year 404s."""
    import nfl_data_py as nfl
    frames = []
    for y in years:
        try:
            df = fn([y])
            if df is None or len(df) == 0:
                continue
            df["season"] = y if "season" not in df.columns else df["season"]
            frames.append(df)
            print(f"[{name}] {y} done.")
        except Exception as e:
            msg = str(e).lower()
            if "404" in msg or "not found" in msg:
                print(f"[{name}] {y} not available yet; skipping.")
                continue
            raise
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def _detect_max_season():
    """Use schedules to infer the latest available season, falling back gracefully."""
    try:
        import nfl_data_py as nfl
        sched = nfl.import_schedules([1999, 2024])  # cheap call to load metadata
        max_year = int(pd.to_datetime(sched.get("gameday")).dt.year.max())
        return max(1999, min(max_year, pd.Timestamp("today").year))
    except Exception:
        # Fall back to current year - 1
        return pd.Timestamp("today").year - 1

def run(seasons_range: list[int] | range | None) -> dict:
    import nfl_data_py as nfl

    # Normalize input (e.g., [2019..2025]) → clamp to available max
    if seasons_range is None:
        seasons_range = range(2019, _detect_max_season() + 1)
    else:
        smin, smax = min(seasons_range), max(seasons_range)
        smax = min(smax, _detect_max_season())
        seasons_range = range(smin, smax + 1)

    years = list(seasons_range)
    print(f"[ETL] Seasons resolved to: {years}")

    out = {}

    # Play-by-play
    pbp = _safe_import(nfl.import_pbp_data, years, "pbp")
    if not pbp.empty:
        p = RAW_DIR / f"pbp_{years[0]}_{years[-1]}.parquet"
        pbp.to_parquet(p, index=False); out["pbp"] = str(p)

    # Weekly
    wk = _safe_import(nfl.import_weekly_data, years, "weekly")
    if not wk.empty:
        p = RAW_DIR / f"weekly_{years[0]}_{years[-1]}.parquet"
        wk.to_parquet(p, index=False); out["weekly"] = str(p)

    # Rosters
    ros = _safe_import(nfl.import_rosters, years, "rosters")
    if not ros.empty:
        p = RAW_DIR / f"rosters_{years[0]}_{years[-1]}.parquet"
        ros.to_parquet(p, index=False); out["rosters"] = str(p)

    # Schedules
    sch = _safe_import(nfl.import_schedules, years, "schedules")
    if not sch.empty:
        p = RAW_DIR / f"schedules_{years[0]}_{years[-1]}.parquet"
        sch.to_parquet(p, index=False); out["schedules"] = str(p)

    # ID map (cross-walk)
    try:
        ids = nfl.import_ids()
        if ids is not None and len(ids) > 0:
            p = RAW_DIR / "ids_latest.parquet"
            ids.to_parquet(p, index=False); out["ids"] = str(p)
    except Exception as e:
        print("IDs import failed (non-fatal):", e)

    return out
