from __future__ import annotations
import os
from pathlib import Path
import pandas as pd

RAW_DIR = Path(os.getenv("RAW_DIR", "data/raw"))
RAW_DIR.mkdir(parents=True, exist_ok=True)

def _safe_import(callable_fn, years, name: str) -> pd.DataFrame:
    """
    Run an nfl_data_py import function one season at a time.
    Swallow 404/Not Found for any season and keep going.
    Never raise — always return a DataFrame (possibly empty).
    """
    import nfl_data_py as nfl  # noqa: F401  (ensures package is present)
    parts = []
    for y in years:
        try:
            df = callable_fn([y])
            if df is None or len(df) == 0:
                print(f"[{name}] {y} empty; skipping.")
                continue
            # tiny memory saver: downcast floats if present
            for c in df.select_dtypes(include="float").columns:
                df[c] = pd.to_numeric(df[c], downcast="float")
            print(f"[{name}] {y} done.")
            parts.append(df)
        except Exception as e:
            msg = (str(e) or "").lower()
            if "404" in msg or "not found" in msg:
                print(f"[{name}] {y} not available; skipping.")
                continue
            # non-404 error: log and keep going instead of aborting
            print(f"[{name}] {y} failed with: {e!r} — skipping.")
            continue
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    return out

def _resolve_years(user_years) -> list[int]:
    # If user supplied a range/list, clamp to <= current year
    if user_years is None:
        smin, smax = 2019, pd.Timestamp.today().year - 0  # allow current year if available
    else:
        smin, smax = min(user_years), max(user_years)
    # don’t exceed current year
    smax = min(smax, pd.Timestamp.today().year)
    return list(range(smin, smax + 1))

def run(seasons) -> dict:
    """
    Downloads core nflverse tables into data/raw/*.parquet.
    Returns dict of file paths. Never raises — downstream can proceed.
    """
    import nfl_data_py as nfl

    years = _resolve_years(seasons)
    print(f"[ETL] Seasons resolved to: {years}")
    out: dict[str, str] = {}

    try:
        pbp = _safe_import(nfl.import_pbp_data, years, "pbp")
        if not pbp.empty:
            p = RAW_DIR / f"pbp_{years[0]}_{years[-1]}.parquet"
            pbp.to_parquet(p, index=False)
            out["pbp"] = str(p)
    except Exception as e:
        print("[pbp] save failed:", e)

    try:
        wk = _safe_import(nfl.import_weekly_data, years, "weekly")
        if not wk.empty:
            p = RAW_DIR / f"weekly_{years[0]}_{years[-1]}.parquet"
            wk.to_parquet(p, index=False)
            out["weekly"] = str(p)
    except Exception as e:
        print("[weekly] save failed:", e)

    try:
        rosters = _safe_import(nfl.import_rosters, years, "rosters")
        if not rosters.empty:
            p = RAW_DIR / f"rosters_{years[0]}_{years[-1]}.parquet"
            rosters.to_parquet(p, index=False)
            out["rosters"] = str(p)
    except Exception as e:
        print("[rosters] save failed:", e)

    try:
        schedules = _safe_import(nfl.import_schedules, years, "schedules")
        if not schedules.empty:
            p = RAW_DIR / f"schedules_{years[0]}_{years[-1]}.parquet"
            schedules.to_parquet(p, index=False)
            out["schedules"] = str(p)
    except Exception as e:
        print("[schedules] save failed:", e)

    # ID map is optional
    try:
        ids = nfl.import_ids()
        if ids is not None and len(ids) > 0:
            p = RAW_DIR / "ids_latest.parquet"
            ids.to_parquet(p, index=False)
            out["ids"] = str(p)
    except Exception as e:
        print("[ids] import failed (non-fatal):", e)

    print("[ETL] wrote:", out)
    return out
