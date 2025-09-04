from __future__ import annotations
import os
from pathlib import Path
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
import joblib

PROC_DIR = Path(os.getenv("PROC_DIR", "data/processed"))
RAW_DIR = Path(os.getenv("RAW_DIR", "data/raw"))
ART_DIR = Path(os.getenv("ART_DIR", "data/artifacts"))

def _ensure_dirs():
    ART_DIR.mkdir(parents=True, exist_ok=True)

def train_and_save():
    _ensure_dirs()
    ratings = pd.read_parquet(PROC_DIR / "team_ratings.parquet")
    sched_files = [p for p in RAW_DIR.glob("schedules_*.parquet")]
    assert sched_files, "Missing schedules. Run ETL first."
    schedules = pd.read_parquet(sched_files[0]) if len(sched_files)==1 else pd.concat([pd.read_parquet(p) for p in sched_files], ignore_index=True)

    # Build a per-game table with pre-game ratings for home/away
    # Merge ratings for each team in the game_id row
    base = ratings[["season","week","game_id","team","off_epa_pp_roll","def_epa_pp_roll","net_epa_rating"]].copy()
    # Home
    home = schedules[["game_id","season","week","home_team","away_team","home_score","away_score"]].copy()
    home = home.merge(base.rename(columns={"team":"home_team"}),
                      on=["game_id","season","week","home_team"], how="left")
    # Away
    full = home.merge(base.rename(columns={"team":"away_team",
                                           "off_epa_pp_roll":"away_off_roll",
                                           "def_epa_pp_roll":"away_def_roll",
                                           "net_epa_rating":"away_net"}),
                      on=["game_id","season","week","away_team"], how="left")

    # Features: differences (home - away)
    full["net_diff"] = full["net_epa_rating"] - full["away_net"]
    full["off_diff"] = full["off_epa_pp_roll"] - full["away_off_roll"]
    full["def_diff"] = full["def_epa_pp_roll"] - full["away_def_roll"]

    # Label: home win
    full["home_win"] = (full["home_score"] > full["away_score"]).astype(int)

    # Drop rows with missing features (early season with insufficient history)
    X = full[["net_diff","off_diff","def_diff"]].dropna()
    y = full.loc[X.index, "home_win"]

    if len(X) < 50:
        raise RuntimeError("Not enough training rows after dropping NaNs. Use a wider season range.")

    # TimeSeries CV + calibration
    tscv = TimeSeriesSplit(n_splits=5)
    model = LogisticRegression(max_iter=1000)
    calib = CalibratedClassifierCV(model, method="isotonic", cv=tscv)
    calib.fit(X, y)

    # Evaluate in-sample (rough)
    p = calib.predict_proba(X)[:,1]
    metrics = {
        "brier": float(brier_score_loss(y, p)),
        "logloss": float(log_loss(y, p))
    }
    # Save
    joblib.dump(calib, ART_DIR / "game_win_clf.joblib")
    pd.Series(metrics).to_json(ART_DIR / "game_win_metrics.json")
    return metrics
