from __future__ import annotations
import os
from pathlib import Path
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import brier_score_loss, log_loss
import joblib

PROC_DIR = Path(os.getenv("PROC_DIR", "data/processed"))
ART_DIR = Path(os.getenv("ART_DIR", "data/artifacts"))

def train_and_save_extended():
    df = pd.read_parquet(PROC_DIR / "game_model_table.parquet").copy()

    # Select features (use what we have; betting columns may be NaN)
    feature_cols = ["net_diff","off_diff","def_diff","rest_diff","travel_diff_km","dome_any"]
    if "closing_spread" in df.columns:
        feature_cols.append("closing_spread")
    if "closing_total" in df.columns:
        feature_cols.append("closing_total")

    df = df.dropna(subset=["home_win"])
    X = df[feature_cols].fillna(0.0)
    y = df["home_win"].astype(int)

    if len(X) < 50:
        raise RuntimeError("Not enough training data")

    tscv = TimeSeriesSplit(n_splits=5)
    base = LogisticRegression(max_iter=2000)
    model = CalibratedClassifierCV(base, method="isotonic", cv=tscv)
    model.fit(X, y)

    p = model.predict_proba(X)[:,1]
    metrics = {"brier": float(brier_score_loss(y, p)), "logloss": float(log_loss(y, p))}

    ART_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model":model,"features":feature_cols}, ART_DIR / "game_win_extended.joblib")
    pd.Series(metrics).to_json(ART_DIR / "game_win_extended_metrics.json")
    return metrics
