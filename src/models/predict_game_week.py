from __future__ import annotations
import os
from pathlib import Path
import argparse
import pandas as pd
import joblib

PROC_DIR = Path(os.getenv("PROC_DIR", "data/processed"))
RAW_DIR = Path(os.getenv("RAW_DIR", "data/raw"))
ART_DIR = Path(os.getenv("ART_DIR", "data/artifacts"))

def predict_week(season:int, week:int) -> str:
    model = joblib.load(ART_DIR / "game_win_clf.joblib")
    ratings = pd.read_parquet(PROC_DIR / "team_ratings.parquet")
    sched_files = [p for p in RAW_DIR.glob("schedules_*.parquet")]
    schedules = pd.read_parquet(sched_files[0]) if len(sched_files)==1 else pd.concat([pd.read_parquet(p) for p in sched_files], ignore_index=True)

    slate = schedules.query("season == @season and week == @week").copy()

    base = ratings[["season","week","game_id","team","off_epa_pp_roll","def_epa_pp_roll","net_epa_rating"]].copy()

    slate = slate.merge(base.rename(columns={"team":"home_team"}),
                        on=["game_id","season","week","home_team"], how="left")
    slate = slate.merge(base.rename(columns={"team":"away_team",
                                             "off_epa_pp_roll":"away_off_roll",
                                             "def_epa_pp_roll":"away_def_roll",
                                             "net_epa_rating":"away_net"}),
                        on=["game_id","season","week","away_team"], how="left")

    slate["net_diff"] = slate["net_epa_rating"] - slate["away_net"]
    slate["off_diff"] = slate["off_epa_pp_roll"] - slate["away_off_roll"]
    slate["def_diff"] = slate["def_epa_pp_roll"] - slate["away_def_roll"]

    X = slate[["net_diff","off_diff","def_diff"]]
    proba = model.predict_proba(X)[:,1]
    out = slate[["game_id","season","week","home_team","away_team"]].copy()
    out["home_win_prob"] = proba

    ART_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ART_DIR / f"predictions_{season}_wk{week}.csv"
    out.to_csv(out_path, index=False)
    return str(out_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    args = ap.parse_args()
    p = predict_week(args.season, args.week)
    print(f"Wrote {p}")

if __name__ == "__main__":
    main()
