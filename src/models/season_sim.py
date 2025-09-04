from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

RAW_DIR = Path(os.getenv("RAW_DIR", "data/raw"))
PROC_DIR = Path(os.getenv("PROC_DIR", "data/processed"))
ART_DIR = Path(os.getenv("ART_DIR", "data/artifacts"))

def simulate_season(season:int, sims:int=5000, use_extended:bool=True, seed:int=42) -> str:
    rng = np.random.default_rng(seed)

    # Load model
    bundle = joblib.load(ART_DIR / ("game_win_extended.joblib" if use_extended else "game_win_clf.joblib"))
    model = bundle["model"] if isinstance(bundle, dict) else bundle
    feat_cols = bundle["features"] if isinstance(bundle, dict) and "features" in bundle else ["net_diff","off_diff","def_diff"]

    # Build features for target season
    df = pd.read_parquet(PROC_DIR / "game_model_table.parquet")
    slate = df[df["season"] == season].copy()

    X = slate[feat_cols].fillna(0.0)
    home_prob = model.predict_proba(X)[:,1]
    slate = slate[["game_id","season","week","home_team","away_team"]].copy()
    slate["home_win_prob"] = home_prob

    # Team meta for conferences
    team_meta = pd.read_csv("data/static/team_meta.csv")

    teams = sorted(set(slate["home_team"]).union(set(slate["away_team"])))
    team_index = {t:i for i,t in enumerate(teams)}

    # Pre-build arrays for quick sim
    h_idx = slate["home_team"].map(team_index).to_numpy()
    a_idx = slate["away_team"].map(team_index).to_numpy()
    p = slate["home_win_prob"].to_numpy()

    wins_matrix = np.zeros((sims, len(teams)), dtype=np.int16)

    for s in range(sims):
        # Draw each game result
        h_wins = rng.binomial(1, p)
        a_wins = 1 - h_wins
        # Tally
        np.add.at(wins_matrix[s], h_idx, h_wins)
        np.add.at(wins_matrix[s], a_idx, a_wins)

    # Summaries
    avg_wins = wins_matrix.mean(axis=0)
    df_wins = pd.DataFrame({"team":teams, "avg_wins":avg_wins})
    # Playoff odds (naive): top 7 per conference by wins in each sim
    conf_map = dict(zip(team_meta["team"], team_meta["conference"]))
    confs = np.array([conf_map[t] for t in teams])
    playoff_counts = dict((t,0) for t in teams)

    for s in range(sims):
        w = wins_matrix[s]
        for conf in ["AFC","NFC"]:
            mask = (confs == conf)
            idxs = np.where(mask)[0]
            top7 = idxs[np.argsort(w[idxs])[::-1][:7]]
            for i in top7:
                playoff_counts[teams[i]] += 1

    playoff_odds = pd.DataFrame({"team":teams, "playoff_odds": [playoff_counts[t]/sims for t in teams]})
    out = df_wins.merge(playoff_odds, on="team", how="left").sort_values("avg_wins", ascending=False)

    ART_DIR.mkdir(parents=True, exist_ok=True)
    path = ART_DIR / f"season_{season}_sim_summary.csv"
    out.to_csv(path, index=False)
    return str(path)
