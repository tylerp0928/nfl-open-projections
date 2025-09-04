from __future__ import annotations
import os
from pathlib import Path
import pandas as pd
import numpy as np

RAW_DIR = Path(os.getenv("RAW_DIR", "data/raw"))
PROC_DIR = Path(os.getenv("PROC_DIR", "data/processed"))
ART_DIR = Path(os.getenv("ART_DIR", "data/artifacts"))

def _safe_div(a,b):
    return np.where(b>0, a/b, 0.0)

def build_player_stat_projections() -> str:
    usage = pd.read_parquet(PROC_DIR / "player_usage.parquet")
    weekly_files = [p for p in RAW_DIR.glob("weekly_*.parquet")]
    weekly = pd.read_parquet(weekly_files[0]) if len(weekly_files)==1 else pd.concat([pd.read_parquet(p) for p in weekly_files], ignore_index=True)

    # Efficiency estimates per player (receiving yds/target, rush yds/carry) from weekly stats
    rec = weekly[["season","week","player_id","player_name","team","targets","receptions","receiving_yards"]].copy()
    ru = weekly[["season","week","player_id","player_name","team","rushing_attempts","rushing_yards"]].copy().rename(
        columns={"rushing_attempts":"carries"})

    # Compute per-player career to date aggregates for shrinkage
    rec_agg = (rec.groupby(["player_id","player_name","team"], dropna=False)
               .agg(T=("targets","sum"), Y=("receiving_yards","sum")).reset_index())
    rec_agg["ypt"] = _safe_div(rec_agg["Y"], rec_agg["T"])

    ru_agg = (ru.groupby(["player_id","player_name","team"], dropna=False)
              .agg(C=("carries","sum"), RY=("rushing_yards","sum")).reset_index())
    ru_agg["ypc"] = _safe_div(ru_agg["RY"], ru_agg["C"])

    # League priors by position are not available here, so use global priors
    # Typical NFL averages ~ 7.5 ypt, ~ 4.3 ypc. We'll compute from data as safer priors.
    rec_prior = rec_agg.loc[rec_agg["T"]>0, "ypt"].mean() if (rec_agg["T"]>0).any() else 7.5
    ru_prior = ru_agg.loc[ru_agg["C"]>0, "ypc"].mean() if (ru_agg["C"]>0).any() else 4.3

    # Empirical-Bayes shrinkage: posterior = (n/(n+tau))*player + (tau/(n+tau))*prior
    tau_rec, tau_ru = 50.0, 80.0
    rec_agg["ypt_shrunk"] = (rec_agg["T"]/(rec_agg["T"]+tau_rec))*rec_agg["ypt"] + (tau_rec/(rec_agg["T"]+tau_rec))*rec_prior
    ru_agg["ypc_shrunk"] = (ru_agg["C"]/(ru_agg["C"]+tau_ru))*ru_agg["ypc"] + (tau_ru/(ru_agg["C"]+tau_ru))*ru_prior

    # Team volume baselines: average team pass attempts & rush attempts over recent 4 weeks
    team_week = weekly.groupby(["season","week","team"], dropna=False).agg(
        team_pass_att=("attempts","sum") if "attempts" in weekly.columns else ("passing_attempts","sum"),
        team_rush_att=("rushing_attempts","sum")
    ).reset_index()

    team_week = team_week.sort_values(["team","season","week"])
    team_week["pass_att_roll4"] = team_week.groupby("team")["team_pass_att"].transform(lambda s: s.shift(1).rolling(4, min_periods=1).mean())
    team_week["rush_att_roll4"] = team_week.groupby("team")["team_rush_att"].transform(lambda s: s.shift(1).rolling(4, min_periods=1).mean())

    # Most recent for each team-season
    team_latest = team_week.groupby(["team","season"]).tail(1)[["team","season","pass_att_roll4","rush_att_roll4"]]

    # Most recent usage for each player-season
    usage_latest = (usage.sort_values(["player_id","season","week"])
                        .groupby(["player_id","season"]).tail(1)[
                            ["player_id","player_name","posteam","season","carry_share_fcast","target_share_fcast"]
                        ].rename(columns={"posteam":"team"}))

    # Join efficiency priors
    usage_latest = usage_latest.merge(rec_agg[["player_id","ypt_shrunk"]], on="player_id", how="left")
    usage_latest = usage_latest.merge(ru_agg[["player_id","ypc_shrunk"]], on="player_id", how="left")

    # Join team volumes
    usage_latest = usage_latest.merge(team_latest, on=["team","season"], how="left")

    # Project next-game counting stats
    usage_latest["proj_targets"] = usage_latest["target_share_fcast"] * usage_latest["pass_att_roll4"].fillna(30)
    usage_latest["proj_rec_yards"] = usage_latest["proj_targets"] * usage_latest["ypt_shrunk"].fillna(rec_prior)

    usage_latest["proj_carries"] = usage_latest["carry_share_fcast"] * usage_latest["rush_att_roll4"].fillna(25)
    usage_latest["proj_rush_yards"] = usage_latest["proj_carries"] * usage_latest["ypc_shrunk"].fillna(ru_prior)

    # Crude TD models: proportional to usage with scaling
    usage_latest["proj_rec_td"] = 0.06 * usage_latest["proj_targets"]  # ~6% TD/target baseline
    usage_latest["proj_rush_td"] = 0.035 * usage_latest["proj_carries"] # ~3.5% TD/carry baseline

    ART_DIR.mkdir(parents=True, exist_ok=True)
    out = usage_latest[["player_id","player_name","team","season",
                        "proj_targets","proj_rec_yards","proj_rec_td",
                        "proj_carries","proj_rush_yards","proj_rush_td"]]
    path = ART_DIR / "player_stat_projections.csv"
    out.to_csv(path, index=False)
    return str(path)
