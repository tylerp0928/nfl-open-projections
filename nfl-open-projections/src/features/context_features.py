from __future__ import annotations
import os
from pathlib import Path
import pandas as pd
import numpy as np

RAW_DIR = Path(os.getenv("RAW_DIR", "data/raw"))
PROC_DIR = Path(os.getenv("PROC_DIR", "data/processed"))
STATIC_DIR = Path("data/static")

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1 = np.radians(lat1); p2 = np.radians(lat2)
    dlat = np.radians(lat2-lat1); dlon = np.radians(lon2-lon1)
    a = np.sin(dlat/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dlon/2)**2
    c = 2*np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R*c

def build_context_features() -> str:
    sched_files = [p for p in RAW_DIR.glob("schedules_*.parquet")]
    pbp_files = [p for p in RAW_DIR.glob("pbp_*.parquet")]
    assert sched_files, "Schedules parquet missing"
    schedules = (pd.read_parquet(sched_files[0]) if len(sched_files)==1 
                 else pd.concat([pd.read_parquet(p) for p in sched_files], ignore_index=True))

    # Stadium lookup
    stad = pd.read_csv(STATIC_DIR / "stadiums.csv")
    stad = stad[["team","lat","lon","roof"]]

    # Rest days per team: derive from schedules
    # Build long form: one row per team per game with date
    gcols = ["game_id","season","week","gameday","home_team","away_team"]
    games = schedules[gcols].copy()
    # Convert dates
    if "gameday" in games:
        games["gameday"] = pd.to_datetime(games["gameday"], errors="coerce")
    else:
        # fallback
        games["gameday"] = pd.to_datetime(games["game_id"].str[0:8], errors="coerce")

    home = games.rename(columns={"home_team":"team"})[["game_id","season","week","gameday","team"]].copy()
    away = games.rename(columns={"away_team":"team"})[["game_id","season","week","gameday","team"]].copy()
    long = pd.concat([home, away], ignore_index=True)

    long = long.sort_values(["team","gameday"])
    long["prev_game_date"] = long.groupby("team")["gameday"].shift(1)
    long["rest_days"] = (long["gameday"] - long["prev_game_date"]).dt.days
    long["rest_days"] = long["rest_days"].fillna(10)  # assume ~10 for Week 1

    # Travel distance: use stadium coords (home stadium as origin, opponent stadium as destination when away)
    # Merge coords for each team
    long = long.merge(stad, on="team", how="left")
    long = long.rename(columns={"lat":"team_lat","lon":"team_lon","roof":"team_roof"})

    # Map opponent for each row
    opp = games.copy()
    opp["home_opp"] = opp["away_team"]
    opp["away_opp"] = opp["home_team"]
    # Merge opponent depending on whether team is home or away
    long = long.merge(games[["game_id","home_team","away_team"]], on="game_id", how="left")
    long["is_home"] = (long["team"] == long["home_team"]).astype(int)
    long["opponent"] = np.where(long["is_home"]==1, long["away_team"], long["home_team"])

    # Opponent stadium coords
    opp_coords = stad.rename(columns={"team":"opponent","lat":"opp_lat","lon":"opp_lon","roof":"opp_roof"})
    long = long.merge(opp_coords, on="opponent", how="left")

    # Distance: if home game, distance=0; else stadium-to-stadium great-circle
    long["travel_km"] = np.where(long["is_home"]==1, 0.0,
                                 haversine(long["team_lat"], long["team_lon"], long["opp_lat"], long["opp_lon"]))

    # Dome/indoor indicator (game-level): prefer home/venue; fall back to opponent if missing
    long["roof_game"] = np.where(long["is_home"]==1, long["team_roof"], long["opp_roof"])
    long["is_dome_like"] = long["roof_game"].fillna("").str.contains("dome|retractable|semi|canopy", case=False).astype(int)

    # Save team-game context
    out = long[["game_id","season","week","team","is_home","rest_days","travel_km","is_dome_like"]].copy()

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    path = PROC_DIR / "context_features.parquet"
    out.to_parquet(path, index=False)
    return str(path)
