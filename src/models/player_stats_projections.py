from __future__ import annotations
import os
from pathlib import Path
import pandas as pd
import numpy as np

RAW_DIR = Path(os.getenv("RAW_DIR", "data/raw"))
ART_DIR = Path(os.getenv("ART_DIR", "data/artifacts"))
PROC_DIR = Path(os.getenv("PROC_DIR", "data/processed"))

def _col(df: pd.DataFrame, candidates: list[str], required: bool = True, default: str | None = None) -> str | None:
    """Return the first existing column (case-insensitive)."""
    cmap = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in cmap:
            return cmap[c.lower()]
    if required and default is None:
        raise KeyError(f"Missing any of columns: {candidates}")
    return default

def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    out = a.astype(float)
    out = out / b.replace({0: np.nan})
    return out.fillna(0.0)

def build_player_stat_projections() -> str:
    # Load weekly data written by ETL
    weekly_parqs = sorted(RAW_DIR.glob("weekly_*.parquet"))
    if not weekly_parqs:
        raise FileNotFoundError("weekly parquet not found in data/raw; run ETL first.")
    wk = pd.read_parquet(weekly_parqs[-1])

    # Map varying column names across nfl_data_py versions
    season = _col(wk, ["season"])
    week   = _col(wk, ["week", "game_week"])
    pid    = _col(wk, ["player_id", "gsis_id", "pfr_id"])
    pname  = _col(wk, ["player_name", "player", "name"], required=False)
    team   = _col(wk, ["recent_team", "team", "posteam"])
    targets = _col(wk, ["targets", "target"], required=False, default=None)
    recs    = _col(wk, ["receptions", "rec"], required=False, default=None)
    rec_yds = _col(wk, ["receiving_yards", "rec_yards", "yards_receiving"], required=False, default=None)
    rec_td  = _col(wk, ["receiving_tds", "rec_tds", "td_receiving"], required=False, default=None)
    rush_att= _col(wk, ["rushing_attempts", "rush_att", "carries", "rushing_att"], required=False, default=None)
    rush_yds= _col(wk, ["rushing_yards", "rush_yards", "yards_rushing"], required=False, default=None)
    rush_td = _col(wk, ["rushing_tds", "rush_tds", "td_rushing"], required=False, default=None)

    # Build a standardized frame
    df = pd.DataFrame({
        "season": wk[season].astype(int),
        "week":   wk[week].astype(int),
        "player_id": wk[pid],
        "team":   wk[team],
    })
    if pname:
        df["player_name"] = wk[pname]
    else:
        df["player_name"] = "Unknown"

    # Fill metrics (missing -> 0)
    for out_col, src in [
        ("targets", targets), ("receptions", recs), ("rec_yards", rec_yds), ("rec_td", rec_td),
        ("rush_att", rush_att), ("rush_yards", rush_yds), ("rush_td", rush_td),
    ]:
        df[out_col] = wk[src].fillna(0).astype(float) if (src and src in wk.columns) else 0.0

    # Team totals per game (for shares/normalization)
    team_tot = (df.groupby(["season","week","team"], as_index=False)
                  .agg(team_targets=("targets","sum"),
                       team_carries=("rush_att","sum")))
    df = df.merge(team_tot, on=["season","week","team"], how="left")
    df["team_targets"] = df["team_targets"].fillna(0.0)
    df["team_carries"] = df["team_carries"].fillna(0.0)

    # Per-game usage shares
    df["target_share"] = _safe_div(df["targets"], df["team_targets"])
    df["carry_share"]  = _safe_div(df["rush_att"], df["team_carries"])

    # Rolling form (last-3) with a prior from player-season mean
    df = df.sort_values(["player_id","season","week"])
    def _proj(g: pd.DataFrame) -> pd.DataFrame:
        ts_mean = g["target_share"].expanding().mean()
        cs_mean = g["carry_share"].expanding().mean()
        ts = g["target_share"].rolling(3, min_periods=1).mean().shift(1).fillna(ts_mean)
        cs = g["carry_share"].rolling(3, min_periods=1).mean().shift(1).fillna(cs_mean)
        # Convert shares to counting stats using latest team totals observed
        team_tgt_next = g["team_targets"].rolling(3, min_periods=1).mean().shift(1).fillna(g["team_targets"].expanding().mean())
        team_car_next = g["team_carries"].rolling(3, min_periods=1).mean().shift(1).fillna(g["team_carries"].expanding().mean())
        proj_targets = (ts * team_tgt_next).clip(lower=0)
        proj_carries = (cs * team_car_next).clip(lower=0)
        # Yardage/TD simple rates
        ypt = (g["rec_yards"] / g["targets"].replace({0:np.nan})).fillna(7.5)   # yards per target prior
        ypc = (g["rush_yards"]/ g["rush_att"].replace({0:np.nan})).fillna(4.2)  # yards per carry prior
        tpr = (g["rec_td"]   / g["targets"].replace({0:np.nan})).fillna(0.04)   # TD per target prior
        ctd = (g["rush_td"]  / g["rush_att"].replace({0:np.nan})).fillna(0.03)  # TD per carry prior
        # EWMA smoothing
        ypt_s = ypt.ewm(alpha=0.5, adjust=False).mean().shift(1).fillna(ypt.mean())
        ypc_s = ypc.ewm(alpha=0.5, adjust=False).mean().shift(1).fillna(ypc.mean())
        tpr_s = tpr.ewm(alpha=0.5, adjust=False).mean().shift(1).fillna(tpr.mean())
        ctd_s = ctd.ewm(alpha=0.5, adjust=False).mean().shift(1).fillna(ctd.mean())

        out = pd.DataFrame({
            "proj_targets": proj_targets,
            "proj_carries": proj_carries,
            "proj_rec_yards": proj_targets * ypt_s,
            "proj_rush_yards": proj_carries * ypc_s,
            "proj_rec_td": proj_targets * tpr_s,
            "proj_rush_td": proj_carries * ctd_s,
        })
        return out

    proj = df.groupby(["player_id","season"], group_keys=False).apply(_proj).reset_index(drop=True)
    out = pd.concat([df.reset_index(drop=True), proj], axis=1)

    # Latest row per player-season is our current projection snapshot
    latest = out.sort_values(["player_id","season","week"]).groupby(["player_id","season"]).tail(1)

    ART_DIR.mkdir(parents=True, exist_ok=True)
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    latest_cols = [
        "player_id","player_name","team","season",
        "proj_targets","proj_rec_yards","proj_rec_td",
        "proj_carries","proj_rush_yards","proj_rush_td",
        "target_share","carry_share"
    ]
    latest[latest_cols].to_csv(ART_DIR / "player_stat_projections.csv", index=False)

    # Keep full per-game frame too (optional for analysis)
    out.to_parquet(PROC_DIR / "player_stat_projections_pergame.parquet", index=False)

    return str(ART_DIR / "player_stat_projections.csv")
