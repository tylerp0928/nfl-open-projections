from __future__ import annotations
import os
from pathlib import Path
import pandas as pd
import numpy as np

RAW_DIR = Path(os.getenv("RAW_DIR", "data/raw"))
PROC_DIR = Path(os.getenv("PROC_DIR", "data/processed"))
ART_DIR = Path(os.getenv("ART_DIR", "data/artifacts"))

def _col(df: pd.DataFrame, candidates: list[str], default: str | None = None) -> str | None:
    """Return the first column name that exists (case-insensitive)."""
    cols = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in cols:
            return cols[c.lower()]
    return default

def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    out = a.astype(float)
    out = out / b.replace({0: np.nan})
    return out.fillna(0.0)

def build_player_usage() -> str:
    # ---- load weekly parquet (created by fetch_nflverse) ----
    weekly_parqs = sorted(RAW_DIR.glob("weekly_*.parquet"))
    if not weekly_parqs:
        raise FileNotFoundError("weekly parquet not found in data/raw; run ETL first.")
    wk = pd.read_parquet(weekly_parqs[-1])  # most recent combined file

    # ---- normalize column names & pick keys ----
    # handle different nfl_data_py versions
    pid = _col(wk, ["player_id", "gsis_id", "pfr_id"])
    pname = _col(wk, ["player_name", "player", "name"])
    team = _col(wk, ["recent_team", "team", "posteam"])
    season = _col(wk, ["season"])
    week = _col(wk, ["week", "game_week"])

    targets = _col(wk, ["targets", "target"])
    rush_att = _col(wk, ["rushing_attempts", "rush_att", "carries", "rushing_att"])
    rec_yds = _col(wk, ["receiving_yards", "rec_yards", "yards_receiving"])
    rush_yds = _col(wk, ["rushing_yards", "rush_yards", "yards_rushing"])

    # sanity
    required = [pid, team, season, week]
    if any(c is None for c in required):
        missing = [n for n,c in zip(["player_id","team","season","week"], required) if c is None]
        raise KeyError(f"Required columns missing from weekly: {missing}")

    for c in [targets, rush_att, rec_yds, rush_yds]:
        if c is None:
            # create zeros if a metric is absent in this schema
            newname = {targets: "targets", rush_att: "rush_att", rec_yds: "rec_yards", rush_yds: "rush_yards"}[c]
            wk[newname] = 0
        else:
            pass

    # create standardized columns
    wk_std = wk[[pid, pname] if pname else [pid]].copy()
    wk_std = wk_std.rename(columns={pid: "player_id"})
    if pname: wk_std = wk_std.rename(columns={pname: "player_name"})
    wk_std["team"]   = wk[team].values
    wk_std["season"] = wk[season].astype(int).values
    wk_std["week"]   = wk[week].astype(int).values
    wk_std["targets"]   = wk[targets].fillna(0).astype(float) if targets in wk.columns else 0.0
    wk_std["rush_att"]  = wk[rush_att].fillna(0).astype(float) if rush_att in wk.columns else 0.0
    wk_std["rec_yards"] = wk[rec_yds].fillna(0).astype(float) if rec_yds in wk.columns else 0.0
    wk_std["rush_yards"]= wk[rush_yds].fillna(0).astype(float) if rush_yds in wk.columns else 0.0

    # ---- compute team totals per game ----
    team_tot = (
        wk_std.groupby(["season","week","team"], as_index=False)
              .agg(team_targets=("targets","sum"),
                   team_carries=("rush_att","sum"))
    )

    usage = wk_std.merge(team_tot, on=["season","week","team"], how="left")
    usage["team_targets"] = usage["team_targets"].fillna(0.0)
    usage["team_carries"] = usage["team_carries"].fillna(0.0)

    # shares per game
    usage["target_share"] = _safe_div(usage["targets"], usage["team_targets"])
    usage["carry_share"]  = _safe_div(usage["rush_att"], usage["team_carries"])

    # ---- simple projection: last-3 avg per player-season ----
    usage = usage.sort_values(["player_id","season","week"])
    def _proj(g: pd.DataFrame) -> pd.Series:
        # mean of last 3 games as "next" projection; shift so it predicts next week
        ts = g["target_share"].rolling(3, min_periods=1).mean().shift(1).fillna(g["target_share"].expanding().mean())
        cs = g["carry_share"].rolling(3, min_periods=1).mean().shift(1).fillna(g["carry_share"].expanding().mean())
        return pd.DataFrame({"proj_target_share_next": ts, "proj_carry_share_next": cs})

    proj = usage.groupby(["player_id","season"], group_keys=False).apply(_proj)
    usage = pd.concat([usage.reset_index(drop=True), proj.reset_index(drop=True)], axis=1)

    # latest row per (player, season) as our current projection snapshot
    latest = usage.sort_values(["player_id","season","week"]).groupby(["player_id","season"]).tail(1)

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    ART_DIR.mkdir(parents=True, exist_ok=True)

    # keep a processed parquet (all games)
    out_parq = PROC_DIR / "player_usage.parquet"
    usage.to_parquet(out_parq, index=False)

    # and a compact CSV for downstream modules / app
    cols = ["player_id","player_name","team","season","proj_target_share_next","proj_carry_share_next",
            "targets","rush_att","team_targets","team_carries"]
    latest[cols].to_csv(ART_DIR / "player_usage_projections.csv", index=False)

    return str(out_parq)
