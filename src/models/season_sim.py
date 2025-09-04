from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd

RAW_DIR = Path(os.getenv("RAW_DIR", "data/raw"))
PROC_DIR = Path(os.getenv("PROC_DIR", "data/processed"))
ART_DIR = Path(os.getenv("ART_DIR", "data/artifacts"))

# Minimal alias map to normalize legacy/short codes
ALIAS = {
    # relocations / old codes
    "OAK": "LV", "SD": "LAC", "STL": "LAR", "WSH": "WAS",
    # short "LA" (some sources use this; map to Rams by default for NFC schedules)
    "LA": "LAR",
    # rare alternates
    "JAX": "JAX",  # keep as-is if appears
}

def norm_team(t: str) -> str:
    if not isinstance(t, str): return t
    t = t.strip().upper()
    return ALIAS.get(t, t)

def _load_game_model_table() -> pd.DataFrame:
    p = PROC_DIR / "game_model_table.parquet"
    if not p.exists():
        raise FileNotFoundError("game_model_table.parquet not found; run enrich step first.")
    df = pd.read_parquet(p)
    # Normalize team codes
    for col in ["home_team", "away_team"]:
        if col in df.columns:
            df[col] = df[col].astype(str).map(norm_team)
    return df

def _load_team_meta() -> pd.DataFrame:
    # Optional: enrich with conf/div from data/static if available
    static = Path("data/static/team_meta.csv")
    if static.exists():
        tm = pd.read_csv(static)
        # Expect columns like team, conf, div (robust to casing)
        cols = {c.lower(): c for c in tm.columns}
        team = cols.get("team") or cols.get("abbr") or list(cols.values())[0]
        conf = cols.get("conf") or cols.get("conference")
        div  = cols.get("div")  or cols.get("division")
        tm = tm.rename(columns={team: "team", conf: "conf", div: "div"})
        tm["team"] = tm["team"].astype(str).map(norm_team)
        return tm[["team","conf","div"]].drop_duplicates()
    else:
        # Fallback minimal map (covers current 32 teams)
        data = []
        # AFC
        data += [(t,"AFC",d) for t,d in [
            ("BUF","East"),("MIA","East"),("NE","East"),("NYJ","East"),
            ("CIN","North"),("CLE","North"),("BAL","North"),("PIT","North"),
            ("HOU","South"),("IND","South"),("JAX","South"),("TEN","South"),
            ("KC","West"),("LAC","West"),("DEN","West"),("LV","West"),
        ]]
        # NFC
        data += [(t,"NFC",d) for t,d in [
            ("DAL","East"),("PHI","East"),("NYG","East"),("WAS","East"),
            ("DET","North"),("GB","North"),("MIN","North"),("CHI","North"),
            ("TB","South"),("NO","South"),("ATL","South"),("CAR","South"),
            ("SF","West"),("SEA","West"),("LAR","West"),("ARI","West"),
        ]]
        return pd.DataFrame(data, columns=["team","conf","div"])

def simulate_season(season: int, sims: int = 5000, use_extended: bool = True) -> str:
    """
    Monte Carlo over a season using our calibrated game win probs.
    Writes season summary CSV (avg wins, playoff-ish odds proxy).
    """
    gmt = _load_game_model_table()
    gmt = gmt[gmt["season"] == season].copy()

    # Pick win prob column
    pcol = "home_win_prob"
    if pcol not in gmt.columns:
        raise KeyError("home_win_prob not found in game_model_table; train model first.")

    teams = sorted(set(gmt["home_team"]).union(set(gmt["away_team"])))
    meta = _load_team_meta()
    conf_map = dict(zip(meta["team"], meta["conf"]))
    div_map  = dict(zip(meta["team"], meta["div"]))

    # Normalize and verify all teams are mappable
    teams = [norm_team(t) for t in teams]
    for t in teams:
        if t not in conf_map:
            raise KeyError(f"Unknown team code after normalization: {t} (add alias or team_meta row)")

    # Build index maps
    idx = {t:i for i,t in enumerate(teams)}
    confs = np.array([conf_map[t] for t in teams], dtype=object)

    # Prepare schedule arrays
    games = gmt[["home_team","away_team",pcol]].copy()
    games["home_team"] = games["home_team"].map(norm_team)
    games["away_team"] = games["away_team"].map(norm_team)

    h = games["home_team"].map(idx).to_numpy()
    a = games["away_team"].map(idx).to_numpy()
    p_home = games[pcol].to_numpy().clip(0.001, 0.999)

    rng = np.random.default_rng(42)
    nteams = len(teams)
    wins = np.zeros((sims, nteams), dtype=np.int16)

    for s in range(sims):
        r = rng.random(len(games))
        home_wins = (r < p_home).astype(np.int8)
        away_wins = 1 - home_wins
        w = np.zeros(nteams, dtype=np.int16)
        # accumulate
        np.add.at(w, h, home_wins)
        np.add.at(w, a, away_wins)
        wins[s] = w

    avg_wins = wins.mean(axis=0)
    # crude playoff odds proxy: top 7 by conf in each sim
    playoff = np.zeros((sims, nteams), dtype=np.int8)
    for s in range(sims):
        w = wins[s]
        for conf in ["AFC","NFC"]:
            mask = (confs == conf)
            idxs = np.where(mask)[0]
            # top 7 win counts in that conf
            top7 = idxs[np.argsort(w[idxs])[-7:]]
            playoff[s, top7] = 1

    playoff_odds = playoff.mean(axis=0)

    out = pd.DataFrame({
        "team": teams,
        "avg_wins": avg_wins,
        "playoff_odds": playoff_odds,
    }).sort_values("avg_wins", ascending=False)

    ART_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ART_DIR / f"season_{season}_sim_summary.csv"
    out.to_csv(out_path, index=False)
    return str(out_path)
