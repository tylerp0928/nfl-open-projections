import os
import io
import json
from pathlib import Path
from typing import Optional
import requests
import pandas as pd
import streamlit as st
import plotly.express as px

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
ART_DIR = DATA_DIR / "artifacts"

st.set_page_config(page_title="NFL Open Projections", layout="wide")

@st.cache_data(show_spinner=False)
def load_parquet(path: Path) -> Optional[pd.DataFrame]:
    try:
        return pd.read_parquet(path)
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path)
    except Exception:
        return None

def gh_dispatch_workflow(repo: str, token: str, workflow: str = "ci.yml", seasons: str = "2019-2025"):
    url = f"https://api.github.com/repos/{repo}/actions/workflows/{workflow}/dispatches"
    headers = {"Authorization": f"Bearer {token}","Accept":"application/vnd.github+json"}
    payload = {"ref": "main", "inputs": {"seasons": seasons}}
    r = requests.post(url, headers=headers, json=payload, timeout=20)
    return r.status_code, r.text

def section_games():
    st.header("Game Probabilities & Team Ratings")
    gmt = load_parquet(PROC_DIR / "game_model_table.parquet")
    if gmt is None or gmt.empty:
        st.info("No game_model_table found yet. Run the pipeline first.")
        return
    seasons = sorted(gmt["season"].dropna().unique().tolist())
    colA, colB, colC = st.columns([1,1,2])
    with colA:
        season = st.selectbox("Season", seasons, index=len(seasons)-1)
    weeks = sorted(gmt.loc[gmt["season"]==season, "week"].dropna().unique().tolist())
    with colB:
        week = st.selectbox("Week", weeks, index=0)

    view = gmt.query("season == @season and week == @week").copy()
    # If extended model predictions were saved to artifacts, prefer them; otherwise recompute ad-hoc via simple diffs
    preds_path = ART_DIR / f"predictions_{season}_wk{week}.csv"
    preds = load_csv(preds_path)
    if preds is not None and not preds.empty:
        st.success("Loaded saved predictions")
        show = preds.copy()
    else:
        st.warning("Saved predictions not found. Using a proxy from net/off/def diffs (not calibrated).")
        # crude score proxy
        show = view[["game_id","home_team","away_team","net_diff","off_diff","def_diff"]].copy()
        # not probabilistic; display diffs only
        show["home_win_prob"] = 0.5 + (show["net_diff"].fillna(0)/10).clip(-0.49,0.49)

    # Plot
    fig = px.bar(show.sort_values("home_win_prob"), x="home_win_prob", y=show.apply(lambda r: f"{r['home_team']} vs {r['away_team']}", axis=1),
                 orientation="h", labels={"home_win_prob":"Home win prob","y":""})
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(show, use_container_width=True)

def section_players():
    st.header("Player Projections")
    # Prefer injury-adjusted usage-based projections
    p_inj = load_csv(ART_DIR / "player_usage_projections_injury_adj.csv")
    p_raw = load_csv(ART_DIR / "player_stat_projections.csv")
    df = p_inj if (p_inj is not None and not p_inj.empty) else p_raw
    if df is None or df.empty:
        st.info("No player projections found yet. Run the pipeline first.")
        return
    teams = ["All"] + sorted(df["team"].dropna().unique().tolist())
    team = st.selectbox("Team", teams, index=0)
    if team != "All":
        df = df[df["team"] == team]
    # Basic fantasy-like score
    if "proj_total_points" not in df.columns:
        df["proj_receiving_points"] = df.get("proj_rec_yards",0)/10 + df.get("proj_rec_td",0)*6
        df["proj_rushing_points"] = df.get("proj_rush_yards",0)/10 + df.get("proj_rush_td",0)*6
        df["proj_total_points"] = df["proj_receiving_points"].fillna(0)+df["proj_rushing_points"].fillna(0)
    topn = st.slider("Show top N", 5, 50, 20)
    top = df.sort_values("proj_total_points", ascending=False).head(topn)
    fig = px.bar(top.sort_values("proj_total_points"), x="proj_total_points", y="player_name", orientation="h",
                 labels={"proj_total_points":"Projected total points","player_name":""})
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(top, use_container_width=True)

def section_teams():
    st.header("Team Ratings (Rolling EPA)")
    tr = load_parquet(PROC_DIR / "team_ratings.parquet")
    if tr is None or tr.empty:
        st.info("No team_ratings parquet yet. Run the pipeline first.")
        return
    teams = sorted(tr["team"].dropna().unique().tolist())
    team = st.selectbox("Team", teams, index=teams.index("KC") if "KC" in teams else 0)
    tt = tr[tr["team"]==team].sort_values(["season","week"])
    fig = px.line(tt, x=tt["season"].astype(str)+"-W"+tt["week"].astype(str), y=["off_epa_pp_roll","def_epa_pp_roll","net_epa_rating"],
                  labels={"value":"EPA per play (rolling)","x":"Season-Week","variable":"metric"})
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(tt.tail(20), use_container_width=True)

def section_sims():
    st.header("Season Simulations")
    sim_files = sorted([p for p in ART_DIR.glob("season_*_sim_summary.csv")])
    if not sim_files:
        st.info("No season simulation summaries found yet.")
        return
    labels = [p.name for p in sim_files]
    choice = st.selectbox("Select summary", labels, index=len(labels)-1)
    df = load_csv(ART_DIR / choice)
    if df is None: 
        st.warning("Could not load simulation file.")
        return
    fig = px.bar(df.sort_values("avg_wins", ascending=False), x="team", y="avg_wins", labels={"avg_wins":"Average Wins"})
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df.sort_values(["playoff_odds"], ascending=False), use_container_width=True)

def section_admin():
    st.header("Admin & Automation")
    st.write("Trigger GitHub Actions CI to refresh data/models (optional).")
    repo = st.text_input("GitHub repo (owner/name)", value=st.secrets.get("GH_REPO",""))
    token = st.text_input("GitHub token (PAT) with repo scope", type="password", value=st.secrets.get("GH_TOKEN",""))
    seasons = st.text_input("Seasons (e.g., 2019-2025)", value="2019-2025")
    if st.button("Dispatch workflow"):
        if not repo or not token:
            st.error("Repo and token are required.")
        else:
            code, text = gh_dispatch_workflow(repo, token, seasons=seasons)
            if code in (201,204):
                st.success("Workflow dispatched.")
            else:
                st.error(f"Dispatch failed: {code} {text}")
    st.divider()
    st.write("Upload artifacts to preview (optional):")
    uploaded = st.file_uploader("Upload CSV/Parquet to /data/artifacts (will not persist on Streamlit Cloud).", accept_multiple_files=True)
    if uploaded:
        for f in uploaded:
            buf = f.read()
            out = ART_DIR / f.name
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "wb") as w:
                w.write(buf)
        st.success("Uploaded to data/artifacts for this session.")

st.sidebar.title("NFL Open Projections")
tab = st.sidebar.radio("Sections", ["Games","Players","Teams","Season Sims","Admin"])

if tab == "Games":
    section_games()
elif tab == "Players":
    section_players()
elif tab == "Teams":
    section_teams()
elif tab == "Season Sims":
    section_sims()
else:
    section_admin()
