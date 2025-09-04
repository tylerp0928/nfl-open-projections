from __future__ import annotations
import os, io, base64
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

RAW_DIR = Path(os.getenv("RAW_DIR", "data/raw"))
PROC_DIR = Path(os.getenv("PROC_DIR", "data/processed"))
ART_DIR = Path(os.getenv("ART_DIR", "data/artifacts"))

def _img_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=140)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")

def _plot_team_win_probs(df: pd.DataFrame) -> str:
    # df: columns [home_team, away_team, home_win_prob]
    labels = [f"{h} vs {a}" for h,a in zip(df["home_team"], df["away_team"])]
    vals = df["home_win_prob"].values
    fig = plt.figure(figsize=(10,6))
    plt.barh(labels, vals)
    plt.xlabel("Home win probability")
    plt.title("This week's home win probabilities")
    return _img_to_base64(fig)

def _plot_top_players(df: pd.DataFrame, value_col: str, title: str) -> str:
    df = df.sort_values(value_col, ascending=False).head(10)
    labels = df["player_name"].fillna("Unknown").tolist()
    vals = df[value_col].values
    fig = plt.figure(figsize=(10,6))
    plt.barh(labels, vals)
    plt.xlabel(value_col.replace("_"," "))
    plt.title(title)
    return _img_to_base64(fig)

def build_weekly_slate_report(season:int, week:int) -> str:
    # Load predictions and projections
    preds_path = ART_DIR / f"predictions_{season}_wk{week}.csv"
    if not preds_path.exists():
        raise FileNotFoundError(f"Missing {preds_path}. Run predict_game_week first.")
    preds = pd.read_csv(preds_path)

    # Player projections (injury adjusted if present)
    inj_adj = ART_DIR / "player_usage_projections_injury_adj.csv"
    if inj_adj.exists():
        pproj = pd.read_csv(inj_adj)
    else:
        pproj = pd.read_csv(ART_DIR / "player_usage_projections.csv")

    # Simple derived fantasy-ish value proxies
    pproj["proj_receiving_points"] = pproj.get("proj_rec_yards",0)/10 + pproj.get("proj_rec_td",0)*6
    pproj["proj_rushing_points"] = pproj.get("proj_rush_yards",0)/10 + pproj.get("proj_rush_td",0)*6
    pproj["proj_total_points"] = pproj["proj_receiving_points"].fillna(0) + pproj["proj_rushing_points"].fillna(0)

    # Build plots
    game_img = _plot_team_win_probs(preds)
    top_total_img = _plot_top_players(pproj, "proj_total_points", "Top projected total points (injury-adjusted if available)")

    # Compose HTML
    html = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset='utf-8'/>
  <title>Slate Report — Week {week}, {season}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; }}
    h1, h2 {{ margin-bottom: 0.2rem; }}
    table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
    th {{ background: #f3f3f3; }}
    .imgwrap {{ margin: 1rem 0; }}
  </style>
</head>
<body>
  <h1>Weekly Slate Report</h1>
  <h2>Season {season}, Week {week}</h2>

  <h3>Game probabilities</h3>
  <div class='imgwrap'><img src='data:image/png;base64,{game_img}' alt='Home win probabilities'/></div>

  <table>
    <thead><tr><th>Home</th><th>Away</th><th>Home Win Prob</th></tr></thead>
    <tbody>
      {''.join(f"<tr><td>{r.home_team}</td><td>{r.away_team}</td><td>{r.home_win_prob:.3f}</td></tr>" for r in preds.itertuples())}
    </tbody>
  </table>

  <h3>Top projected players</h3>
  <div class='imgwrap'><img src='data:image/png;base64,{top_total_img}' alt='Top projected total points'/></div>

  <table>
    <thead><tr><th>Player</th><th>Team</th><th>Proj Targets</th><th>Proj Rec Yds</th><th>Proj Rec TD</th><th>Proj Carries</th><th>Proj Rush Yds</th><th>Proj Rush TD</th><th>Total Pts</th></tr></thead>
    <tbody>
      {''.join(
        f"<tr><td>{r.player_name}</td><td>{r.team}</td><td>{getattr(r,'proj_targets',0):.1f}</td><td>{getattr(r,'proj_rec_yards',0):.1f}</td><td>{getattr(r,'proj_rec_td',0):.2f}</td><td>{getattr(r,'proj_carries',0):.1f}</td><td>{getattr(r,'proj_rush_yards',0):.1f}</td><td>{getattr(r,'proj_rush_td',0):.2f}</td><td>{getattr(r,'proj_total_points',0):.1f}</td></tr>"
        for r in pproj.sort_values('proj_total_points', ascending=False).head(30).itertuples()
      )}
    </tbody>
  </table>

  <p style='color:#777'>Note: Injury adjustments apply multiplicative usage scaling by latest known status. Data sources are open (nflverse via nfl_data_py).</p>
</body>
</html>
"""

    ART_DIR.mkdir(parents=True, exist_ok=True)
    out = ART_DIR / f"slate_report_{season}_wk{week}.html"
    out.write_text(html, encoding="utf-8")
    return str(out)
