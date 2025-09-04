from __future__ import annotations
from src.utils.config import parse_args
from src.etl.fetch_nflverse import run as etl_run
from src.etl.fetch_betting_weather import fetch_betting_lines, build_betting_game_features
from src.etl.fetch_injuries import fetch_injuries
from src.features.team_ratings import build_team_epa_rolling
from src.features.player_usage import build_player_usage
from src.features.context_features import build_context_features
from src.features.injury_adjustments import build_injury_adjustments
from src.models.enrich_game_features import build_game_model_table
from src.models.train_game_win_ext import train_and_save_extended
from src.models.player_stats_projections import build_player_stat_projections
from src.models.apply_injury_to_usage import apply_injury_to_player_projections
from src.models.season_sim import simulate_season
from src.models.predict_game_week import predict_week
from src.reports.slate_report import build_weekly_slate_report

def main():
    cfg = parse_args()
    print("[ETL] nflverse core...")
    etl_paths = etl_run(cfg.seasons)
    print(etl_paths)

    print("[ETL] betting lines (optional)...")
    bpath = fetch_betting_lines(cfg.seasons)
    if bpath:
        print("betting ->", bpath)
        print("[FEAT] betting features...")
        bf = build_betting_game_features()
        print("betting_features ->", bf)
    else:
        print("betting lines not available; continuing without.")

    print("[ETL] injuries (optional)...")
    ipath = fetch_injuries(cfg.seasons)
    if ipath:
        print("injuries ->", ipath)

    print("[FEAT] team ratings (EPA rolling)...")
    tr = build_team_epa_rolling()
    print("team_ratings ->", tr)

    print("[FEAT] player usage...")
    pu = build_player_usage()
    print("player_usage ->", pu)

    print("[FEAT] context features (rest/travel/dome)...")
    cf = build_context_features()
    print("context_features ->", cf)

    if ipath:
        print("[FEAT] injury adjustments...")
        ia = build_injury_adjustments()
        print("injury_adjustments ->", ia)

    print("[MODEL] build enriched game table...")
    tbl = build_game_model_table()
    print("game_model_table ->", tbl)

    print("[MODEL] train extended game-win model...")
    metrics = train_and_save_extended()
    print("extended metrics ->", metrics)

    print("[MODEL] player stat projections...")
    pstats = build_player_stat_projections()
    print("player_stat_projections ->", pstats)

    # Injury-apply to usage projections
    print("[MODEL] apply injury adjustments to usage projections...")
    adjp = apply_injury_to_player_projections()
    print("injury-adjusted usage projections ->", adjp)

    # Season sim for last season in range
    season = max(cfg.seasons)
    print(f"[SIM] Monte Carlo season {season}...")
    sres = simulate_season(season=season, sims=2000, use_extended=True)
    print("season_sim ->", sres)

    # Predict most recent regular-season week that exists in schedules for reporting
    # For simplicity assume week 1
    try:
        pw = predict_week(season=season, week=1)
        print("predictions ->", pw)
        rep = build_weekly_slate_report(season=season, week=1)
        print("slate report ->", rep)
    except Exception as e:
        print("Slate report generation skipped:", e)

if __name__ == "__main__":
    main()
