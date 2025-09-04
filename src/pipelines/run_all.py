from __future__ import annotations
import json
from src.utils.config import parse_args
from src.etl.fetch_nflverse import run as etl_run
from src.features.team_ratings import build_team_epa_rolling
from src.features.player_usage import build_player_usage
from src.models.train_game_win import train_and_save
from src.models.player_projections import build_simple_usage_projections

def main():
    cfg = parse_args()
    print("[ETL] fetching open nflverse data...")
    etl_paths = etl_run(cfg.seasons)
    print(json.dumps(etl_paths, indent=2))

    print("[FEAT] building team ratings...")
    team_path = build_team_epa_rolling()
    print(f"team_ratings -> {team_path}")

    print("[FEAT] building player usage...")
    usage_path = build_player_usage()
    print(f"player_usage -> {usage_path}")

    print("[MODEL] training game win model...")
    metrics = train_and_save()
    print(f"game_win metrics: {metrics}")

    print("[MODEL] building simple player usage projections...")
    proj_path = build_simple_usage_projections()
    print(f"player projections -> {proj_path}")

if __name__ == "__main__":
    main()
