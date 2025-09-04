from __future__ import annotations
import os
from pathlib import Path
import pandas as pd

PROC_DIR = Path(os.getenv("PROC_DIR", "data/processed"))
ART_DIR = Path(os.getenv("ART_DIR", "data/artifacts"))

def apply_injury_to_player_projections() -> str:
    usage_proj = pd.read_csv(ART_DIR / "player_usage_projections.csv")
    # Most recent season per player already in the file; we align by (player_id, team, season)
    inj_path = PROC_DIR / "injury_adjustments.parquet"
    if not inj_path.exists():
        print("No injury adjustments found; copying projections through.")
        out = usage_proj.copy()
        out.to_csv(ART_DIR / "player_usage_projections_injury_adj.csv", index=False)
        return str(ART_DIR / "player_usage_projections_injury_adj.csv")

    inj = pd.read_parquet(inj_path)
    # Take the latest week row per player within season as the current status
    inj = inj.sort_values(["player_id","season","week"]).groupby(["player_id","season"]).tail(1)

    out = usage_proj.merge(inj[["player_id","season","inj_multiplier"]], on=["player_id","season"], how="left")
    out["inj_multiplier"] = out["inj_multiplier"].fillna(1.0)

    # Scale forecast shares by injury multiplier and renormalize per team roughly
    out["adj_carry_share_next"] = out["proj_carry_share_next"] * out["inj_multiplier"]
    out["adj_target_share_next"] = out["proj_target_share_next"] * out["inj_multiplier"]

    out.to_csv(ART_DIR / "player_usage_projections_injury_adj.csv", index=False)
    return str(ART_DIR / "player_usage_projections_injury_adj.csv")
