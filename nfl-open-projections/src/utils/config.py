from __future__ import annotations
import argparse
from dataclasses import dataclass

@dataclass
class RunConfig:
    seasons: list[int]

def parse_args() -> RunConfig:
    p = argparse.ArgumentParser(description="NFL open projections pipeline")
    p.add_argument("--seasons", required=True, type=str,
                   help="Season range like 2019-2024 or list like 2019,2020,2021")
    args = p.parse_args()
    txt = args.seasons.strip()
    if "-" in txt:
        start, end = [int(x) for x in txt.split("-")]
        years = list(range(start, end+1))
    else:
        years = [int(x) for x in txt.split(",")]
    return RunConfig(seasons=years)
