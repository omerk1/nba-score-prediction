"""
Persistence screen for possession-table aggregates.

For every candidate aggregate: split each team's season into consecutive
blocks of N games, compute the aggregate per block from pooled possessions,
and correlate block b with block b+1 across all teams (pooled) and after
demeaning within team. A stat that does not predict its own next block cannot
predict next-game outcomes, so this runs before any CV work. Also reports the
same-block correlation with raw net rating as a redundancy check.

Usage:
    venv/bin/python3 scripts/pbp_persistence_screen.py --seasons 2024-25 [--block-games 10]
"""

import argparse
import logging
import sqlite3
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data_processing.fetch_data import _date_to_season
from src.pbp.aggregates import assign_blocks, block_aggregates, persistence, sequence_stats_per_game, team_view
from src.pbp.shots import load_shots, location_adjusted_net_rating, shot_quality_blocks
from src.utils.config_loader import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
logger = logging.getLogger(__name__)

OUT_PATH = Path("outputs/pbp_persistence_screen.csv")

STATS = [
    # baselines
    "margin_pg", "net_rtg", "off_rtg", "def_rtg", "pace48",
    # context-filtered margins
    "net_rtg_nogarbage", "off_rtg_nogarbage", "def_rtg_nogarbage",
    "net_rtg_luckadj", "net_rtg_luckadj_nogarbage", "own_3p_luck_per100", "opp_fg3_pct",
    "net_rtg_clutch", "net_rtg_nonclutch", "net_rtg_trailing8", "net_rtg_leading8", "net_rtg_close",
    "net_rtg_1h", "net_rtg_2h",
    # style / four factors / shot mix
    "tov_rate", "oreb_per100", "fta_rate", "three_rate", "rim_rate", "mid_rate", "efg",
    # shot quality vs a league location baseline (offense and defense)
    "xpps", "pps", "pps_vs_x", "opp_xpps", "opp_pps", "opp_pps_vs_x",
    "net_rtg_xadj", "net_rtg_xadj_half",
    # lineup continuity and lineup-conditioned efficiency
    "top_lineup_share", "top3_lineup_share", "lineups_per100",
    "net_rtg_top1_lineup", "net_rtg_top5_lineups", "net_rtg_bench_lineups", "net_rtg_top5_minus_bench",
    # order-aware
    "largest_run_for", "largest_run_against", "lead_changes", "time_leading_share",
]


def load_possessions(pbp_db: str, raw_db: str, seasons: list[str]) -> pd.DataFrame:
    with sqlite3.connect(f"file:{pbp_db}?mode=ro", uri=True) as c:
        poss = pd.read_sql_query("SELECT * FROM possessions", c)
    with sqlite3.connect(f"file:{raw_db}?mode=ro", uri=True) as c:
        games = pd.read_sql_query("SELECT game_id, game_date, season_type FROM game", c)
    games = games[games.season_type == "Regular Season"]
    games["season"] = games.game_date.map(_date_to_season)
    games = games[games.season.isin(seasons)]
    poss = poss.merge(games[["game_id", "game_date", "season"]], on="game_id")
    return poss


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", required=True)
    ap.add_argument("--block-games", type=int, default=10)
    ap.add_argument("--min-block-games", type=int, default=None, help="keep partial blocks of at least this many games")
    args = ap.parse_args()
    cfg = load_config()
    seasons = [s.strip() for s in args.seasons.split(",")]

    poss = load_possessions(cfg.pbp.db_path, cfg.data_paths.raw_db, seasons)
    logger.info(f"{len(poss):,} possessions, {poss.game_id.nunique():,} games, seasons {seasons}")
    seq = sequence_stats_per_game(poss)

    with sqlite3.connect(f"file:{cfg.pbp.db_path}?mode=ro", uri=True) as c:
        shots = load_shots(c, list(poss.game_id.unique()))
    logger.info(f"{len(shots):,} field-goal attempts for the shot-quality baseline")

    results = []
    for season, sp in poss.groupby("season"):
        tv = assign_blocks(team_view(sp), args.block_games, args.min_block_games)
        blocks = block_aggregates(tv, seq)
        block_map = tv[["team", "game_id", "block"]].drop_duplicates()
        sq = shot_quality_blocks(shots[shots.game_id.isin(sp.game_id.unique())], block_map)
        blocks = blocks.merge(sq.reset_index(), on=["team", "block"], how="left")
        blocks["net_rtg_xadj"] = location_adjusted_net_rating(blocks)
        blocks["net_rtg_xadj_half"] = location_adjusted_net_rating(blocks, own_weight=0.5)
        blocks["season"] = season
        blocks["team_season"] = blocks.team.astype(str) + "_" + season
        results.append(blocks)
    blocks = pd.concat(results, ignore_index=True)
    # Persistence pairs never cross a season boundary: block ids restart per season,
    # and team_season keys the pairing.
    blocks_p = blocks.drop(columns="team").rename(columns={"team_season": "team"})
    logger.info(f"{len(blocks_p):,} team-blocks of {args.block_games} games")

    res = persistence(blocks_p, STATS).sort_values("r_next_margin", ascending=False)
    res["block_games"] = args.block_games
    res["seasons"] = ",".join(seasons)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT_PATH, mode="a", header=not OUT_PATH.exists(), index=False)
    pd.set_option("display.width", 160)
    print(res.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nwritten to {OUT_PATH}")


if __name__ == "__main__":
    main()
