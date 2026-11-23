"""
Live comparison of the corrected model (src/serving/residual_shadow.py)
against the delivered picks, on logged games that have been played.

Reports margin MAE of both on the same games, then the strong picks (edge
>= 5%) split into: picked by both, live only (the corrected model drops
them), corrected only (it adds them): count, hit rate, flat-stake return at
the logged odds.

Decision rule, fixed before any live data: review once 300 games with a
correction are graded; the corrected picks go live only if their margin MAE
is not worse and the picks it changes did not do worse than the ones it
replaced. A sanity check, not a significance test: the offline evaluation
already measured the accuracy gain.

Usage (repo root):
    venv/bin/python3 scripts/compare_residual_correction.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.serving.grade_predictions import LOG_PATH, grade, load_results  # noqa: E402
from src.serving.recommended_bets import STRONG_EDGE  # noqa: E402
from src.serving.residual_shadow import corrected_rec  # noqa: E402

REVIEW_AT_GAMES = 300
KEY = ["game_date", "home_team_id", "away_team_id", "market", "side"]


def corrected_log(log: pd.DataFrame) -> pd.DataFrame:
    """Rows that carry a correction, once as logged and once as corrected."""
    has = log["rec_json"].map(lambda s: "residual_correction" in json.loads(s))
    live = log[has].copy()
    corr = live.copy()
    corr["rec_json"] = corr["rec_json"].map(lambda s: json.dumps(corrected_rec(json.loads(s))))
    return live, corr


def summarize_picks(bets: pd.DataFrame) -> dict:
    decided = bets[bets["outcome"] != "push"]
    return {
        "bets": len(bets),
        "hit_rate": round((decided["outcome"] == "win").mean(), 3) if len(decided) else None,
        "roi": round(bets["profit"].mean(), 3) if len(bets) else None,
    }


def main() -> None:
    log = pd.read_csv(LOG_PATH)
    # A same-day re-run logs the game again; keep the latest.
    log = log.sort_values("created_at").drop_duplicates(
        ["game_date", "home_team_id", "away_team_id"], keep="last"
    )
    live_log, corr_log = corrected_log(log)
    results = load_results()
    live, corr = grade(live_log, results), grade(corr_log, results)
    if live.empty:
        print(f"No graded games with a correction yet ({len(live_log)} logged).")
        return

    games = live.drop_duplicates(KEY[:3]).set_index(KEY[:3])
    cgames = corr.drop_duplicates(KEY[:3]).set_index(KEY[:3])
    n = len(games)
    print(f"Graded games with a correction: {n} (review at {REVIEW_AT_GAMES})")
    print(
        f"Margin MAE: live {games['pred_diff_error'].abs().mean():.2f}, "
        f"corrected {cgames['pred_diff_error'].abs().mean():.2f}"
    )

    both = live.merge(corr[KEY + ["edge"]], on=KEY, suffixes=("", "_corrected"))
    live_strong = both["edge"] >= STRONG_EDGE
    corr_strong = both["edge_corrected"] >= STRONG_EDGE
    rows = {
        "both": both[live_strong & corr_strong],
        "live only (dropped)": both[live_strong & ~corr_strong],
        "corrected only (new)": both[~live_strong & corr_strong],
    }
    print(f"\nStrong picks (edge >= {STRONG_EDGE:.0%}):")
    print(pd.DataFrame({k: summarize_picks(v) for k, v in rows.items()}).T.to_string())


if __name__ == "__main__":
    main()
