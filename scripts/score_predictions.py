"""Grade outputs/predictions_log.csv against final scores and print the
edge-bucket table (src/serving/grade_predictions.py). Read-only."""

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.serving import grade_predictions as G  # noqa: E402


def main() -> int:
    if not G.LOG_PATH.exists():
        print(f"{G.LOG_PATH} not found — nothing to grade yet.")
        return 1
    log = pd.read_csv(G.LOG_PATH)
    bets = G.grade(log, G.load_results())
    games = bets.drop_duplicates(["game_date", "home_team_id", "away_team_id"])
    print(f"{len(log)} logged games, {len(games)} played, {len(bets)} priced sides graded\n")
    if bets.empty:
        return 0

    print(
        f"Predicted diff MAE {games['pred_diff_error'].abs().mean():.2f} | "
        f"predicted total MAE {games['pred_total_error'].abs().mean():.2f}\n"
    )
    table = G.summarize(bets)
    table["hit_rate"] = table["hit_rate"].map("{:.1%}".format)
    table["model_prob"] = table["model_prob"].map("{:.1%}".format)
    table["roi"] = table["roi"].map("{:+.1%}".format)
    print(table.to_string())
    print("\nroi = profit per 1-unit flat stake on every side in the bucket (pushes = 0).")
    print("A real edge shows rising hit rate/ROI with the bucket and hit rate near model_prob.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
