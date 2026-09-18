"""
Render one retrieval context row into the LLM prompt.

Two variants:
- anonymized (default): no player name, team, opponent, or date. Only the
  listing text, the numeric history, and game context. This is the variant
  whose scores count.
- named: adds player, team and game date. Used ONLY for the memorization check
  (if the named variant scores better on pre-cutoff seasons, the model is
  recalling outcomes rather than reasoning).

The facts are exactly the CONTEXT_COLUMNS the tabular baselines consume, so
the LLM and the baselines are compared on identical information.
"""

import math

import numpy as np
import pandas as pd
from nba_api.stats.static import teams as nba_teams

_TEAM_ABBR = {t["id"]: t["abbreviation"] for t in nba_teams.get_teams()}

SYSTEM_INSTRUCTIONS = (
    "You estimate whether an NBA player listed on the official pre-game injury report "
    "will play in the game the report covers (the next day). You are given the listing, "
    "the player's own history on prior reports, their recent playing time, and the game "
    "context. All numbers describe the situation before the report was published. "
    "League base rates: players listed Questionable play about 53% of the time, "
    "Doubtful about 6%. Move away from the base rate only when the facts justify it. "
    "Return a JSON object with exactly these fields: "
    '"p_play" (float in [0,1], probability the player plays), '
    '"expected_minutes_share" (float in [0,1], fraction of their usual minutes if they play), '
    '"rationale" (one sentence).'
)


def _num(x, fmt="{:.0f}", unknown="unknown"):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return unknown
    return fmt.format(x)


def _pct(x):
    return _num(x, "{:.0%}")


def build_few_shot_block(train: pd.DataFrame, n_shots: int, variant: str, seed: int = 7) -> str:
    """Worked examples with their real outcomes, drawn only from training rows.

    Stratified by status so Doubtful is represented, and sorted so the block is
    stable for a given (train set, n, seed) and therefore cache-friendly.
    """
    if not n_shots or train.empty:
        return ""
    rng = np.random.default_rng(seed)
    n_doubtful = max(1, round(n_shots * float((train["status"] == "Doubtful").mean())))
    picks = []
    for status, k in (("Doubtful", n_doubtful), ("Questionable", n_shots - n_doubtful)):
        pool = train.index[train["status"] == status]
        if len(pool):
            picks.extend(rng.choice(pool, size=min(k, len(pool)), replace=False))
    parts = ["Worked examples from earlier seasons, with what actually happened:\n"]
    for i, idx in enumerate(sorted(picks), 1):
        row = train.loc[idx]
        outcome = "PLAYED" if row["played"] == 1 else "DID NOT PLAY"
        parts.append(f"Example {i}:\n{render_prompt(row, variant)}\nOutcome: {outcome}\n")
    parts.append("Now estimate for this case:\n")
    return "\n".join(parts)


def render_prompt(row: pd.Series, variant: str = "anonymized") -> str:
    if variant not in ("anonymized", "named"):
        raise ValueError(variant)
    lines = []
    if variant == "named":
        lines.append(
            f"Player: {row['player_name']} ({_TEAM_ABBR.get(int(row['team_id']), '?')}), "
            f"game date {row['game_date']}, report date {row['report_date']}."
        )
    reason = (row.get("reason") or "").strip() or "(no reason given)"
    lines.append(f'Listing: status={row["status"]}; reason="{reason}".')

    n_unc = int(row["own_n_prior_uncertain"])
    hist = f"Prior uncertain listings for this player: {n_unc}"
    if n_unc:
        hist += f", played in {_pct(row['own_prior_play_rate'])} of them"
    n_same = int(row["own_n_same_reason"])
    hist += f"; with this same reason: {n_same}"
    if n_same:
        hist += f", played in {_pct(row['own_same_reason_play_rate'])}"
    hist += (
        f". Consecutive prior days on the report: {int(row['listed_streak_days'])}. "
        f"Most recent prior listing status: {'Out' if row['last_status_out'] else 'not Out / none'}."
    )
    lines.append(hist)

    if math.isnan(row["min_last10_mean"]):
        lines.append("Recent playing time: no games played yet this season before this report.")
    else:
        lines.append(
            f"Recent playing time: {_num(row['min_last10_mean'], '{:.1f}')} min avg over last 10 games; "
            f"{_num(row['min_last_game'])} min in last game; "
            f"{_num(row['days_since_last_played'])} days since last game played; "
            f"missed {int(row['team_games_missed_last10'])} of the team's last 10 games; "
            f"played {_pct(row['season_games_played_share'])} of team games this season."
        )
    lines.append(
        f"Season profile: {_num(row['imp_minutes_per_game'], '{:.1f}')} min/game, "
        f"{_num(row['imp_pts_per_game'], '{:.1f}')} pts/game, usage {_pct(row['imp_usage_rate'])}."
    )
    venue = "home" if row["is_home"] == 1 else ("away" if row["is_home"] == 0 else "unknown venue")
    lines.append(
        f"Game: {venue}; {_num(row['rest_days'])} rest days"
        f"{' (second night of a back-to-back)' if row['back_to_back'] == 1 else ''}; "
        f"day {_num(row['days_into_season'])} of the season; "
        f"team win pct {_num(row['team_win_pct_std'], '{:.2f}')}, opponent {_num(row['opp_win_pct_std'], '{:.2f}')}."
    )
    return "\n".join(lines)
