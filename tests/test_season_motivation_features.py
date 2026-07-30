"""
Regression tests for `season_motivation.py` (`compute_standings_metrics`,
`compute_roster_behavior_scores`) and `feature_builder.py`'s
`_add_season_motivation_features`.

Standings/schedule need no new backfill (both are derived directly from the
`game` table); roster-behavior reuses `injury_features.sqlite`'s
`player_importance` + `player_injuries` tables, same as the injury pipeline's
own `_get_importance_map` formula. `motivation_score` combines the two via
`pressure_raw * (1 - roster_behavior_weight * roster_behavior_score)` (a raw-
component-columns redesign was tried and found to perform worse under CV, see
docs/SEASON_MOTIVATION_LOG.md section 5/7 -- reverted back to this combined-
score design). These tests lock in:

  1. The best-record team in a conference has games_to_clinch_ceiling == 0
     (nothing above to improve past -- the "no team above" convention).
  2. The worst-record team has games_to_clinch_floor == 0 (nothing below to
     fall behind -- the "no team below" convention).
  3. standings_pressure == 1.0 exactly when a team is tied at the playoff line
     (GB_from_line == 0), regardless of games remaining.
  4. standings_pressure decays toward 0 for a team far from the line with few
     games left, but stays substantial when far from the line with many games
     left -- the two-sided, games-remaining-moderated decay the formula is
     designed to produce.
  5. A player Out for 'Rest' (a non-injury reason) contributes to
     roster_behavior_score; a player Out for a genuine injury reason does not.
  6. min_importance_games gates out a player with too few weekly
     player_importance snapshots from counting toward full-strength quality.
  7. No player_importance history at all -> roster_behavior_score 0.0, not NaN
     (legitimate "nothing to report", not missing data).
  8. season_motivation.enabled=False short-circuits before touching any DB.
  9. Missing injury features DB soft-disables (warn + return df unchanged).
  10. motivation_score equals the documented formula exactly:
      clip(pressure_raw * (1 - roster_behavior_weight * roster_behavior_score), 0, 1).
  11. roster_behavior_weight=0.0 reduces motivation_score to pure standings
      pressure (the roster-behavior term fully disabled).
"""

import sqlite3
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.feature_engineering.feature_builder import FeatureBuilder
from src.feature_engineering.season_motivation import (
    compute_standings_metrics,
    compute_roster_behavior_scores,
    compute_recent_minutes_trend_scores,
)

# Four real East-conference team IDs (per season_motivation._TEAM_CONFERENCE)
# used for a controlled mini-season -- compute_standings_metrics only ranks
# teams that actually appear in the input games_df, so using just these 4 real
# IDs gives a clean, fully-controlled 4-team "conference" without needing to
# monkeypatch the real conference mapping.
TEAM_A = 1610612737  # ATL
TEAM_B = 1610612738  # BOS
TEAM_C = 1610612739  # CLE
TEAM_D = 1610612741  # CHI

SEASON_ID = 22024
GAME_COLS = ["GAME_ID", "GAME_DATE", "SEASON_ID", "HOME_TEAM_ID", "AWAY_TEAM_ID", "HOME_TEAM_WINS"]


def _game_row(game_id, game_date, home, away, home_wins):
    return {
        "GAME_ID": game_id, "GAME_DATE": pd.Timestamp(game_date), "SEASON_ID": SEASON_ID,
        "HOME_TEAM_ID": home, "AWAY_TEAM_ID": away, "HOME_TEAM_WINS": int(home_wins),
    }


def _round_robin_games():
    """4-team round robin (6 games), each team plays each other once.
    Entering 2024-01-05 (the last played date), standings are:
      A = 2-0 (undisputed 1st, no ties above)
      D = 0-2 (undisputed last, no ties below)
      B, C = 1-1 each (tied for the middle -- irrelevant to the ceiling==0 /
      floor==0 assertions below, which only depend on A being strictly best
      and D strictly worst, not on how the tied middle sorts)."""
    return pd.DataFrame([
        _game_row("g1", "2024-01-01", TEAM_A, TEAM_B, True),   # A 1-0, B 0-1
        _game_row("g2", "2024-01-01", TEAM_C, TEAM_D, True),   # C 1-0, D 0-1
        _game_row("g3", "2024-01-03", TEAM_A, TEAM_C, True),   # A 2-0, C 1-1
        _game_row("g4", "2024-01-03", TEAM_B, TEAM_D, True),   # B 1-1, D 0-2
        _game_row("g5", "2024-01-05", TEAM_A, TEAM_D, True),   # A 3-0, D 0-3 (after this game)
        _game_row("g6", "2024-01-05", TEAM_B, TEAM_C, False),  # B 1-2, C 2-1 (after this game)
    ])[GAME_COLS]


def _line_and_chaser_games(line_wins, chaser_wins, chaser_losses, chaser_future_games):
    """Team LINE (rank 1, playoff_line_seed=1) plays only filler opponent F;
    team CHASER plays only filler opponent F too (never LINE directly -- head-
    to-head isn't modeled). LINE's record is `line_wins`-0 with 0 games left.
    CHASER's record entering the snapshot date is `chaser_wins`-`chaser_losses`,
    with `chaser_future_games` additional (unplayed, dated after the snapshot)
    games against F to control games_remaining independently.
    """
    filler = TEAM_C
    rows = []
    date = pd.Timestamp("2024-01-01")
    gid = 0
    for _ in range(line_wins):
        rows.append(_game_row(f"line{gid}", date, TEAM_A, filler, True))
        gid += 1
        date += pd.Timedelta(days=1)
    chaser_date = pd.Timestamp("2024-01-01")
    cid = 0
    for _ in range(chaser_wins):
        rows.append(_game_row(f"chaser{cid}", chaser_date, TEAM_B, filler, True))
        cid += 1
        chaser_date += pd.Timedelta(days=1)
    for _ in range(chaser_losses):
        rows.append(_game_row(f"chaser{cid}", chaser_date, TEAM_B, filler, False))
        cid += 1
        chaser_date += pd.Timedelta(days=1)
    snapshot_date = chaser_date  # first date strictly after all played games
    # Anchor game (LINE vs filler again) exactly on snapshot_date -- guarantees
    # snapshot_date is an actual game date in the panel (compute_standings_metrics
    # only produces snapshots for dates something is played on), independent of
    # whether chaser_future_games happens to supply one itself. Harmless
    # duplicate-date scheduling, same as the round-robin test's g1/g2.
    rows.append(_game_row("anchor", snapshot_date, TEAM_A, filler, True))
    for _ in range(chaser_future_games):
        rows.append(_game_row(f"chaserfuture{cid}", chaser_date, TEAM_B, filler, True))
        cid += 1
        chaser_date += pd.Timedelta(days=1)
    return pd.DataFrame(rows)[GAME_COLS], snapshot_date


class TestStandingsMetrics:

    def test_best_record_team_has_zero_ceiling(self):
        """Undisputed best record (no ties above it) -> nothing above to
        improve past -> games_to_clinch_ceiling == 0 by convention."""
        panel = compute_standings_metrics(_round_robin_games(), playoff_line_seed=3)
        row = panel[(panel["team_id"] == TEAM_A) & (panel["snapshot_date"] == pd.Timestamp("2024-01-05"))]
        assert row["games_to_clinch_ceiling"].iloc[0] == 0.0

    def test_worst_record_team_has_zero_floor(self):
        """Undisputed worst record (no ties below it) -> nothing below to fall
        behind -> games_to_clinch_floor == 0 by convention."""
        panel = compute_standings_metrics(_round_robin_games(), playoff_line_seed=3)
        row = panel[(panel["team_id"] == TEAM_D) & (panel["snapshot_date"] == pd.Timestamp("2024-01-05"))]
        assert row["games_to_clinch_floor"].iloc[0] == 0.0

    def test_pressure_is_exactly_one_when_tied_at_the_line(self):
        """GB_from_line == 0 -> pressure_raw == 1.0 exactly, regardless of games
        remaining -- the formula's peak case."""
        games, snapshot_date = _line_and_chaser_games(line_wins=3, chaser_wins=3, chaser_losses=0, chaser_future_games=5)
        panel = compute_standings_metrics(games, playoff_line_seed=1)
        row = panel[(panel["team_id"] == TEAM_B) & (panel["snapshot_date"] == snapshot_date)]
        assert row["pressure_raw"].iloc[0] == pytest.approx(1.0)

    def test_pressure_decays_far_from_line_with_few_games_left(self):
        """Far from the line (GB large) with 0 games remaining -> pressure
        clips to exactly 0.0 -- mathematically eliminated/clinched reads as
        zero pressure, per the brief's stated boundary condition."""
        games, snapshot_date = _line_and_chaser_games(line_wins=10, chaser_wins=0, chaser_losses=10, chaser_future_games=0)
        panel = compute_standings_metrics(games, playoff_line_seed=1)
        row = panel[(panel["team_id"] == TEAM_B) & (panel["snapshot_date"] == snapshot_date)]
        assert row["pressure_raw"].iloc[0] == pytest.approx(0.0)

    def test_pressure_stays_substantial_far_from_line_with_many_games_left(self):
        """Same GB gap as the test above, but with 50 games still to play --
        pressure should NOT collapse to 0, since the race is still
        mathematically very much alive. This is the two-sided-decay property:
        pressure depends on the gap RELATIVE to games remaining, not the gap
        alone."""
        games, snapshot_date = _line_and_chaser_games(line_wins=10, chaser_wins=0, chaser_losses=10, chaser_future_games=50)
        panel = compute_standings_metrics(games, playoff_line_seed=1)
        row = panel[(panel["team_id"] == TEAM_B) & (panel["snapshot_date"] == snapshot_date)]
        # GB_from_line = 10, games_remaining = 50 -> pressure = 1 - 10/51
        assert row["pressure_raw"].iloc[0] == pytest.approx(1 - 10 / 51, abs=1e-6)


NON_INJURY_REASON = "Rest"
INJURY_REASON = "Injury/Illness - Left Ankle; Sprain"


def _write_injury_features_db(path, importance_rows, injury_rows):
    """importance_rows: list of (player_id, player_name, team_id, as_of_date,
    minutes_per_game, pts_per_game, usage_rate) tuples.
    injury_rows: list of (game_date, team_id, player_name, status, reason)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute(
        """CREATE TABLE player_importance (
            player_id INTEGER, player_name TEXT, team_id INTEGER, as_of_date TEXT,
            minutes_per_game REAL, pts_per_game REAL, usage_rate REAL, updated_at TEXT NOT NULL,
            PRIMARY KEY (player_id, team_id, as_of_date)
        )"""
    )
    conn.execute(
        """CREATE TABLE player_injuries (
            game_date TEXT NOT NULL, team_id INTEGER NOT NULL, player_name TEXT NOT NULL,
            status TEXT NOT NULL, reason TEXT, source TEXT NOT NULL DEFAULT 'pdf',
            PRIMARY KEY (game_date, team_id, player_name)
        )"""
    )
    for player_id, player_name, team_id, as_of_date, minutes, pts, usg in importance_rows:
        conn.execute(
            "INSERT INTO player_importance (player_id, player_name, team_id, as_of_date, "
            "minutes_per_game, pts_per_game, usage_rate, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (player_id, player_name, team_id, as_of_date, minutes, pts, usg, "2024-01-01T00:00:00"),
        )
    for game_date, team_id, player_name, status, reason in injury_rows:
        conn.execute(
            "INSERT INTO player_injuries (game_date, team_id, player_name, status, reason, source) "
            "VALUES (?, ?, ?, ?, ?, 'pdf')",
            (game_date, team_id, player_name, status, reason),
        )
    conn.commit()
    conn.close()


IMPORTANCE_WEIGHTS = MagicMock(minutes_share=0.4, usage_rate=0.4, pts_share=0.2)


# All 15 real East-conference team IDs (per season_motivation._TEAM_CONFERENCE),
# used for the dual-threshold pressure test below -- need a realistically
# full conference to have a meaningful, unambiguous rank 5/6/10.
_EAST_15 = [
    1610612737, 1610612738, 1610612739, 1610612741, 1610612748, 1610612749,
    1610612751, 1610612752, 1610612753, 1610612754, 1610612755, 1610612761,
    1610612764, 1610612765, 1610612766,
]
_FILLER_WEST_TEAM = 1610612744  # GSW -- West, never ranked within the East group


def _fifteen_team_tiers_games():
    """15 real East teams, each with a distinct win total against a shared West
    filler (losses=20 for every team, so win_pct differs cleanly by win count
    alone, no ties). Wins per team (rank1..rank15): a wide gap around the
    playoff_line_seed=10 boundary but a narrow, 1-win gap between rank5 and
    rank6 -- team at rank5 is comfortably clear of missing the postseason
    (10th) but in a real fight for a direct playoff berth (6th vs 7th). One
    game per calendar day (never two games the same day) to keep each team's
    cumulative win/loss sequence unambiguous under date normalization."""
    wins_by_rank = [50, 45, 40, 36, 33, 32, 20, 18, 16, 14, 12, 10, 8, 6, 4]
    rows = []
    gid = 0
    date = pd.Timestamp("2024-01-01")
    for team_id, wins in zip(_EAST_15, wins_by_rank):
        for _ in range(wins):
            rows.append({
                "GAME_ID": f"g{gid}", "GAME_DATE": date, "SEASON_ID": SEASON_ID,
                "HOME_TEAM_ID": team_id, "AWAY_TEAM_ID": _FILLER_WEST_TEAM, "HOME_TEAM_WINS": 1,
            })
            gid += 1
            date += pd.Timedelta(days=1)
        for _ in range(20):
            rows.append({
                "GAME_ID": f"g{gid}", "GAME_DATE": date, "SEASON_ID": SEASON_ID,
                "HOME_TEAM_ID": team_id, "AWAY_TEAM_ID": _FILLER_WEST_TEAM, "HOME_TEAM_WINS": 0,
            })
            gid += 1
            date += pd.Timedelta(days=1)
    # Anchor game (rank1 team vs filler, one more win) strictly after every
    # other row -- grounds a valid, unambiguous snapshot_date at which every
    # team's full accumulated record above is already "final" (no more of
    # their own games exist past this point).
    snapshot_date = date
    rows.append({
        "GAME_ID": "anchor", "GAME_DATE": snapshot_date, "SEASON_ID": SEASON_ID,
        "HOME_TEAM_ID": _EAST_15[0], "AWAY_TEAM_ID": _FILLER_WEST_TEAM, "HOME_TEAM_WINS": 1,
    })
    return pd.DataFrame(rows)[GAME_COLS], snapshot_date


class TestDualThresholdPressure:

    def test_single_threshold_misses_direct_berth_fight(self):
        """rank5 team is far from the 10-line (postseason cutoff) but in a
        real 1-win fight for the 6-line (direct berth) -- single-threshold
        pressure (playoff_line_seed=10 only, direct_playoff_seed omitted)
        reads this as near-zero, missing the real stakes entirely."""
        games, snapshot_date = _fifteen_team_tiers_games()
        panel = compute_standings_metrics(games, playoff_line_seed=10)
        rank5_team = _EAST_15[4]
        row = panel[(panel["team_id"] == rank5_team) & (panel["snapshot_date"] == snapshot_date)]
        assert row["pressure_raw"].iloc[0] == pytest.approx(0.0)

    def test_dual_threshold_catches_direct_berth_fight(self):
        """Same team/scenario, but with direct_playoff_seed=6 set -- pressure
        should now be nonzero, reflecting the real 6-vs-7 stakes the
        single-threshold version missed entirely. GB from the 6-line is 0.5
        with 0 games remaining -> pressure_direct = 1 - 0.5/1 = 0.5;
        pressure_postseason (from the 10-line) clips to 0.0 as in the test
        above. Default direct_playoff_weight=0.5 -> weighted average =
        0.5*0.5 + 0.5*0.0 = 0.25 (not the 0.5 a max-based combination would
        give -- see compute_standings_metrics' docstring for why max was
        replaced by a weighted average)."""
        games, snapshot_date = _fifteen_team_tiers_games()
        panel = compute_standings_metrics(games, playoff_line_seed=10, direct_playoff_seed=6)
        rank5_team = _EAST_15[4]
        row = panel[(panel["team_id"] == rank5_team) & (panel["snapshot_date"] == snapshot_date)]
        assert row["pressure_raw"].iloc[0] == pytest.approx(0.25)


class TestRosterBehaviorScores:

    def test_rest_reason_contributes_to_score(self, tmp_path):
        """A player Out for 'Rest' (non-injury) must contribute to
        roster_behavior_score -- the core behavioral-tanking signal."""
        db_path = tmp_path / "injury_features.sqlite"
        _write_injury_features_db(
            db_path,
            importance_rows=[
                (501, "Star Player", TEAM_A, "2024-01-01", 35.0, 25.0, 0.30),
                (501, "Star Player", TEAM_A, "2024-01-08", 35.0, 25.0, 0.30),
                (601, "Role Player", TEAM_A, "2024-01-01", 15.0, 8.0, 0.15),
                (601, "Role Player", TEAM_A, "2024-01-08", 15.0, 8.0, 0.15),
            ],
            injury_rows=[("2024-01-15", TEAM_A, "Star Player", "Out", NON_INJURY_REASON)],
        )
        team_dates = pd.DataFrame([{"team_id": TEAM_A, "game_date": pd.Timestamp("2024-01-15"), "season_id": SEASON_ID}])
        season_start = {SEASON_ID: pd.Timestamp("2023-10-01")}

        result = compute_roster_behavior_scores(team_dates, str(db_path), IMPORTANCE_WEIGHTS, min_importance_games=2,
                                                  season_start_by_season=season_start)
        row = result[(result["team_id"] == TEAM_A) & (result["game_date"] == pd.Timestamp("2024-01-15"))]
        assert row["roster_behavior_score"].iloc[0] > 0.0

    def test_injury_reason_does_not_contribute(self, tmp_path):
        """A player Out for a genuine injury reason must NOT contribute --
        only official non-injury reasons (rest, personal reasons, coach's
        decision) count as behavioral."""
        db_path = tmp_path / "injury_features.sqlite"
        _write_injury_features_db(
            db_path,
            importance_rows=[
                (501, "Star Player", TEAM_A, "2024-01-01", 35.0, 25.0, 0.30),
                (501, "Star Player", TEAM_A, "2024-01-08", 35.0, 25.0, 0.30),
                (601, "Role Player", TEAM_A, "2024-01-01", 15.0, 8.0, 0.15),
                (601, "Role Player", TEAM_A, "2024-01-08", 15.0, 8.0, 0.15),
            ],
            injury_rows=[("2024-01-15", TEAM_A, "Star Player", "Out", INJURY_REASON)],
        )
        team_dates = pd.DataFrame([{"team_id": TEAM_A, "game_date": pd.Timestamp("2024-01-15"), "season_id": SEASON_ID}])
        season_start = {SEASON_ID: pd.Timestamp("2023-10-01")}

        result = compute_roster_behavior_scores(team_dates, str(db_path), IMPORTANCE_WEIGHTS, min_importance_games=2,
                                                  season_start_by_season=season_start)
        row = result[(result["team_id"] == TEAM_A) & (result["game_date"] == pd.Timestamp("2024-01-15"))]
        assert row["roster_behavior_score"].iloc[0] == 0.0

    def test_min_importance_games_gate_excludes_thin_history(self, tmp_path):
        """A player with fewer than min_importance_games weekly snapshots must
        not count toward full_strength_quality -- guards against a
        single-game callup's one noisy snapshot skewing the baseline. Here the
        rested player only has 1 snapshot (gate=2), so they're excluded
        entirely and the score must be 0.0, not based on a thin sample."""
        db_path = tmp_path / "injury_features.sqlite"
        _write_injury_features_db(
            db_path,
            importance_rows=[
                (501, "Star Player", TEAM_A, "2024-01-08", 35.0, 25.0, 0.30),  # only 1 snapshot
            ],
            injury_rows=[("2024-01-15", TEAM_A, "Star Player", "Out", NON_INJURY_REASON)],
        )
        team_dates = pd.DataFrame([{"team_id": TEAM_A, "game_date": pd.Timestamp("2024-01-15"), "season_id": SEASON_ID}])
        season_start = {SEASON_ID: pd.Timestamp("2023-10-01")}

        result = compute_roster_behavior_scores(team_dates, str(db_path), IMPORTANCE_WEIGHTS, min_importance_games=2,
                                                  season_start_by_season=season_start)
        row = result[(result["team_id"] == TEAM_A) & (result["game_date"] == pd.Timestamp("2024-01-15"))]
        assert row["roster_behavior_score"].iloc[0] == 0.0

    def test_no_importance_history_returns_zero_not_nan(self, tmp_path):
        """No player_importance rows at all for this team -> 0.0, a legitimate
        "nothing to report" case, not a missing-data/NaN case."""
        db_path = tmp_path / "injury_features.sqlite"
        _write_injury_features_db(db_path, importance_rows=[], injury_rows=[])
        team_dates = pd.DataFrame([{"team_id": TEAM_A, "game_date": pd.Timestamp("2024-01-15"), "season_id": SEASON_ID}])
        season_start = {SEASON_ID: pd.Timestamp("2023-10-01")}

        result = compute_roster_behavior_scores(team_dates, str(db_path), IMPORTANCE_WEIGHTS, min_importance_games=2,
                                                  season_start_by_season=season_start)
        row = result[(result["team_id"] == TEAM_A) & (result["game_date"] == pd.Timestamp("2024-01-15"))]
        assert row["roster_behavior_score"].iloc[0] == 0.0
        assert not pd.isna(row["roster_behavior_score"].iloc[0])


class TestRecentMinutesTrendScores:

    def test_genuine_minutes_drop_contributes_to_score(self, tmp_path):
        """A player whose cumulative minutes-per-game average dropped
        meaningfully between a snapshot >= lookback_weeks ago and the current
        one must contribute a nonzero recent_minutes_trend_score -- the core
        "soft tanking" signal, distinct from roster_behavior_score (which only
        sees an official Out designation)."""
        db_path = tmp_path / "injury_features.sqlite"
        _write_injury_features_db(
            db_path,
            importance_rows=[
                (501, "Star Player", TEAM_A, "2023-12-01", 35.0, 25.0, 0.30),  # prior (>=4 weeks before target)
                (501, "Star Player", TEAM_A, "2023-12-08", 35.0, 25.0, 0.30),
                (501, "Star Player", TEAM_A, "2024-01-08", 20.0, 14.0, 0.30),  # current -- minutes cut nearly in half
            ],
            injury_rows=[],
        )
        team_dates = pd.DataFrame([{"team_id": TEAM_A, "game_date": pd.Timestamp("2024-01-15"), "season_id": SEASON_ID}])
        season_start = {SEASON_ID: pd.Timestamp("2023-10-01")}

        result = compute_recent_minutes_trend_scores(team_dates, str(db_path), IMPORTANCE_WEIGHTS, min_importance_games=2,
                                                       season_start_by_season=season_start, lookback_weeks=4)
        row = result[(result["team_id"] == TEAM_A) & (result["game_date"] == pd.Timestamp("2024-01-15"))]
        assert row["recent_minutes_trend_score"].iloc[0] > 0.0

    def test_flat_or_increased_minutes_gives_zero(self, tmp_path):
        """A player whose minutes stayed flat or increased must contribute
        exactly 0.0 -- no "bonus" for playing more than the season norm, only
        reductions count."""
        db_path = tmp_path / "injury_features.sqlite"
        _write_injury_features_db(
            db_path,
            importance_rows=[
                (501, "Star Player", TEAM_A, "2023-12-01", 30.0, 20.0, 0.28),
                (501, "Star Player", TEAM_A, "2023-12-08", 30.0, 20.0, 0.28),
                (501, "Star Player", TEAM_A, "2024-01-08", 34.0, 24.0, 0.30),  # up, not down
            ],
            injury_rows=[],
        )
        team_dates = pd.DataFrame([{"team_id": TEAM_A, "game_date": pd.Timestamp("2024-01-15"), "season_id": SEASON_ID}])
        season_start = {SEASON_ID: pd.Timestamp("2023-10-01")}

        result = compute_recent_minutes_trend_scores(team_dates, str(db_path), IMPORTANCE_WEIGHTS, min_importance_games=2,
                                                       season_start_by_season=season_start, lookback_weeks=4)
        row = result[(result["team_id"] == TEAM_A) & (result["game_date"] == pd.Timestamp("2024-01-15"))]
        assert row["recent_minutes_trend_score"].iloc[0] == 0.0

    def test_no_prior_enough_snapshot_gives_zero(self, tmp_path):
        """A player with only recent snapshots (nothing at or before the
        lookback cutoff) has no valid comparison point -- must be excluded,
        not treated as a drop from an implicit zero."""
        db_path = tmp_path / "injury_features.sqlite"
        _write_injury_features_db(
            db_path,
            importance_rows=[
                (501, "Rookie", TEAM_A, "2024-01-01", 20.0, 12.0, 0.20),  # too recent for a 4-week lookback
                (501, "Rookie", TEAM_A, "2024-01-08", 15.0, 9.0, 0.18),
            ],
            injury_rows=[],
        )
        team_dates = pd.DataFrame([{"team_id": TEAM_A, "game_date": pd.Timestamp("2024-01-15"), "season_id": SEASON_ID}])
        season_start = {SEASON_ID: pd.Timestamp("2023-10-01")}

        result = compute_recent_minutes_trend_scores(team_dates, str(db_path), IMPORTANCE_WEIGHTS, min_importance_games=2,
                                                       season_start_by_season=season_start, lookback_weeks=4)
        row = result[(result["team_id"] == TEAM_A) & (result["game_date"] == pd.Timestamp("2024-01-15"))]
        assert row["recent_minutes_trend_score"].iloc[0] == 0.0

    def test_no_importance_history_returns_zero_not_nan(self, tmp_path):
        """No player_importance rows at all -> 0.0, a legitimate "nothing to
        report" case, not a missing-data/NaN case."""
        db_path = tmp_path / "injury_features.sqlite"
        _write_injury_features_db(db_path, importance_rows=[], injury_rows=[])
        team_dates = pd.DataFrame([{"team_id": TEAM_A, "game_date": pd.Timestamp("2024-01-15"), "season_id": SEASON_ID}])
        season_start = {SEASON_ID: pd.Timestamp("2023-10-01")}

        result = compute_recent_minutes_trend_scores(team_dates, str(db_path), IMPORTANCE_WEIGHTS, min_importance_games=2,
                                                       season_start_by_season=season_start, lookback_weeks=4)
        row = result[(result["team_id"] == TEAM_A) & (result["game_date"] == pd.Timestamp("2024-01-15"))]
        assert row["recent_minutes_trend_score"].iloc[0] == 0.0
        assert not pd.isna(row["recent_minutes_trend_score"].iloc[0])


def _mock_config(raw_db_path, injury_db_path, enabled: bool = True, playoff_line_seed: int = 10,
                  direct_playoff_seed: int = None, direct_playoff_weight: float = 0.5,
                  roster_behavior_weight: float = 1.0, min_importance_games: int = 5,
                  recent_trend_lookback_weeks: int = 4):
    mock_cfg = MagicMock()
    mock_cfg.data_paths = MagicMock(raw_db=str(raw_db_path))
    mock_cfg.season_motivation = MagicMock(
        enabled=enabled, playoff_line_seed=playoff_line_seed, direct_playoff_seed=direct_playoff_seed,
        direct_playoff_weight=direct_playoff_weight,
        roster_behavior_weight=roster_behavior_weight, min_importance_games=min_importance_games,
        recent_trend_lookback_weeks=recent_trend_lookback_weeks,
    )
    mock_cfg.injury_features = MagicMock(db_path=str(injury_db_path), importance_weights=IMPORTANCE_WEIGHTS)
    mock_cfg.datasets_loading = MagicMock(
        data_start_date="2023-10-01", test_end_date="2024-06-01", allowed_season_types=["Regular Season"],
    )
    return mock_cfg


def _write_game_db(path, games_df):
    """Minimal `game` table matching NBADataLoader._GAME_SELECT's required
    columns -- box-score columns not needed for season_motivation are left as
    0/NULL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute(
        """CREATE TABLE game (
            game_id TEXT PRIMARY KEY, game_date TEXT, season_id TEXT, season_type TEXT,
            team_id_home INTEGER, team_id_away INTEGER, pts_home REAL, pts_away REAL, wl_home TEXT,
            fg_pct_home REAL, ft_pct_home REAL, fg3_pct_home REAL, ast_home INTEGER, reb_home INTEGER,
            fg_pct_away REAL, ft_pct_away REAL, fg3_pct_away REAL, ast_away INTEGER, reb_away INTEGER,
            fgm_home INTEGER, fga_home INTEGER, fg3m_home INTEGER, fg3a_home INTEGER, ftm_home INTEGER, fta_home INTEGER,
            fgm_away INTEGER, fga_away INTEGER, fg3m_away INTEGER, fg3a_away INTEGER, ftm_away INTEGER, fta_away INTEGER
        )"""
    )
    for _, row in games_df.iterrows():
        conn.execute(
            "INSERT INTO game (game_id, game_date, season_id, season_type, team_id_home, team_id_away, "
            "pts_home, pts_away, wl_home) VALUES (?, ?, ?, 'Regular Season', ?, ?, 100, 90, ?)",
            (
                row["GAME_ID"], row["GAME_DATE"].strftime("%Y-%m-%d"), str(row["SEASON_ID"]),
                int(row["HOME_TEAM_ID"]), int(row["AWAY_TEAM_ID"]), "W" if row["HOME_TEAM_WINS"] else "L",
            ),
        )
    conn.commit()
    conn.close()


def _query_df(game_date, home_team_id=TEAM_A, away_team_id=TEAM_B, season_id=SEASON_ID):
    return pd.DataFrame([{
        "GAME_ID": "target", "GAME_DATE": pd.Timestamp(game_date), "SEASON_ID": season_id,
        "HOME_TEAM_ID": home_team_id, "AWAY_TEAM_ID": away_team_id,
    }])


class TestAddSeasonMotivationFeatures:

    @patch("src.feature_engineering.feature_builder.load_config")
    def test_disabled_returns_df_unchanged(self, mock_config, tmp_path):
        mock_config.return_value = _mock_config(tmp_path / "nonexistent.sqlite", tmp_path / "nonexistent2.sqlite", enabled=False)
        df = _query_df("2024-01-15")
        fb = FeatureBuilder(rolling_windows=[3])
        result = fb._add_season_motivation_features(df)
        pd.testing.assert_frame_equal(result, df)

    @patch("src.feature_engineering.feature_builder.load_config")
    def test_missing_injury_db_soft_disables(self, mock_config, tmp_path):
        raw_db = tmp_path / "nba_api.sqlite"
        _write_game_db(raw_db, _round_robin_games())
        mock_config.return_value = _mock_config(raw_db, tmp_path / "does_not_exist.sqlite", enabled=True)
        df = _query_df("2024-01-05")
        fb = FeatureBuilder(rolling_windows=[3])
        result = fb._add_season_motivation_features(df)
        pd.testing.assert_frame_equal(result, df)

    @patch("src.feature_engineering.feature_builder.load_config")
    def test_motivation_score_matches_documented_formula(self, mock_config, tmp_path):
        """motivation_score must equal clip(pressure_raw * (1 -
        roster_behavior_weight * roster_behavior_score), 0, 1) exactly --
        locks in the actual combination formula, not just its bounds."""
        raw_db = tmp_path / "nba_api.sqlite"
        injury_db = tmp_path / "injury_features.sqlite"
        _write_game_db(raw_db, _round_robin_games())
        _write_injury_features_db(
            injury_db,
            importance_rows=[
                (501, "Star Player", TEAM_A, "2024-01-01", 35.0, 25.0, 0.30),
                (501, "Star Player", TEAM_A, "2024-01-08", 35.0, 25.0, 0.30),
            ],
            injury_rows=[("2024-01-05", TEAM_A, "Star Player", "Out", NON_INJURY_REASON)],
        )
        mock_config.return_value = _mock_config(raw_db, injury_db, enabled=True, playoff_line_seed=3,
                                                  roster_behavior_weight=0.6, min_importance_games=2)

        df = _query_df("2024-01-05", home_team_id=TEAM_A, away_team_id=TEAM_D)
        fb = FeatureBuilder(rolling_windows=[3])
        result = fb._add_season_motivation_features(df)

        standings = compute_standings_metrics(_round_robin_games(), playoff_line_seed=3)
        pressure = standings[
            (standings["team_id"] == TEAM_A) & (standings["snapshot_date"] == pd.Timestamp("2024-01-05"))
        ]["pressure_raw"].iloc[0]
        team_dates = pd.DataFrame([{"team_id": TEAM_A, "game_date": pd.Timestamp("2024-01-05"), "season_id": SEASON_ID}])
        roster_score = compute_roster_behavior_scores(
            team_dates, str(injury_db), IMPORTANCE_WEIGHTS, min_importance_games=2,
            season_start_by_season={SEASON_ID: pd.Timestamp("2023-10-01")},
        )["roster_behavior_score"].iloc[0]

        expected = max(0.0, min(1.0, pressure * (1 - 0.6 * roster_score)))
        assert result.loc[0, "home_team_motivation_score"] == pytest.approx(expected)

    @patch("src.feature_engineering.feature_builder.load_config")
    def test_zero_roster_behavior_weight_reduces_to_pure_pressure(self, mock_config, tmp_path):
        """roster_behavior_weight=0.0 must fully disable the roster-behavior
        term -- motivation_score should equal standings pressure alone, even
        with a resting player present."""
        raw_db = tmp_path / "nba_api.sqlite"
        injury_db = tmp_path / "injury_features.sqlite"
        _write_game_db(raw_db, _round_robin_games())
        _write_injury_features_db(
            injury_db,
            importance_rows=[
                (501, "Star Player", TEAM_A, "2024-01-01", 35.0, 25.0, 0.30),
                (501, "Star Player", TEAM_A, "2024-01-08", 35.0, 25.0, 0.30),
            ],
            injury_rows=[("2024-01-05", TEAM_A, "Star Player", "Out", NON_INJURY_REASON)],
        )
        mock_config.return_value = _mock_config(raw_db, injury_db, enabled=True, playoff_line_seed=3,
                                                  roster_behavior_weight=0.0, min_importance_games=2)

        df = _query_df("2024-01-05", home_team_id=TEAM_A, away_team_id=TEAM_D)
        fb = FeatureBuilder(rolling_windows=[3])
        result = fb._add_season_motivation_features(df)

        standings = compute_standings_metrics(_round_robin_games(), playoff_line_seed=3)
        pressure = standings[
            (standings["team_id"] == TEAM_A) & (standings["snapshot_date"] == pd.Timestamp("2024-01-05"))
        ]["pressure_raw"].iloc[0]

        assert result.loc[0, "home_team_motivation_score"] == pytest.approx(pressure)
