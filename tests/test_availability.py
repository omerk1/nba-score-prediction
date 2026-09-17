"""
Availability module tests: name normalization, the report-date -> game-date
label rule, and the point-in-time bound on retrieval (every retrieved fact
strictly predates the report date).
"""

import sqlite3

import numpy as np
import pandas as pd
import pytest

from src.availability import labels as L
from src.availability import retrieval as R
from src.availability.names import normalize_name, squash


class TestNormalizeName:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("Jaren Jackson Jr.", "jaren jackson"),
            ("Nikola Jokić", "nikola jokic"),
            ("Dereck LivelyII", "dereck lively"),
            ("Robert WilliamsIII", "robert williams"),
            ("Gary TrentJr.", "gary trent"),
            ("P.J. Washington", "p j washington"),
            ("Williams, Robert", "robert williams"),
            ("", ""),
        ],
    )
    def test_examples(self, raw, expected):
        assert normalize_name(raw) == expected

    def test_squash(self):
        assert squash("robert williams") == "robertwilliams"


def _game_log():
    # Team 1 plays on 01-10 and 01-12; player 7 plays 01-10 only, player 8 plays both.
    return pd.DataFrame(
        {
            "player_id": [7, 8, 8],
            "player_name": ["Ann Smith", "Bo Jones", "Bo Jones"],
            "team_id": [1, 1, 1],
            "game_id": ["g1", "g1", "g2"],
            "game_date": ["2024-01-10", "2024-01-10", "2024-01-12"],
            "season": ["2023-24"] * 3,
            "season_type": ["Regular Season"] * 3,
            "minutes": [20.0, 30.0, 31.0],
            "pts": [5.0, 10.0, 12.0],
        }
    )


@pytest.fixture
def dbs(tmp_path):
    inj = tmp_path / "inj.sqlite"
    with sqlite3.connect(inj) as c:
        c.execute(
            "CREATE TABLE player_injuries (game_date TEXT, team_id INTEGER, player_name TEXT, "
            "status TEXT, reason TEXT, source TEXT)"
        )
        rows = [
            (
                "2024-01-09",
                1,
                "Ann Smith",
                "Questionable",
                "Injury/Illness - Left Ankle; Sprain",
                "pdf",
            ),  # -> g1, played
            (
                "2024-01-11",
                1,
                "Ann Smith",
                "Questionable",
                "Injury/Illness - Left Ankle; Sprain",
                "pdf",
            ),  # -> g2, DNP
            (
                "2024-01-11",
                1,
                "Bo Jones",
                "Doubtful",
                "Injury/Illness - Right Knee; Soreness",
                "pdf",
            ),  # -> g2, played
            ("2024-01-10", 1, "Bo Jones", "Questionable", "G League - Two-Way", "pdf"),  # gleague, dropped
            ("2024-01-20", 1, "Ann Smith", "Questionable", "", "pdf"),  # no team game, dropped
            ("2024-01-11", 1, "Nobody Here", "Questionable", "", "pdf"),  # unresolved, dropped
            (
                "2024-01-09",
                1,
                "Bo Jones",
                "Out",
                "Injury/Illness - Right Knee; Soreness",
                "pdf",
            ),  # history only
        ]
        c.executemany("INSERT INTO player_injuries VALUES (?,?,?,?,?,?)", rows)
        c.execute(
            "CREATE TABLE player_importance (player_id INTEGER, player_name TEXT, team_id INTEGER, "
            "as_of_date TEXT, minutes_per_game REAL, pts_per_game REAL, usage_rate REAL, updated_at TEXT)"
        )
        c.executemany(
            "INSERT INTO player_importance VALUES (?,?,?,?,?,?,?,?)",
            [
                (7, "Ann Smith", 1, "2024-01-08", 20.0, 5.0, 0.2, ""),
                (
                    7,
                    "Ann Smith",
                    1,
                    "2024-01-11",
                    99.0,
                    99.0,
                    0.9,
                    "",
                ),  # dated ON the second report day: must not be used
            ],
        )
    avail = tmp_path / "avail.sqlite"
    from src.availability.db import get_conn

    conn = get_conn(str(avail))
    conn.executemany(
        "INSERT INTO player_game_log VALUES (?,?,?,?,?,?,?,?,?)",
        _game_log().itertuples(index=False, name=None),
    )
    conn.commit()
    conn.close()
    return str(inj), str(avail)


class TestLabels:
    def test_next_day_rule_and_drops(self, dbs):
        inj, avail = dbs
        df, rep = L.build_labels(inj, avail)
        assert rep.n_listings == 6  # Out row excluded by status filter
        assert rep.n_gleague == 1
        assert rep.n_no_team_game == 1
        assert rep.n_unresolved == 1
        assert rep.n_labeled == 3
        assert list(df["game_date"]) == ["2024-01-10", "2024-01-12", "2024-01-12"]
        by = df.set_index(["player_name", "game_date"])["played"]
        assert by[("Ann Smith", "2024-01-10")] == 1
        assert by[("Ann Smith", "2024-01-12")] == 0
        assert by[("Bo Jones", "2024-01-12")] == 1

    def test_history_includes_out(self, dbs):
        inj, avail = dbs
        h = L.build_history(inj, avail)
        assert (h["status"] == "Out").sum() == 1


class TestRetrievalPointInTime:
    def test_facts_strictly_before_report_date(self, dbs, monkeypatch):
        inj, avail = dbs
        labels, _ = L.build_labels(inj, avail)
        history = L.build_history(inj, avail)
        games = pd.DataFrame(
            {
                "game_id": ["g1", "g2"],
                "game_date": ["2024-01-10", "2024-01-12"],
                "team_id_home": [1, 2],
                "team_id_away": [2, 1],
                "pts_home": [100, 90],
                "pts_away": [95, 99],
            }
        )

        class _SW:
            severe, moderate, minor = 1.0, 0.6, 0.3

        class _Cfg:
            class injury_features:
                severity_weights = _SW()

        monkeypatch.setattr(R, "load_config", lambda: _Cfg())
        ctx = R.build_context(labels, history, L.load_game_log(avail), R.load_importance(inj), games)
        ctx = ctx.set_index(["player_name", "report_date"])

        first = ctx.loc[("Ann Smith", "2024-01-09")]
        # Nothing has happened before the first report: no prior listings, no minutes.
        assert first["own_n_prior_uncertain"] == 0
        assert np.isnan(first["min_last10_mean"])
        assert first["imp_minutes_per_game"] == 20.0

        second = ctx.loc[("Ann Smith", "2024-01-11")]
        # The 01-10 game (before the 01-11 report) is visible; the 01-12 outcome is not.
        assert second["own_n_prior_uncertain"] == 1
        assert second["own_prior_play_rate"] == 1.0
        assert second["own_same_reason_play_rate"] == 1.0
        assert second["min_last_game"] == 20.0
        assert second["days_since_last_played"] == 1
        # Importance snapshot dated on the report day itself is excluded.
        assert second["imp_minutes_per_game"] == 20.0
        assert second["is_home"] == 0 and second["rest_days"] == 2

        bo = ctx.loc[("Bo Jones", "2024-01-11")]
        assert bo["last_status_out"] == 1
        assert bo["listed_streak_days"] == 0  # listed 01-09 and 01-10 (gleague row is filtered from history)

    def test_context_columns_complete(self):
        assert len(set(R.CONTEXT_COLUMNS)) == len(R.CONTEXT_COLUMNS)


class TestReasonFlags:
    def test_illness_prefix_not_counted(self):
        class _SW:
            severe, moderate, minor = 1.0, 0.6, 0.3

        df = pd.DataFrame(
            {
                "status": ["Questionable"] * 3,
                "reason": ["Injury/Illness - Left Ankle; Sprain", "Injury/Illness - N/a; Illness", ""],
            }
        )
        f = R._reason_flags(df, _SW())
        assert list(f["is_illness"]) == [0, 1, 0]
        assert list(f["severity"]) == [0.6, 0.3, 0.3]
        assert list(f["reason_empty"]) == [0, 0, 1]
