"""
Tests for the game-date-aware injury PDF parser: date parsing, carry-forward
across rows and continuation pages, and team-name resolution including the
alias that previously dropped every Clippers listing.
"""

import pytest

from src.news_scraping.scrapers import nba_injury_pdf as P


class TestParseGameDate:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("10/21/2025", "2025-10-21"),
            ("01/05/26", "2026-01-05"),
            ("2025-10-21", "2025-10-21"),
            ("", None),
            ("   ", None),
            (None, None),
            ("not a date", None),
        ],
    )
    def test_formats(self, raw, expected):
        assert P._parse_game_date(raw) == expected


class TestTeamResolution:
    def test_clippers_alias_resolves(self):
        """The reports write 'LA Clippers'; nba_api's full_name is 'Los Angeles
        Clippers'. Before the alias, every Clippers row was silently dropped."""
        assert P._TEAM_MAP.get("LA Clippers") is None
        assert P._TEAM_MAP_ALIASES["LA Clippers"] == "LAC"
        assert P._TEAM_MAP_CONCAT["LAClippers"] == "LAC"

    def test_standard_names_still_resolve(self):
        assert P._TEAM_MAP["Boston Celtics"] == "BOS"
        assert P._TEAM_MAP_CONCAT["BostonCeltics"] == "BOS"


class _FakePage:
    def __init__(self, table):
        self._table = table

    def extract_table(self, *a, **k):
        return self._table


class _FakePdf:
    def __init__(self, pages):
        self.pages = [_FakePage(p) for p in pages]

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


HEADER = ["Game Date", "Game Time", "Matchup", "Team", "Player Name", "Current Status", "Reason"]


def _row(date="", team="", player="", status="Out", reason="x"):
    return [date, "7:00 (ET)", "BOS@NYK", team, player, status, reason]


@pytest.fixture
def parse(monkeypatch):
    def run(pages):
        monkeypatch.setattr(P.pdfplumber, "open", lambda _: _FakePdf(pages))
        return P._parse_pdf(b"")

    return run


class TestParsePdf:
    def test_date_and_team_carry_forward(self, parse):
        page = [
            HEADER,
            _row("10/21/2025", "Boston Celtics", "Jayson Tatum"),
            _row("", "", "Jaylen Brown", status="Questionable"),  # both carried forward
            _row("10/22/2025", "LA Clippers", "Kawhi Leonard"),  # new date and team
            _row("", "", "James Harden", status="Doubtful"),
        ]
        rows, stats = parse([page])
        assert [r["game_date"] for r in rows] == ["2025-10-21", "2025-10-21", "2025-10-22", "2025-10-22"]
        assert [r["team_abbreviation"] for r in rows] == ["BOS", "BOS", "LAC", "LAC"]
        assert stats["rows_kept"] == 4
        assert stats["rows_no_date"] == 0
        assert stats["rows_no_team"] == 0
        assert stats["has_date_column"] is True

    def test_date_carries_across_a_continuation_page(self, parse):
        first = [HEADER, _row("10/21/2025", "Boston Celtics", "Jayson Tatum")]
        # Continuation pages have 4 columns and no header, so no date column at all.
        cont = [["New York Knicks", "Jalen Brunson", "Out", "knee"]]
        rows, stats = parse([first, cont])
        assert [r["game_date"] for r in rows] == ["2025-10-21", "2025-10-21"]
        assert [r["team_abbreviation"] for r in rows] == ["BOS", "NYK"]
        assert stats["pages"] == 2

    def test_untracked_status_and_unknown_team_are_counted(self, parse):
        page = [
            HEADER,
            _row("10/21/2025", "Boston Celtics", "Available Guy", status="Available"),
            _row("10/21/2025", "Not A Team", "Ghost Player"),
        ]
        rows, stats = parse([page])
        assert rows == []
        assert stats["rows_untracked_status"] == 1
        assert stats["rows_no_team"] == 1
        assert stats["unknown_team_names"] == ["Not A Team"]

    def test_page_without_a_table_is_counted_not_silent(self, parse):
        rows, stats = parse([[HEADER, _row("10/21/2025", "Boston Celtics", "Jayson Tatum")], []])
        assert len(rows) == 1
        assert stats["pages_no_table"] == 1
