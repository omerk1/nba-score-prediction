import datetime
import json

import pytest

from src.serving import winner_lines as WL
from src.serving.capture_winner import _tab_pattern
from src.serving.team_lookup import build_nickname_to_team_id

NICK = build_nickname_to_team_id()

# Page text as Winner renders it (copied from a real capture): bidi marks
# around the signs, a quarter-market section and a WNBA section that must
# be ignored, and a total row whose teams only appear in its title.
SPREAD_VIEW = """\
ארצות הברית - NBA
הימור יתרון - רבע ראשון
הימור יתרון - כולל הארכות אם יהיו
הימור יתרון - כולל הארכות אם יהיו
ארצות הברית — NBA
דטרויט פיסטונס (‪-‬1.5)
1.85
בוסטון סלטיקס (‪+‬1.5)
1.75
כדורסל,
בשעה
21:59
SD
7+
סקר
הימור יתרון - רבע ראשון
ארצות הברית — NBA
דטרויט פיסטונס (‪-‬0.5)
1.85
בוסטון סלטיקס (‪+‬0.5)
1.75
כדורסל,
בשעה
21:59
הימור יתרון - כולל הארכות אם יהיו
ארצות הברית — WNBA
גולדן סטייט ולקיריס (-2.5)
1.65
אטלנטה (+2.5)
1.65
בשעה
02:59
"""
TOTAL_VIEW = """\
מעל/מתחת נקודות - כולל הארכות אם יהיו
ארצות הברית — NBA
מעל 222.5
1.80
מתחת 222.5
1.70
כדורסל,
בשעה
21:59  דטרויט פיסטונס - בוסטון סלטיקס 
SD
7+
"""


def _snapshot(tmp_path, *texts):
    path = tmp_path / "snap.json"
    path.write_text(json.dumps({"views": [{"text": t} for t in texts]}, ensure_ascii=False))
    return path


def test_spread_and_total_merge_into_one_pick(tmp_path):
    (pick,) = WL.picks_from_snapshot(_snapshot(tmp_path, SPREAD_VIEW, TOTAL_VIEW))
    assert pick == {
        "home_team_id": NICK["Pistons"],
        "away_team_id": NICK["Celtics"],
        "home_spread": -1.5,
        "home_spread_odds": 1.85,
        "away_spread_odds": 1.75,
        "total_line": 222.5,
        "over_odds": 1.8,
        "under_odds": 1.7,
    }


def test_quarter_market_and_wnba_rows_are_ignored():
    games = WL.parse_view(SPREAD_VIEW, NICK)
    assert len(games) == 1
    assert next(iter(games.values()))["home_spread"] == -1.5


def test_total_only_game_has_no_spread(tmp_path):
    (pick,) = WL.picks_from_snapshot(_snapshot(tmp_path, TOTAL_VIEW))
    assert "home_spread" not in pick and pick["total_line"] == 222.5


def test_unknown_nba_team_name_fails_loudly():
    text = TOTAL_VIEW.replace("בוסטון סלטיקס", "בוסטון משהו")
    with pytest.raises(WL.UnknownTeamError, match="בוסטון משהו"):
        WL.parse_view(text, NICK)


def test_every_table_nickname_is_a_real_team():
    assert set(WL.HEBREW_TEAM_NICKNAMES.values()) == set(NICK)


@pytest.mark.parametrize(
    "label,day,expected",
    [("ג 20.10", datetime.date(2026, 10, 20), True), ("ב 05.10", datetime.date(2026, 10, 5), True),
     ("ב 5.10", datetime.date(2026, 10, 5), True), ("ד 21.10", datetime.date(2026, 10, 20), False),
     ("18.10 20:14", datetime.date(2026, 10, 18), False)],
)
def test_tab_pattern(label, day, expected):
    assert bool(_tab_pattern(day).search(label)) is expected
