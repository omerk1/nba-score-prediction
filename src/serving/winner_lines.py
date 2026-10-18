"""
Parses the text of Winner's NBA lines page (captured by
capture_winner.capture_nba_lines) into picks. The page text is exact —
team names, lines and odds are printed values — so unlike a screenshot
read by a vision model there is nothing to misread or invent.

Only two markets are read, both "including overtime": the point spread
(the main list) and over/under total points (a market-filter chip on the
same list). Each game's rows carry a tip-off time; spread and total rows
for the same game are joined on (time, teams).

Home/away: the first team printed in a spread row (the right-hand box)
and the first team in a total row's "A - B" title are the home team —
cross-checked against the NBA schedule downstream, which has the final
say, so a layout change can't silently flip the home-court term.
"""

import json
import re
from pathlib import Path

from src.serving.team_lookup import build_nickname_to_team_id

SPREAD_MARKET = "הימור יתרון - כולל הארכות אם יהיו"
TOTAL_MARKET = "מעל/מתחת נקודות - כולל הארכות אם יהיו"
NBA_LEAGUE_LINE = "ארצות הברית — NBA"
LEAGUE_PREFIX = "ארצות הברית — "

# Winner's Hebrew display names -> canonical NBA nicknames.
HEBREW_TEAM_NICKNAMES = {
    "אטלנטה הוקס": "Hawks",
    "בוסטון סלטיקס": "Celtics",
    "ברוקלין נטס": "Nets",
    "שארלוט הורנטס": "Hornets",
    "שיקגו בולס": "Bulls",
    "קליבלנד קאבלירס": "Cavaliers",
    "קליבלנד קבלירס": "Cavaliers",
    "דאלאס מאבריקס": "Mavericks",
    "דנבר נאגטס": "Nuggets",
    "דטרויט פיסטונס": "Pistons",
    "גולדן סטייט ווריורס": "Warriors",
    "יוסטון רוקטס": "Rockets",
    "אינדיאנה פייסרס": "Pacers",
    "לוס אנג'לס קליפרס": "Clippers",
    "לוס אנג'לס לייקרס": "Lakers",
    "ממפיס גריזליס": "Grizzlies",
    "מיאמי היט": "Heat",
    "מילווקי באקס": "Bucks",
    "מינסוטה טימברוולבס": "Timberwolves",
    "ניו אורלינס פליקאנס": "Pelicans",
    "ניו יורק ניקס": "Knicks",
    "אוקלהומה סיטי ת'אנדר": "Thunder",
    "אורלנדו מג'יק": "Magic",
    "פילדלפיה 76": "76ers",
    "פיניקס סאנס": "Suns",
    "פורטלנד טרייל בלייזרס": "Trail Blazers",
    "סקרמנטו קינגס": "Kings",
    "סאן אנטוניו ספרס": "Spurs",
    "טורונטו ראפטורס": "Raptors",
    "יוטה ג'אז": "Jazz",
    "וושינגטון וויזארדס": "Wizards",
}

_BIDI_MARKS = re.compile("[‎‏‪-‮]")
_SPREAD_LINE = re.compile(r"^(.+?) \(([+-]\d+(?:\.\d+)?)\)$")
_ODDS = re.compile(r"^\d+\.\d+$")
_OVER = re.compile(r"^מעל (\d+(?:\.\d+)?)$")
_UNDER = re.compile(r"^מתחת (\d+(?:\.\d+)?)$")
_TIME = re.compile(r"^\d{2}:\d{2}$")
_TIME_TITLE = re.compile(r"^(\d{2}:\d{2})\s+(.+?) - (.+)$")
# A row's time/title sits a few lines after its odds (sport label, "בשעה").
_LOOKAHEAD = 6


class UnknownTeamError(ValueError):
    pass


def _clean(text: str) -> list[str]:
    text = _BIDI_MARKS.sub("", text).replace("\xa0", " ").replace("’", "'")
    return [s for s in (line.strip() for line in text.splitlines()) if s]


def _team_id(name: str, nick2id: dict) -> int:
    nick = HEBREW_TEAM_NICKNAMES.get(name.strip())
    if nick is None or nick not in nick2id:
        raise UnknownTeamError(f"unrecognized Winner team name: {name!r}")
    return nick2id[nick]


def _find(lines: list[str], start: int, pattern: re.Pattern):
    for line in lines[start : start + _LOOKAHEAD]:
        m = pattern.match(line)
        if m:
            return m
    return None


def parse_view(text: str, nick2id: dict | None = None) -> dict:
    """One captured page view -> {(time, frozenset(team_ids)): partial pick}.
    Rows are only read under an NBA spread/total section header, so other
    leagues, quarter/half markets and the WNBA are ignored. Raises
    UnknownTeamError for an NBA row whose team name isn't in the table —
    a silent skip would drop a real game."""
    nick2id = nick2id or build_nickname_to_team_id()
    lines = _clean(text)
    games: dict = {}
    market = None
    for i, line in enumerate(lines):
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        if nxt.startswith(LEAGUE_PREFIX):
            market = line if nxt == NBA_LEAGUE_LINE else None
            continue
        if market == SPREAD_MARKET:
            a = _SPREAD_LINE.match(line)
            if not a or i + 3 >= len(lines):
                continue
            b = _SPREAD_LINE.match(lines[i + 2])
            if not (b and _ODDS.match(nxt) and _ODDS.match(lines[i + 3])):
                continue
            t = _find(lines, i + 4, _TIME)
            if not t:
                continue
            home, away = _team_id(a.group(1), nick2id), _team_id(b.group(1), nick2id)
            games.setdefault((t.group(0), frozenset((home, away))), {}).update(
                home_team_id=home,
                away_team_id=away,
                home_spread=float(a.group(2)),
                home_spread_odds=float(nxt),
                away_spread_odds=float(lines[i + 3]),
            )
        elif market == TOTAL_MARKET:
            over = _OVER.match(line)
            if not over or i + 3 >= len(lines):
                continue
            under = _UNDER.match(lines[i + 2])
            if not (under and _ODDS.match(nxt) and _ODDS.match(lines[i + 3])):
                continue
            t = _find(lines, i + 4, _TIME_TITLE)
            if not t:
                continue
            home, away = _team_id(t.group(2), nick2id), _team_id(t.group(3), nick2id)
            game = games.setdefault((t.group(1), frozenset((home, away))), {})
            # The spread's sign is tied to its own row's home/away order,
            # so a spread row's team ids take precedence over the title's.
            game.setdefault("home_team_id", home)
            game.setdefault("away_team_id", away)
            game.update(
                total_line=float(over.group(1)),
                over_odds=float(nxt),
                under_odds=float(lines[i + 3]),
            )
    return games


def picks_from_snapshot(path: str | Path) -> list[dict]:
    """The capture's JSON snapshot -> one pick per game, spread and total
    merged across the captured views. The pick's team ids come from
    whichever row read them first; the schedule gate re-orients home/away."""
    snapshot = json.loads(Path(path).read_text())
    nick2id = build_nickname_to_team_id()
    merged: dict = {}
    for view in snapshot["views"]:
        for key, fields in parse_view(view["text"], nick2id).items():
            game = merged.setdefault(key, {})
            if "home_spread" in game and "home_spread" not in fields:
                fields = {k: v for k, v in fields.items() if k not in ("home_team_id", "away_team_id")}
            game.update(fields)
    return list(merged.values())
