"""Reduce a game's play-by-play event stream to one row per possession.

Segmentation rule: a possession is a maximal run of consecutive *core* events
(field-goal attempts, non-technical free throws, turnovers, rebounds) by the
same team within a period. The team with the ball changes exactly when a core
event belongs to the other team, so and-ones, offensive rebounds, dead-ball
team rebounds after a missed first free throw, and loose-ball fouls all fall
out correctly without special cases. Technical free throws are excluded from
segmentation (they are not a possession outcome) and tallied separately so
the game total still reconciles against the box score.

Timing: a possession ends at its terminal event (the make, the last free
throw, the turnover) or at the opponent's defensive rebound; the next
possession starts at that same instant, so durations sum to the period length.

Lineups: PlayByPlayV3 does not state who starts each period, and its
Substitution rows carry the incoming player by last name only. Period
starters are inferred from first appearances (a player with any on-court
event before being subbed in was on the floor at the period start; bench
technical fouls are ignored as evidence). Incoming names are resolved against
the team's roster of ids seen anywhere in the game, excluding players already
on the floor. Whenever a team's on-floor set cannot be pinned to exactly five
players the possession's *_complete flag is 0 for the rest of that period --
the rate is reported per game and is the thing to improve (e.g. by joining
box-score starters) if it turns out to matter.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

import pandas as pd
from nba_api.stats.static import players as static_players

REG_PERIOD_SECS = 720.0
OT_PERIOD_SECS = 300.0
CORE_TYPES = {"Made Shot", "Missed Shot", "Free Throw", "Turnover", "Rebound"}
_CLOCK_RE = re.compile(r"PT(\d+)M([\d.]+)S")
_SUB_RE = re.compile(r"SUB:\s*(.+?)\s+FOR\s+(.+)$")
_FT_TRIP_RE = re.compile(r"(\d) of (\d)")
_SUFFIX_RE = re.compile(r"\s+(jr\.?|sr\.?|ii|iii|iv)$")
_PREFIX_RE = re.compile(r"^([a-z]+)\.\s+(.+)$")


_STATIC_FIRST_NAMES: dict[int, str] | None = None


def _static_first_names() -> dict[int, str]:
    """player id -> normalised first name, built once from the bundled list
    (find_player_by_id is a linear scan; per-player calls made parsing ~10x slower)."""
    global _STATIC_FIRST_NAMES
    if _STATIC_FIRST_NAMES is None:
        _STATIC_FIRST_NAMES = {int(p["id"]): _norm_name(p["first_name"]) for p in static_players.get_players()}
    return _STATIC_FIRST_NAMES


def _norm_name(name: str) -> str:
    """Lower-case ASCII form: 'Schröder' -> 'schroder', matching substitution text."""
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", name) if not unicodedata.combining(ch)
    ).strip().lower()


_UMLAUT_MAP = str.maketrans({"ö": "oe", "ü": "ue", "ä": "ae", "Ö": "oe", "Ü": "ue", "Ä": "ae"})


def _name_variants(name: str) -> set[str]:
    """Both ASCII conventions the feed mixes: 'Pöltl' appears as 'Poltl' in one
    column and 'Poeltl' in another."""
    return {_norm_name(name), _norm_name(name.translate(_UMLAUT_MAP))}


def parse_clock(clock: str) -> float:
    """'PT11M42.00S' -> 702.0 seconds remaining in the period."""
    m = _CLOCK_RE.match(clock or "")
    if not m:
        return 0.0
    return int(m.group(1)) * 60 + float(m.group(2))


def period_length(period: int) -> float:
    return REG_PERIOD_SECS if period <= 4 else OT_PERIOD_SECS


def period_start_secs(period: int) -> float:
    """Game seconds elapsed at the start of `period`."""
    if period <= 4:
        return (period - 1) * REG_PERIOD_SECS
    return 4 * REG_PERIOD_SECS + (period - 5) * OT_PERIOD_SECS


def secs_elapsed(period: int, clock: str) -> float:
    return period_start_secs(period) + period_length(period) - parse_clock(clock)


def _is_last_ft(sub_type: str) -> bool:
    m = _FT_TRIP_RE.search(sub_type or "")
    return bool(m) and m.group(1) == m.group(2)


@dataclass
class _Poss:
    period: int
    off_team: int
    start_secs: float
    margin_start: int
    lineup_off: tuple[int, ...]
    lineup_def: tuple[int, ...]
    lineup_off_complete: bool
    lineup_def_complete: bool
    end_secs: float = 0.0
    points: int = 0
    outcome: str = "period_end"
    n_fga: int = 0
    n_fg3a: int = 0
    n_fta: int = 0
    n_oreb: int = 0
    n_tov: int = 0
    n_fgm: int = 0
    n_fg3m: int = 0
    n_ftm: int = 0
    last_shot_value: int | None = None
    last_shot_distance: int | None = None
    last_shot_x: int | None = None
    last_shot_y: int | None = None
    last_shot_subtype: str | None = None
    last_core_secs: float = 0.0
    prev_core: str = ""  # action_type of the last core event, for oreb counting
    prev_core_last_ft: bool = False
    extra: dict = field(default_factory=dict)


class _LineupTracker:
    """Per-team on-floor sets, rebuilt at each period start."""

    def __init__(self, events: pd.DataFrame, team_ids: list[int]):
        self.team_ids = team_ids
        # team -> player id -> (exact name forms, suffix-stripped forms). The
        # substitution text uses 'James' when unique on the roster, 'L. James'
        # when a surname is shared, ASCII where the name column keeps accents
        # ('Schroder' vs 'Schröder'), and drops generational suffixes ('Butler'
        # for 'Butler III'). Exact forms are matched first so 'Jackson' does
        # not collide with a teammate stored as 'Jackson Jr.'.
        self.roster: dict[int, dict[int, tuple[set[str], set[str]]]] = {t: {} for t in team_ids}
        # First names from nba_api's bundled static player list (no API call):
        # with three teammates named Williams the text uses 'Jay. Williams' /
        # 'Jal. Williams', which only a first-name prefix can separate.
        self.first_name: dict[int, str] = {}
        played = events[(events.person_id > 0) & (events.team_id.isin(team_ids))]
        for t, pid, name, name_i in played[["team_id", "person_id", "player_name", "player_name_i"]].drop_duplicates().itertuples(index=False):
            exact, stripped = self.roster[int(t)].setdefault(int(pid), (set(), set()))
            for n in (name, name_i):
                if isinstance(n, str) and n.strip():
                    for v in _name_variants(n):
                        exact.add(v)
                        stripped.add(_SUFFIX_RE.sub("", v))
            if int(pid) not in self.first_name:
                self.first_name[int(pid)] = _static_first_names().get(int(pid), "")
        self.starters = self._infer_starters(events)
        self.on_floor: dict[int, set[int]] = {t: set() for t in team_ids}
        self.complete: dict[int, bool] = {t: False for t in team_ids}

    def _infer_starters(self, events: pd.DataFrame) -> dict[tuple[int, int], set[int]]:
        starters: dict[tuple[int, int], set[int]] = {}
        for period, grp in events.groupby("period", sort=True):
            for t in self.team_ids:
                seen_in: set[int] = set()
                on_at_start: set[int] = set()
                for r in grp[grp.team_id == t].itertuples(index=False):
                    pid = int(r.person_id)
                    if pid <= 0:
                        continue
                    if r.action_type == "Substitution":
                        if pid not in seen_in:
                            on_at_start.add(pid)  # subbed out before ever subbed in
                        in_pid = self._resolve_incoming(t, r.description, exclude=set())
                        if in_pid is not None:
                            seen_in.add(in_pid)
                        continue
                    if r.action_type == "Foul" and "Technical" in (r.sub_type or ""):
                        continue  # bench techs are not on-court evidence
                    if pid not in seen_in:
                        on_at_start.add(pid)
                starters[(int(period), t)] = on_at_start
        return starters

    def _resolve_incoming(self, team: int, description: str, exclude: set[int]) -> int | None:
        m = _SUB_RE.match(description or "")
        if not m:
            return None
        names = _name_variants(m.group(1))
        roster = self.roster[team]
        # 1) exact forms ('james', 'l. james'); 2) first-name prefix + surname
        # ('jay. williams'); 3) suffix-stripped surname ('butler' for 'Butler III').
        cands = [pid for pid, forms in roster.items() if names & forms[0] and pid not in exclude]
        if not cands:
            pm = _PREFIX_RE.match(next(iter(names)))
            if pm:
                prefix = pm.group(1)
                surnames = {n.split(". ", 1)[1] for n in names if ". " in n}
                cands = [
                    pid for pid, forms in roster.items()
                    if pid not in exclude and (surnames & forms[0] or surnames & forms[1])
                    and self.first_name.get(pid, "").startswith(prefix)
                ]
        if not cands:
            cands = [pid for pid, forms in roster.items() if names & forms[1] and pid not in exclude]
        return cands[0] if len(cands) == 1 else None

    def start_period(self, period: int) -> None:
        for t in self.team_ids:
            self.on_floor[t] = set(self.starters.get((period, t), set()))
            self.complete[t] = len(self.on_floor[t]) == 5

    def substitute(self, team: int, out_pid: int, description: str) -> None:
        if team not in self.on_floor:
            return
        if out_pid in self.on_floor[team]:
            self.on_floor[team].discard(out_pid)
        else:
            self.complete[team] = False
        in_pid = self._resolve_incoming(team, description, exclude=self.on_floor[team])
        if in_pid is None:
            self.complete[team] = False
        else:
            self.on_floor[team].add(in_pid)
        if len(self.on_floor[team]) != 5:
            self.complete[team] = False

    def snapshot(self, team: int) -> tuple[tuple[int, ...], bool]:
        return tuple(sorted(self.on_floor.get(team, ()))), bool(self.complete.get(team, False))


def _team_ids(events: pd.DataFrame) -> tuple[int | None, int | None]:
    home = events.loc[(events.location == "h") & (events.team_id > 0), "team_id"]
    away = events.loc[(events.location == "v") & (events.team_id > 0), "team_id"]
    return (int(home.iloc[0]) if len(home) else None, int(away.iloc[0]) if len(away) else None)


def build_possessions(
    events: pd.DataFrame,
    home_team_id: int | None = None,
    away_team_id: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Parse one game's events (pbp_events columns, any order) into possessions.

    Returns (possessions_df, summary) where summary holds per-team possession
    counts, points from possessions, technical-FT points and the home/away
    ids -- everything needed to reconcile against the box score.
    """
    ev = events.sort_values("action_id").reset_index(drop=True).copy()
    if "player_name_i" not in ev.columns:
        ev["player_name_i"] = ""
    for c in ("score_home", "score_away", "description", "sub_type", "action_type", "player_name", "player_name_i", "location"):
        ev[c] = ev[c].fillna("").astype(str)
    ev["team_id"] = ev["team_id"].fillna(0).astype(int)
    ev["person_id"] = ev["person_id"].fillna(0).astype(int)
    ev["period"] = ev["period"].astype(int)

    h, a = _team_ids(ev)
    home = home_team_id if home_team_id is not None else h
    away = away_team_id if away_team_id is not None else a
    if home is None or away is None:
        raise ValueError("could not determine home/away team ids from events")
    other = {home: away, away: home}

    # Team-level rows (shot-clock turnovers, team rebounds in some seasons) put
    # the team id in person_id and leave team_id at 0 -- normalise them so they
    # segment possessions like any other core event.
    team_row = (ev.team_id == 0) & ev.person_id.isin([home, away])
    ev.loc[team_row, "team_id"] = ev.loc[team_row, "person_id"]
    ev.loc[team_row, "person_id"] = 0
    game_id = str(ev["game_id"].iloc[0])

    lineups = _LineupTracker(ev, [home, away])
    score = {home: 0, away: 0}
    tech_pts = {home: 0, away: 0}
    poss: list[_Poss] = []
    cur: _Poss | None = None
    cur_period = 0
    period_end_secs = 0.0

    def close(end_secs: float) -> None:
        nonlocal cur
        if cur is None:
            return
        cur.end_secs = end_secs
        poss.append(cur)
        cur = None

    def open_new(team: int, period: int, start_secs: float) -> _Poss:
        lo, loc = lineups.snapshot(team)
        ld, ldc = lineups.snapshot(other[team])
        return _Poss(
            period=period, off_team=team, start_secs=start_secs,
            margin_start=score[team] - score[other[team]],
            lineup_off=lo, lineup_def=ld, lineup_off_complete=loc, lineup_def_complete=ldc,
            last_core_secs=start_secs,
        )

    def start_period(period: int) -> None:
        nonlocal cur_period, period_end_secs
        close(period_end_secs)
        cur_period = period
        period_end_secs = period_start_secs(period) + period_length(period)
        lineups.start_period(period)

    for r in ev.itertuples(index=False):
        period = int(r.period)
        if period != cur_period:
            start_period(period)
        atype = r.action_type
        secs = secs_elapsed(period, r.clock)

        if atype == "period":
            if r.sub_type == "end":
                close(period_end_secs)
            continue
        if atype == "Substitution":
            lineups.substitute(r.team_id, r.person_id, r.description)
            continue

        # Score tracking. Only scoring rows (made shots, free throws) update the
        # running score: Instant Replay rows also carry a score but it can be
        # stale or reflect a reversed basket, and trusting it broke
        # reconciliation on ~1.4% of games. Possession points are the nominal
        # value of the scoring event; the difference between the official
        # running score and the nominal tally (overturned baskets) is reported
        # per team as a correction so game totals still reconcile.
        new_score = None
        nominal = 0
        if atype in ("Made Shot", "Free Throw"):
            if r.score_home != "" and r.score_away != "":
                new_score = (int(r.score_home), int(r.score_away))
            nominal = int(r.shot_value) if atype == "Made Shot" else int(not r.description.startswith("MISS"))
        # The update is applied only after a possession opens, so margin_start is pre-event.

        if atype == "Free Throw" and "Technical" in r.sub_type:
            if r.team_id in tech_pts:
                tech_pts[r.team_id] += nominal
            if new_score:
                score[home], score[away] = new_score
            continue
        if atype not in CORE_TYPES or r.team_id not in other:
            continue
        pts = nominal

        team = r.team_id
        if cur is None or cur.off_team != team:
            # Team with the ball changed. A defensive rebound ends the prior
            # possession now; any other transition ended it at its own terminal event.
            prev_end = secs if atype == "Rebound" else (cur.last_core_secs if cur else period_start_secs(period))
            if cur is not None and atype != "Rebound" and cur.outcome == "period_end":
                # prior run had no terminal event (e.g. missed shot, no rebound row); end it here
                prev_end = secs
                cur.outcome = "miss" if cur.prev_core == "Missed Shot" else cur.outcome
            close(prev_end)
            cur = open_new(team, period, prev_end)
        if new_score:
            score[home], score[away] = new_score

        cur.last_core_secs = secs
        if atype == "Made Shot":
            cur.n_fga += 1
            cur.n_fg3a += int(r.shot_value == 3)
            cur.n_fgm += 1
            cur.n_fg3m += int(r.shot_value == 3)
            cur.points += pts
            cur.outcome = "made_fg"
            _record_shot(cur, r)
        elif atype == "Missed Shot":
            cur.n_fga += 1
            cur.n_fg3a += int(r.shot_value == 3)
            cur.outcome = "miss"
            _record_shot(cur, r)
        elif atype == "Free Throw":
            cur.n_fta += 1
            cur.n_ftm += pts
            cur.points += pts
            cur.outcome = "ft" if pts > 0 else "miss"
        elif atype == "Turnover":
            cur.n_tov += 1
            cur.outcome = "turnover"
        elif atype == "Rebound":
            # Only a live-ball rebound of a missed FG or final FT is a real offensive rebound.
            if cur.prev_core == "Missed Shot" or (cur.prev_core == "Free Throw" and cur.prev_core_last_ft):
                cur.n_oreb += 1
        cur.prev_core = atype
        cur.prev_core_last_ft = atype == "Free Throw" and _is_last_ft(r.sub_type)

    close(period_end_secs)

    rows = []
    for i, p in enumerate(poss):
        d = other[p.off_team]
        rows.append({
            "game_id": game_id, "poss_idx": i, "period": p.period,
            "off_team_id": p.off_team, "def_team_id": d, "is_home_off": int(p.off_team == home),
            "start_secs": p.start_secs, "end_secs": p.end_secs,
            "duration": round(p.end_secs - p.start_secs, 2),
            "margin_start": p.margin_start,
            "margin_end": p.margin_start + p.points,
            "points": p.points, "outcome": p.outcome,
            "n_fga": p.n_fga, "n_fg3a": p.n_fg3a, "n_fta": p.n_fta, "n_oreb": p.n_oreb, "n_tov": p.n_tov,
            "n_fgm": p.n_fgm, "n_fg3m": p.n_fg3m, "n_ftm": p.n_ftm,
            "last_shot_value": p.last_shot_value, "last_shot_distance": p.last_shot_distance,
            "last_shot_x": p.last_shot_x, "last_shot_y": p.last_shot_y, "last_shot_subtype": p.last_shot_subtype,
            "lineup_off": ",".join(map(str, p.lineup_off)), "lineup_def": ",".join(map(str, p.lineup_def)),
            "lineup_off_complete": int(p.lineup_off_complete), "lineup_def_complete": int(p.lineup_def_complete),
        })
    df = pd.DataFrame(rows)
    # margin_end must reflect opponent tech FTs too, but those are rare and
    # game-level; possession-level margin uses only possession points.
    pts_home_poss = int(df.loc[df.off_team_id == home, "points"].sum()) if len(df) else 0
    pts_away_poss = int(df.loc[df.off_team_id == away, "points"].sum()) if len(df) else 0
    summary = {
        "game_id": game_id, "home_team_id": home, "away_team_id": away,
        "n_poss_home": int((df.off_team_id == home).sum()) if len(df) else 0,
        "n_poss_away": int((df.off_team_id == away).sum()) if len(df) else 0,
        "pts_home_poss": pts_home_poss, "pts_away_poss": pts_away_poss,
        "tech_ft_pts_home": tech_pts[home], "tech_ft_pts_away": tech_pts[away],
        # Official running score minus nominal tally: nonzero only when a
        # basket was overturned on review (or the feed dropped a scoring row).
        "score_correction_home": score[home] - pts_home_poss - tech_pts[home],
        "score_correction_away": score[away] - pts_away_poss - tech_pts[away],
        "final_score_home": score[home], "final_score_away": score[away],
        "lineup_complete_rate": float((df.lineup_off_complete & df.lineup_def_complete).mean()) if len(df) else 0.0,
        "n_events": int(len(ev)),
    }
    return df, summary


def _record_shot(p: _Poss, r) -> None:
    p.last_shot_value = int(r.shot_value) if pd.notna(r.shot_value) else None
    p.last_shot_distance = int(r.shot_distance) if pd.notna(r.shot_distance) else None
    p.last_shot_x = int(r.x_legacy) if pd.notna(r.x_legacy) else None
    p.last_shot_y = int(r.y_legacy) if pd.notna(r.y_legacy) else None
    p.last_shot_subtype = r.sub_type or None
