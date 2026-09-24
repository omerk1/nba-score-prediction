"""
Build the labeled availability dataset: one row per (report_date, team, player)
injury listing with an uncertain status, joined to what actually happened.

Report date vs. game date
-------------------------
`player_injuries.game_date` is the DATE OF THE PDF REPORT, not the game. The
scraper keeps the latest report of each day (11PM ET) and the parser discards
the PDF's own "Game Date" column, so a report stored under date D holds rows
for games on D (already tipped off, in practice only Out players remain) and
rows for games on D+1. Measured on the full table: 99% of Questionable/Doubtful
listings have their team's next game exactly one day after the report date;
Out listings are a 62/38 mix of same-day and next-day.

Rule used here: an uncertain listing dated D refers to the team's game on D+1.
If the team has no game on D+1 the label is undefined and the row is dropped
(and counted). Out listings, used only as history, map to D+1 when the team
plays on D+1 and to D otherwise.

Label rules
- `played`  = 1 if the player has a box-score row with minutes > 0 for that
              game, else 0 (only players who played appear in the game log).
- `minutes` = box-score minutes (0 when they did not play).
- Dropped and counted: G League / two-way listings (not an injury; the formula
  scorer skips them for the same reason) and names that cannot be resolved to
  any player in that team's game log that season (never guessed).

Labels are used for evaluation only. Nothing here may be read by a prompt or
by the feature pipeline for the game being predicted.
"""

import logging
import sqlite3
from dataclasses import dataclass

import pandas as pd

from src.availability.db import DEFAULT_DB_PATH
from src.availability.names import normalize_name, squash
from src.news_scraping.extractors.formula_scorer import is_gleague

logger = logging.getLogger(__name__)

UNCERTAIN_STATUSES = ("Questionable", "Doubtful")
ALL_STATUSES = ("Out", "Questionable", "Doubtful")


@dataclass
class JoinReport:
    n_listings: int
    n_gleague: int
    n_no_team_game: int
    n_unresolved: int
    n_labeled: int

    @property
    def resolved_share(self) -> float:
        """Share of listings with a defined label that resolved to a player."""
        eligible = self.n_listings - self.n_gleague - self.n_no_team_game
        return self.n_labeled / eligible if eligible else 0.0


def season_of(date_str: str) -> str:
    y, m = int(date_str[:4]), int(date_str[5:7])
    start = y if m >= 10 else y - 1
    return f"{start}-{str(start + 1)[2:]}"


def load_listings(injury_db: str, statuses=None) -> pd.DataFrame:
    with sqlite3.connect(injury_db) as conn:
        df = pd.read_sql_query(
            "SELECT game_date AS report_date, team_id, player_name, status, reason FROM player_injuries",
            conn,
        )
    if statuses is not None:
        df = df[df["status"].isin(statuses)]
    df = df.copy()
    df["name_key"] = df["player_name"].map(normalize_name)
    return df


def load_game_log(availability_db: str = DEFAULT_DB_PATH) -> pd.DataFrame:
    with sqlite3.connect(availability_db) as conn:
        return pd.read_sql_query(
            "SELECT player_id, player_name, team_id, game_id, game_date, season, season_type, minutes, pts "
            "FROM player_game_log",
            conn,
        )


def _team_games(game_log: pd.DataFrame) -> pd.DataFrame:
    return game_log[["team_id", "game_date", "game_id", "season"]].drop_duplicates(["team_id", "game_date"])


def _league_keys(game_log: pd.DataFrame) -> pd.DataFrame:
    """(season, name_key) -> player_id over the WHOLE league, not one team.

    Resolving within a team's own game log drops two legitimate groups: players
    who missed a full season through injury (Ben Simmons in 2021-22 never
    appeared for Philadelphia, so he could not be found there) and players
    traded mid-season. Dropping the first group biases the labels toward players
    who do play, which is exactly the quantity being estimated. League-wide
    resolution finds both; the label join is on (player_id, game_id), which is
    team-agnostic and answers the right question -- did this player log minutes
    in this particular game.
    """
    r = game_log[["season", "player_id", "player_name"]].drop_duplicates()
    r = r.assign(name_key=r["player_name"].map(normalize_name))
    r = r.assign(squashed=r["name_key"].map(squash))
    # A normalized key shared by two players in one season is ambiguous; drop it
    # rather than guess.
    exact = r.drop_duplicates(["season", "name_key"], keep=False)[["season", "name_key", "player_id"]]
    sq = r.drop_duplicates(["season", "squashed"], keep=False)[["season", "squashed", "player_id"]]
    return exact, sq


def _resolve_players_league(listings: pd.DataFrame, game_log: pd.DataFrame) -> pd.DataFrame:
    exact, sq = _league_keys(game_log)
    out = listings.merge(exact, on=["season", "name_key"], how="left")
    miss = out["player_id"].isna()
    if miss.any():
        fb = out.loc[miss].assign(squashed=out.loc[miss, "name_key"].map(squash))
        fb = fb.merge(sq.rename(columns={"player_id": "pid2"}), on=["season", "squashed"], how="left")
        out.loc[miss, "player_id"] = fb["pid2"].values
    return out


def _roster_keys(game_log: pd.DataFrame) -> pd.DataFrame:
    """(team_id, season, name_key) -> player_id, from players who appear in that team's log."""
    r = game_log[["team_id", "season", "player_id", "player_name"]].drop_duplicates()
    r = r.assign(name_key=r["player_name"].map(normalize_name))
    return r.assign(squashed=r["name_key"].map(squash))


def _attach_game(listings: pd.DataFrame, team_games: pd.DataFrame, allow_same_day: bool) -> pd.DataFrame:
    """Map report_date -> the game the listing refers to (see module docstring)."""
    next_day = (pd.to_datetime(listings["report_date"]) + pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d")
    out = listings.assign(game_date=next_day).merge(team_games, on=["team_id", "game_date"], how="left")
    if allow_same_day:
        miss = out["game_id"].isna()
        same = listings.loc[miss.values].assign(game_date=listings.loc[miss.values, "report_date"])
        same = same.merge(team_games, on=["team_id", "game_date"], how="left")
        out.loc[miss, ["game_date", "game_id", "season"]] = same[["game_date", "game_id", "season"]].values
    return out


def _resolve_players(listings: pd.DataFrame, roster: pd.DataFrame) -> pd.DataFrame:
    exact = roster[["team_id", "season", "name_key", "player_id"]].drop_duplicates(
        ["team_id", "season", "name_key"]
    )
    out = listings.merge(exact, on=["team_id", "season", "name_key"], how="left")
    # Fallback for PDF rows whose tokens were glued together by the extractor.
    miss = out["player_id"].isna()
    if miss.any():
        sq = roster[["team_id", "season", "squashed", "player_id"]].drop_duplicates(
            ["team_id", "season", "squashed"]
        )
        sq = sq.rename(columns={"player_id": "player_id_sq"})
        fb = out.loc[miss].assign(squashed=out.loc[miss, "name_key"].map(squash))
        fb = fb.merge(sq, on=["team_id", "season", "squashed"], how="left")
        out.loc[miss, "player_id"] = fb["player_id_sq"].values
    return out


def build_labels(
    injury_db: str,
    availability_db: str = DEFAULT_DB_PATH,
    statuses=UNCERTAIN_STATUSES,
) -> tuple[pd.DataFrame, JoinReport]:
    listings = load_listings(injury_db, statuses)
    game_log = load_game_log(availability_db)
    n_listings = len(listings)

    gl_mask = listings["reason"].fillna("").map(is_gleague)
    n_gleague = int(gl_mask.sum())
    listings = listings[~gl_mask].reset_index(drop=True)

    allow_same_day = "Out" in statuses
    listings = _attach_game(listings, _team_games(game_log), allow_same_day)
    no_game = listings["game_id"].isna()
    n_no_team_game = int(no_game.sum())
    listings = listings[~no_game].reset_index(drop=True)

    listings = _resolve_players(listings, _roster_keys(game_log))
    unresolved = listings["player_id"].isna()
    n_unresolved = int(unresolved.sum())
    if n_unresolved:
        sample = listings.loc[unresolved, "player_name"].value_counts().head(10)
        logger.info(f"unresolved names (top): {sample.to_dict()}")
    listings = listings[~unresolved].copy()
    listings["player_id"] = listings["player_id"].astype(int)

    minutes = game_log[["player_id", "game_id", "minutes"]]
    labeled = listings.merge(minutes, on=["player_id", "game_id"], how="left")
    labeled["minutes"] = labeled["minutes"].fillna(0.0)
    labeled["played"] = (labeled["minutes"] > 0).astype(int)

    report = JoinReport(n_listings, n_gleague, n_no_team_game, n_unresolved, len(labeled))
    cols = [
        "report_date",
        "game_date",
        "season",
        "team_id",
        "game_id",
        "player_id",
        "player_name",
        "status",
        "reason",
        "minutes",
        "played",
    ]
    labeled = labeled[cols].sort_values(["game_date", "team_id", "player_name"]).reset_index(drop=True)
    return labeled, report


DATED_DB_PATH = "data/raw/injury_dates.sqlite"


def build_labels_dated(
    dated_db: str = DATED_DB_PATH,
    availability_db: str = DEFAULT_DB_PATH,
    statuses=UNCERTAIN_STATUSES,
) -> tuple[pd.DataFrame, JoinReport]:
    """Labels from the re-parsed reports, which carry each row's real game date
    (scripts/rebuild_injury_dates.py). Preferred over `build_labels`: no D+1
    approximation, 2,287 more uncertain listings, and the rows that the old
    parser dropped silently -- every Clippers listing among them.

    `report_date` is kept because retrieval still bounds on it: a listing is
    knowable from the moment its report is published, not from the game date.
    """
    with sqlite3.connect(dated_db) as conn:
        listings = pd.read_sql_query(
            "SELECT game_date, report_date, team_id, player_name, status, reason, date_source "
            "FROM player_injuries_dated",
            conn,
        )
    n_listings_all = len(listings)
    if statuses is not None:
        listings = listings[listings["status"].isin(statuses)]
    n_listings = len(listings)

    gl_mask = listings["reason"].fillna("").map(is_gleague)
    n_gleague = int(gl_mask.sum())
    listings = listings[~gl_mask].reset_index(drop=True)
    listings["season"] = listings["game_date"].map(season_of)
    listings["name_key"] = listings["player_name"].map(normalize_name)

    game_log = load_game_log(availability_db)
    listings = listings.merge(
        _team_games(game_log), on=["team_id", "game_date"], how="left", suffixes=("", "_log")
    )
    no_game = listings["game_id"].isna()
    n_no_team_game = int(no_game.sum())
    listings = listings[~no_game].reset_index(drop=True)

    listings = _resolve_players_league(listings, game_log)
    unresolved = listings["player_id"].isna()
    n_unresolved = int(unresolved.sum())
    if n_unresolved:
        logger.info(
            f"unresolved names (top): {listings.loc[unresolved, 'player_name'].value_counts().head(10).to_dict()}"
        )
    # A name that resolves nowhere in the league that season belongs to someone
    # who never logged a minute, so the label is 0 and only the history features
    # are unavailable. Keeping the row avoids the survivorship bias that
    # team-scoped resolution introduced.
    listings["player_id"] = listings["player_id"].astype("Int64")

    labeled = listings.merge(
        game_log[["player_id", "game_id", "minutes"]], on=["player_id", "game_id"], how="left"
    )
    labeled["minutes"] = labeled["minutes"].fillna(0.0)
    labeled["played"] = (labeled["minutes"] > 0).astype(int)

    report = JoinReport(n_listings, n_gleague, n_no_team_game, n_unresolved, len(labeled))
    logger.info(f"dated labels: {n_listings_all} total rows, {n_listings} with status in {statuses}")
    cols = [
        "report_date",
        "game_date",
        "season",
        "team_id",
        "game_id",
        "player_id",
        "player_name",
        "status",
        "reason",
        "date_source",
        "minutes",
        "played",
    ]
    return labeled[cols].sort_values(["game_date", "team_id", "player_name"]).reset_index(drop=True), report


def build_history_dated(
    dated_db: str = DATED_DB_PATH, availability_db: str = DEFAULT_DB_PATH
) -> pd.DataFrame:
    """All listings including Out, for retrieval over a player's prior listings."""
    df, _ = build_labels_dated(dated_db, availability_db, statuses=ALL_STATUSES)
    return df


def build_history(injury_db: str, availability_db: str = DEFAULT_DB_PATH) -> pd.DataFrame:
    """All listings (Out included) with outcomes, for retrieval of a player's prior
    listings. Same join rules as build_labels; unresolved/no-game rows dropped."""
    df, _ = build_labels(injury_db, availability_db, statuses=ALL_STATUSES)
    return df
