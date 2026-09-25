"""
Canonical NBA team-nickname <-> team_id lookup. Extracted from
scripts/market_benchmark.py (which used it to join Polymarket's team names
onto nba_api team IDs) so src/serving/extract_picks.py can reuse the exact
same mapping for screenshot-extracted team names instead of duplicating it.
"""

from nba_api.stats.static import teams as nba_teams


def build_nickname_to_team_id() -> dict:
    return {t["nickname"]: t["id"] for t in nba_teams.get_teams()}
