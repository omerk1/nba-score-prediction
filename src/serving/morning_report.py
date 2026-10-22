"""
Morning performance report: grades the logged predictions against final
scores (grade_predictions) and renders one HTML page — last night's
results, then everything so far in plain terms. Betting performance covers strong bets only; the question the report answers is
whether the strong bets (edge >= STRONG_EDGE) pay off. Marginal bets are
shown on the daily slate but deliberately not tracked here.
"""

import datetime
import html

import pandas as pd

from src.serving.grade_predictions import LOG_PATH, grade, load_results
from src.serving.recommended_bets import STRONG_EDGE
from src.serving.slate_html import _CSS
from src.serving.team_lookup import build_team_id_to_name

_ICON = {"win": "✅", "loss": "❌", "push": "➖"}


def _record(bets: pd.DataFrame) -> tuple[str, str]:
    """('7-4-1', '+3.2% · +0.4u'): wins-losses(-pushes) and flat-stake return."""
    if bets.empty:
        return "—", "no bets"
    w, l, p = (int((bets["outcome"] == o).sum()) for o in ("win", "loss", "push"))
    rec = f"{w}-{l}" + (f"-{p}" if p else "")
    return rec, f"{bets['profit'].mean():+.1%} · {bets['profit'].sum():+.1f}u"


def build_report(log: pd.DataFrame, results: pd.DataFrame) -> tuple[str, str] | None:
    """(caption, html page), or None when nothing is graded yet."""
    bets = grade(log, results)
    if bets.empty:
        return None
    names = build_team_id_to_name()
    games = bets.drop_duplicates(["game_date", "home_team_id", "away_team_id"])
    day = games["game_date"].max()
    recommended = bets[bets["edge"] >= STRONG_EDGE]

    def team(i):
        return html.escape(str(names.get(i, i)))

    night = []
    for g in games[games["game_date"] == day].itertuples():
        night.append(
            f"<tr><td>{team(g.away_team_id)} @ {team(g.home_team_id)}</td>"
            f'<td class="n">{g.pred_diff:+.1f} → {g.actual_diff:+.0f}</td>'
            f'<td class="n">{g.pred_total:.0f} → {g.actual_total:.0f}</td></tr>'
        )
    night_bets = []
    for b in recommended[recommended["game_date"] == day].sort_values("edge", ascending=False).itertuples():
        side = {"home": team(b.home_team_id), "away": team(b.away_team_id)}.get(b.side, b.side.title())
        night_bets.append(
            f"<tr><td>{_ICON[b.outcome]} {side} <span class='mute'>({b.market}, "
            f"{team(b.away_team_id)} @ {team(b.home_team_id)})</span></td>"
            f'<td class="n">{b.odds:.2f}</td><td class="n">{b.edge * 100:+.1f}%</td></tr>'
        )
    last_rec, last_ret = _record(recommended[recommended["game_date"] == day])

    all_rec, all_ret = _record(recommended)

    mae_diff, mae_total = games["pred_diff_error"].abs().mean(), games["pred_total_error"].abs().mean()
    page = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>Morning report {day:%b %d}</title><style>{_CSS}</style></head><body><main>"
        f"<h1>Morning report · games of {day:%a %b %d}</h1>"
        f'<h2>Last night\'s strong bets (edge ≥ 5%) — {last_rec} ({last_ret})</h2>'
        f'<div class="card"><table><tr><th>Bet</th><th class="n">Odds</th><th class="n">Edge</th></tr>'
        f'{"".join(night_bets) or "<tr><td>No strong bets</td><td></td><td></td></tr>"}</table></div>'
        '<h2>Last night\'s games — model vs. reality</h2>'
        '<div class="card"><table><tr><th>Game (away @ home)</th><th class="n">Home margin<br>predicted → actual</th><th class="n">Total points<br>predicted → actual</th></tr>'
        f'{"".join(night)}</table><p class="mute">Home margin = home score minus away score, so + means the home team won. The gap between the two numbers is the model\'s miss.</p></div>'
        f"<h2>Everything so far · {len(games)} games</h2>"
        '<div class="card"><table>'
        f'<tr><td>Typical miss on the winning margin</td><td class="n">{mae_diff:.1f} pts</td></tr>'
        f'<tr><td>Typical miss on total points</td><td class="n">{mae_total:.1f} pts</td></tr>'
        f'<tr><td>All strong bets</td><td class="n">{all_rec} · {all_ret}</td></tr></table>'
        '<p class="mute">A record is wins-losses(-pushes). Return = profit from staking 1 unit (u) on every bet, '
        "as % per bet and total units.</p></div>"
        "</main></body></html>"
    )
    caption = (
        f"<b>Morning report · {day:%a %b %d}</b>\n"
        f"Last night's strong bets: {last_rec} ({last_ret})\n"
        f"All {len(games)} games so far: strong bets {all_rec} ({all_ret}); margin miss {mae_diff:.1f} pts"
    )
    return caption, page


def run(out_dir="outputs/slates") -> tuple[str, str] | str:
    """(caption, path of the written page), or a plain message when there is
    nothing to report."""
    from pathlib import Path

    if not LOG_PATH.exists():
        return "Morning report: no predictions log yet."
    built = build_report(pd.read_csv(LOG_PATH), load_results())
    if built is None:
        return "Morning report: no logged predictions have final scores yet."
    caption, page = built
    path = Path(out_dir) / f"morning_{datetime.date.today().isoformat()}.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page, encoding="utf-8")
    return caption, str(path)
