"""
Self-contained HTML rendering of a day's recommend_game() results — the
phone-friendly full analysis behind the short Telegram summary. One file,
inline CSS, no external requests, light/dark aware.
"""

import html

from src.serving.recommended_bets import SIDES, STRONG_EDGE, pick_label, recommended_bets

_CSS = """
:root{--bg:#f6f7f9;--card:#fff;--ink:#14171c;--mute:#667085;--line:#e4e7ec;--good:#067647;--goodbg:#e7f6ee}
@media (prefers-color-scheme:dark){:root{--bg:#0f1115;--card:#181b21;--ink:#eceff4;--mute:#98a2b3;--line:#2a2f38;--good:#4ade80;--goodbg:#12301f}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 -apple-system,system-ui,sans-serif}
main{max-width:720px;margin:0 auto;padding:16px}h1{font-size:20px;margin:4px 0}h2{font-size:15px;margin:22px 0 8px;color:var(--mute);text-transform:uppercase;letter-spacing:.04em}
.sub{color:var(--mute);margin:0 0 8px}.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px 14px;margin:8px 0}
.bet{display:flex;justify-content:space-between;gap:12px;align-items:baseline}.bet b{font-size:16px}.edge{color:var(--good);background:var(--goodbg);border-radius:999px;padding:1px 9px;font-weight:600;white-space:nowrap}
.weak .edge{color:var(--mute);background:var(--line)}.mute{color:var(--mute);font-size:13px}table{width:100%;border-collapse:collapse;font-size:14px;margin-top:6px}th,td{text-align:left;padding:5px 4px;border-top:1px solid var(--line)}th{color:var(--mute);font-weight:500}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}tr.rec td{color:var(--good);font-weight:600}summary{cursor:pointer;font-weight:600}
"""


def _e(x) -> str:
    return html.escape(str(x))


def _pct(p: float) -> str:
    return f"{p:.0%}"


_SHORT = {"Out": "out", "Doubtful": "doubtful", "Questionable": "questionable", "Day-To-Day": "day-to-day"}


def _team_injuries(players: list[dict]) -> str:
    parts = []
    for p in players:
        text = _e(p["player_name"])
        extra = [_SHORT[p["status"]]] if p["status"] != "Out" else []
        parts.append(text + (f" ({_e(', '.join(extra))})" if extra else ""))
    return ", ".join(parts) or "none listed"


def _injury_line(injuries, ids_names: list[tuple[int, str]], out_only: bool) -> str:
    """Per-team list of absent players. `injuries` None = no snapshot today,
    said plainly rather than shown as 'none listed'."""
    if injuries is None:
        return '<div class="mute">Injury data unavailable today.</div>'
    chunks = []
    for team_id, name in ids_names:
        players = injuries.get(team_id, [])
        if out_only:
            players = [p for p in players if p["status"] == "Out"]
        chunks.append(f"<b>{_e(name)}</b>: {_team_injuries(players)}")
    label = "Out" if out_only else "Injury report"
    return f'<div class="mute">{label} — ' + " · ".join(chunks) + "</div>"


def _bet_card(b: dict, injuries) -> str:
    weak = b["edge"] < STRONG_EDGE
    odds = f" @{b['odds']:.2f}" if b["odds"] is not None else ""
    market = (
        f"break-even {_pct(b['market_prob'])}" if b["market_prob"] is not None else "no price"
    )
    return (
        f'<div class="card{" weak" if weak else ""}"><div class="bet"><b>{_e(b["pick"])}{_e(odds)}</b>'
        f'<span class="edge">edge {b["edge"] * 100:+.1f}%</span></div>'
        f'<div class="mute">{_e(b["game"])} · {_e(b["market"])} · model {_pct(b["model_prob"])} · {market}</div>'
        + _injury_line(injuries, [(b["away_team_id"], b["away"]), (b["home_team_id"], b["home"])], out_only=True)
        + "</div>"
    )


def _market_rows(rec: dict, home: str, away: str, min_edge: float) -> str:
    rows = []
    for market, side, p_key, odds_key, mkt_key, edge_key in SIDES:
        if edge_key not in rec:
            continue
        edge = rec[edge_key]
        odds = rec.get(odds_key)
        cls = ' class="rec"' if edge > min_edge else ""
        rows.append(
            f"<tr{cls}><td>{_e(pick_label(market, side, rec, home, away))}</td>"
            f'<td class="n">{"" if odds is None else f"{odds:.2f}"}</td>'
            f'<td class="n">{_pct(rec[p_key])}</td><td class="n">{_pct(rec[mkt_key])}</td>'
            f'<td class="n">{edge * 100:+.1f}%</td></tr>'.replace("-", "−", 0)
        )
    if not rows:
        return ""
    return (
        '<table><tr><th>Side</th><th class="n">Odds</th><th class="n">Model</th>'
        '<th class="n">Break-even</th><th class="n">Edge</th></tr>' + "".join(rows) + "</table>"
    )


def _game_card(rec: dict, team_names: dict, min_edge: float, injuries) -> str:
    home = str(team_names.get(rec["home_team_id"], rec["home_team_id"]))
    away = str(team_names.get(rec["away_team_id"], rec["away_team_id"]))
    diff = rec["predicted_diff"]
    leader = f"{home} by {diff:.1f}" if diff >= 0 else f"{away} by {-diff:.1f}"
    win = rec["home_win_probability"]
    return (
        f'<details class="card"><summary>{_e(away)} @ {_e(home)}</summary>'
        f'<p class="sub">Model: {_e(home)} {rec["predicted_home_score"]} – {rec["predicted_away_score"]} {_e(away)}'
        f" · {_e(leader)} · total {rec['predicted_total']:.1f}<br>"
        f"Win chance: {_e(home)} {_pct(win)} · {_e(away)} {_pct(1 - win)}</p>"
        f"{_injury_line(injuries, [(rec['away_team_id'], away), (rec['home_team_id'], home)], out_only=False)}"
        f"{_market_rows(rec, home, away, min_edge)}</details>"
    )


def render_slate_html(
    recs: list[dict], team_names: dict, run_label: str, min_edge: float, injuries=None
) -> str:
    """`injuries`: injury_lookup.injuries_for_date result; None = unavailable."""
    bets = sorted(
        (b for rec in recs for b in recommended_bets(rec, min_edge, team_names)),
        key=lambda b: b["edge"],
        reverse=True,
    )
    strong = [b for b in bets if b["edge"] >= STRONG_EDGE]
    marginal = [b for b in bets if b["edge"] < STRONG_EDGE]
    picks = ""
    if strong:
        picks += f"<h2>Strong · edge ≥ {STRONG_EDGE:.0%}</h2>" + "".join(_bet_card(b, injuries) for b in strong)
    if marginal:
        picks += f"<h2>Marginal · edge 0–{STRONG_EDGE:.0%}</h2>" + "".join(_bet_card(b, injuries) for b in marginal)
    if not bets:
        picks = '<div class="card mute">No side has a positive edge today.</div>'
    games = "".join(_game_card(r, team_names, min_edge, injuries) for r in recs)
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>NBA picks {_e(run_label)}</title><style>{_CSS}</style></head><body><main>"
        f"<h1>NBA picks · {_e(run_label)}</h1>"
        f'<p class="sub">{len(recs)} games · {len(bets)} recommended bets ({len(strong)} strong). '
        "Edge = the model's chance of the bet winning minus the chance the odds imply "
        "(break-even). Positive means the model thinks the price is too generous.</p>"
        f"{picks}<h2>All games — tap to drill down</h2>{games}"
        "</main></body></html>"
    )
