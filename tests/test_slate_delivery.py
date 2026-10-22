from types import SimpleNamespace

from src.serving import notify_telegram as N
from src.serving.notify_telegram import SendStatus
from src.serving.recommended_bets import recommended_bets
from src.serving.slate_html import render_slate_html

NAMES = {1: "Lakers", 2: "Celtics"}

REC = {
    "home_team_id": 1,
    "away_team_id": 2,
    "predicted_home_score": 114.2,
    "predicted_away_score": 108.9,
    "predicted_diff": 5.3,
    "predicted_total": 223.1,
    "home_win_probability": 0.68,
    "home_spread": -4.5,
    "home_cover_probability": 0.60,
    "away_cover_probability": 0.40,
    "home_spread_odds": 1.9,
    "away_spread_odds": 1.9,
    "home_spread_market_probability": 0.526,
    "away_spread_market_probability": 0.526,
    "home_spread_edge": 0.074,
    "away_spread_edge": -0.126,
    "total_line": 222.5,
    "over_probability": 0.5,
    "under_probability": 0.5,
    "over_odds": 1.9,
    "under_odds": 1.9,
    "over_market_probability": 0.526,
    "under_market_probability": 0.526,
    "over_edge": -0.026,
    "under_edge": -0.026,
}


def test_only_edges_above_bar_are_recommended():
    bets = recommended_bets(REC, 0.05, NAMES)
    assert [b["pick"] for b in bets] == ["Lakers -4.5"]
    assert recommended_bets(REC, 0.10, NAMES) == []


def test_html_lists_recommended_and_escapes():
    page = render_slate_html([REC], {1: "<b>Lakers", 2: "Celtics"}, "Thu Oct 22", 0.05)
    assert "Strong" in page and "&lt;b&gt;Lakers" in page and "<b>Lakers" not in page


def test_html_without_bets_says_so():
    page = render_slate_html([REC], NAMES, "Thu Oct 22", 0.50)
    assert "No side has a positive edge" in page


class _Transport:
    def __init__(self, fail_documents=False):
        self.calls, self.fail_documents = [], fail_documents

    def post(self, url, **kwargs):
        self.calls.append((url.rsplit("/", 1)[-1], kwargs))
        ok = not (self.fail_documents and url.endswith("sendDocument"))
        return SimpleNamespace(status_code=200 if ok else 400, text="bad", json=lambda: {})


def _setup(monkeypatch, tmp_path, transport):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "t")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "c")
    monkeypatch.setattr(N, "build_team_id_to_name", lambda: NAMES)
    monkeypatch.setattr(N.requests, "post", transport.post)


def test_send_slate_is_a_single_document_message(monkeypatch, tmp_path):
    t = _Transport()
    _setup(monkeypatch, tmp_path, t)
    assert N.send_slate([REC], "Thu Oct 22", reports_dir=tmp_path) is SendStatus.SENT
    assert [c[0] for c in t.calls] == ["sendDocument"]
    assert "Lakers -4.5" in t.calls[0][1]["data"]["caption"]
    assert (tmp_path / "Thu_Oct_22.html").exists()


def test_send_slate_falls_back_to_full_text_when_document_fails(monkeypatch, tmp_path):
    t = _Transport(fail_documents=True)
    _setup(monkeypatch, tmp_path, t)
    assert N.send_slate([REC], "Thu Oct 22", reports_dir=tmp_path) is SendStatus.SENT
    assert [c[0] for c in t.calls] == ["sendDocument", "sendMessage"]


def test_html_shows_out_players_and_flags_missing_snapshot():
    inj = {1: [{"player_name": "A Star", "status": "Out", "reason": "knee"}], 2: []}
    page = render_slate_html([REC], NAMES, "Thu Oct 22", 0.05, injuries=inj)
    assert "A Star" in page and "none listed" in page
    assert "Injury data unavailable" in render_slate_html([REC], NAMES, "Thu Oct 22", 0.05, injuries=None)


def test_injuries_for_date_distinguishes_no_snapshot(tmp_path):
    import sqlite3

    from src.serving.injury_lookup import injuries_for_date

    db = tmp_path / "i.sqlite"
    c = sqlite3.connect(db)
    c.execute("create table scrape_log (game_date, source)")
    c.execute("create table player_injuries (game_date, team_id, player_name, status, reason, source)")
    c.execute("insert into scrape_log values ('2026-10-22','espn')")
    c.executemany(
        "insert into player_injuries values ('2026-10-22',?,?,?,?,'espn')",
        [(1, "B", "Day-To-Day", ""), (1, "A", "Out", "knee"), (1, "C", "Probable", "")],
    )
    c.commit()
    c.close()
    got = injuries_for_date("2026-10-22", str(db))
    assert [p["player_name"] for p in got[1]] == ["A", "B"]
    assert injuries_for_date("2026-10-23", str(db)) is None
