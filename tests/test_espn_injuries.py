from src.news_scraping.scrapers import espn_injuries


class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def _fetch(monkeypatch, injuries):
    monkeypatch.setattr(espn_injuries.requests, "get", lambda *a, **k: _Resp({"injuries": injuries}))
    return espn_injuries.fetch_current_injuries()


def _player(status="Out"):
    return {"status": status, "shortComment": "knee", "athlete": {"displayName": "A Player"}}


def test_team_resolved_from_display_name_when_abbreviation_missing(monkeypatch):
    out = _fetch(monkeypatch, [{"displayName": "Atlanta Hawks", "injuries": [_player()]}])
    assert out[0]["team_abbreviation"] == "ATL"


def test_abbreviation_used_when_present(monkeypatch):
    entry = {"team": {"abbreviation": "BOS"}, "displayName": "Atlanta Hawks", "injuries": [_player()]}
    assert _fetch(monkeypatch, [entry])[0]["team_abbreviation"] == "BOS"


def test_untracked_status_filtered(monkeypatch):
    entry = {"displayName": "Atlanta Hawks", "injuries": [_player("Probable")]}
    assert _fetch(monkeypatch, [entry]) == []


def test_espn_clippers_spelling(monkeypatch):
    out = _fetch(monkeypatch, [{"displayName": "LA Clippers", "injuries": [_player()]}])
    assert out[0]["team_abbreviation"] == "LAC"
