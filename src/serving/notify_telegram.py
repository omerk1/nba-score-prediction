"""
One-way Telegram delivery of recommend_game() results — a compact
phone-notification rendering, separate from format_recommendation()'s
terminal output (team IDs, heatmap rows). One module per channel; a
dispatcher only arrives if a second channel ever does. Scope and format
decisions: docs/features/serving/telegram_notify_scope.md.

Parse mode is HTML, not MarkdownV2: MarkdownV2 reserves 18 characters
including '.', '-', '(', '+' — every number and spread would need
escaping, and one miss makes the API reject the whole message (400).
HTML needs only &, <, > escaped.

Secrets (TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID) come from the
environment — callers (the CLI scripts) load .env via python-dotenv;
this module never reads config.yaml for them.
"""

import html
import logging
import os

import requests

from src.utils.config_loader import load_config

logger = logging.getLogger(__name__)

TELEGRAM_MESSAGE_LIMIT = 4096
_BLOCK_SEPARATOR = "\n\n"


def build_team_id_to_name() -> dict:
    """id -> nickname, the reverse of team_lookup's mapping — same
    nba_api static source (offline data, no network call)."""
    from nba_api.stats.static import teams as nba_teams

    return {t["id"]: t["nickname"] for t in nba_teams.get_teams()}


def _pct(p: float) -> str:
    return f"{p:.1%}"


def _market_line(label: str, model_prob: float, market_prob=None, edge=None) -> str:
    """One market row: model probability, plus market/edge when odds were
    given. A positive-edge row is bolded — that's the line worth acting on."""
    text = f"{label}: {_pct(model_prob)}"
    if market_prob is not None:
        text += f" | mkt {_pct(market_prob)} | edge {edge:+.1%}"
        if edge > 0:
            text = f"<b>{text}</b>"
    return text


def format_game_block(rec: dict, team_names: dict) -> str:
    """Compact HTML block for one game. Every market is optional, mirroring
    recommend_game()'s dict. Team names pass through html.escape — they
    come from an external source (nba_api), however tame in practice."""
    home = html.escape(str(team_names.get(rec["home_team_id"], rec["home_team_id"])))
    away = html.escape(str(team_names.get(rec["away_team_id"], rec["away_team_id"])))

    lines = [
        f"<b>{home} vs {away}</b> (home vs away)",
        f"Predicted: {rec['predicted_home_score']} - {rec['predicted_away_score']}"
        f" (diff {rec['predicted_diff']:+.1f}, total {rec['predicted_total']:.1f})",
        f"Home win: {_pct(rec['home_win_probability'])}",
    ]

    if "home_moneyline_market_probability" in rec:
        lines.append(
            _market_line(
                "ML home",
                rec["home_win_probability"],
                rec["home_moneyline_market_probability"],
                rec["home_moneyline_edge"],
            )
        )
    if "away_moneyline_market_probability" in rec:
        lines.append(
            _market_line(
                "ML away",
                rec["away_win_probability"],
                rec["away_moneyline_market_probability"],
                rec["away_moneyline_edge"],
            )
        )
    if "home_spread" in rec:
        lines.append(
            _market_line(
                f"Spread {rec['home_spread']:+g} home cover",
                rec["home_cover_probability"],
                rec.get("home_spread_market_probability"),
                rec.get("home_spread_edge"),
            )
        )
        if "away_spread_market_probability" in rec:
            lines.append(
                _market_line(
                    "Away cover",
                    rec["away_cover_probability"],
                    rec["away_spread_market_probability"],
                    rec["away_spread_edge"],
                )
            )
    if "total_line" in rec:
        lines.append(
            _market_line(
                f"Total {rec['total_line']:g} over",
                rec["over_probability"],
                rec.get("over_market_probability"),
                rec.get("over_edge"),
            )
        )
        if "under_market_probability" in rec:
            lines.append(
                _market_line(
                    "Under",
                    rec["under_probability"],
                    rec["under_market_probability"],
                    rec["under_edge"],
                )
            )
    return "\n".join(lines)


def format_telegram_message(recs: list[dict]) -> list[str]:
    """Pure: recommendation dicts -> ready-to-send message chunks. Whole
    game blocks are greedily packed up to Telegram's 4096-char limit,
    never split mid-game; a single block somehow over the limit (can't
    happen at this format) is truncated with a marker."""
    if not recs:
        return []
    team_names = build_team_id_to_name()
    blocks = [format_game_block(rec, team_names) for rec in recs]

    messages: list[str] = []
    current = ""
    for block in blocks:
        if len(block) > TELEGRAM_MESSAGE_LIMIT:
            block = block[: TELEGRAM_MESSAGE_LIMIT - 12] + "\n…truncated"
        candidate = f"{current}{_BLOCK_SEPARATOR}{block}" if current else block
        if len(candidate) > TELEGRAM_MESSAGE_LIMIT:
            messages.append(current)
            current = block
        else:
            current = candidate
    messages.append(current)
    return messages


def send_telegram(
    messages: list[str],
    token: str,
    chat_id: str,
    parse_mode: str = "HTML",
    timeout_seconds: int = 10,
    transport=None,
) -> bool:
    """POSTs each message to the Bot API. Returns True only if every
    message was accepted. Never raises — delivery is a convenience, the
    recommendations are already computed and printed by the caller.

    `transport` is anything with .post(url, json=..., timeout=...)
    (default: the requests module); tests inject a fake."""
    transport = transport or requests
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    all_ok = True
    for text in messages:
        payload = {"chat_id": chat_id, "text": text, "parse_mode": parse_mode}
        for attempt in (1, 2):  # one immediate retry, on timeout only
            try:
                response = transport.post(url, json=payload, timeout=timeout_seconds)
                if response.status_code != 200:
                    # Telegram 400 bodies name the offending entity/escape
                    logger.error(f"Telegram send failed ({response.status_code}): {response.text}")
                    all_ok = False
                break
            except requests.exceptions.Timeout:
                if attempt == 2:
                    logger.error("Telegram send timed out twice; giving up on this message")
                    all_ok = False
            except Exception as e:
                logger.error(f"Telegram send failed: {e}")
                all_ok = False
                break
    return all_ok


def send_recommendations(recs: list[dict], config=None) -> bool:
    """Top-level entry for callers holding recommend_game() dicts: checks
    the config gate and env secrets, formats, sends. Returns False (never
    raises) when disabled, unconfigured, or any send failed."""
    config = config or load_config()
    tg = getattr(getattr(config, "notifications", None), "telegram", None)
    if tg is None or not tg.enabled:
        logger.info("Telegram notifications disabled — skipping send")
        return False

    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        logger.error(
            "notifications.telegram.enabled is true but TELEGRAM_BOT_TOKEN/"
            "TELEGRAM_CHAT_ID missing from the environment (.env) — skipping send"
        )
        return False

    messages = format_telegram_message(recs)
    if not messages:
        logger.info("No recommendations to send")
        return False
    return send_telegram(
        messages,
        token=token,
        chat_id=chat_id,
        parse_mode=tg.parse_mode,
        timeout_seconds=tg.timeout_seconds,
    )
