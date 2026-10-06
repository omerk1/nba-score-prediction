"""
One-way Telegram delivery of recommend_game() results — a compact
phone-notification rendering, separate from format_recommendation()'s
terminal output (team IDs, heatmap rows). One module per channel; a
dispatcher only arrives if a second channel ever does. Scope and format
decisions: docs/features/serving/telegram_notify_scope.md.

Parse mode is HTML, hard-coded, not configurable: the formatter emits
HTML tags and html-escapes, so no other mode can work — a knob would
only invite a value that 400-rejects every message. (MarkdownV2 was
ruled out in the scope: it reserves 18 characters including '.', '-',
'(', '+', and one missed escape rejects the whole message.)

Secrets (TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID) come from the
environment — callers (the CLI scripts) load .env via python-dotenv;
this module never reads config.yaml for them, and redacts the token
from logged errors (requests exception text embeds the full URL,
token included).
"""

import html
import logging
import os
import re
import time
from enum import Enum

import requests

from src.serving.team_lookup import build_team_id_to_name
from src.utils.config_loader import load_config

logger = logging.getLogger(__name__)

TELEGRAM_MESSAGE_LIMIT = 4096
MAX_RATE_LIMIT_WAIT_SECONDS = 30
_BLOCK_SEPARATOR = "\n\n"


class SendStatus(str, Enum):
    """Tri-state so callers can tell a real delivery failure from the
    normal disabled no-op — in the scheduled daily job the two must not
    look the same, or a bad chat_id fails silently forever."""

    SENT = "sent"
    SKIPPED = "skipped"
    FAILED = "failed"


def _pct(p: float) -> str:
    return f"{p:.1%}"


def _market_line(label: str, model_prob: float, market_prob=None, edge=None) -> str:
    """One market row: model probability, plus market/edge when both were
    derived (recommend_game sets them together, but guard anyway — a
    half-present pair must degrade to the model-only row, not crash the
    send). A positive-edge row is bolded — that's the line worth acting on."""
    text = f"{label}: {_pct(model_prob)}"
    if market_prob is not None and edge is not None:
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
                rec.get("home_moneyline_edge"),
            )
        )
    if "away_moneyline_market_probability" in rec:
        lines.append(
            _market_line(
                "ML away",
                rec["away_win_probability"],
                rec["away_moneyline_market_probability"],
                rec.get("away_moneyline_edge"),
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
                    rec.get("away_spread_edge"),
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
                    rec.get("under_edge"),
                )
            )
    return "\n".join(lines)


def _truncate_block(block: str) -> str:
    """An oversize block is truncated as PLAIN TEXT: a raw slice could cut
    inside a tag or leave a <b> unclosed, and Telegram 400-rejects the
    whole chunk for invalid HTML — the fallback must not be the thing that
    breaks the message. Tags are stripped (entities like &amp; stay, still
    valid HTML), and a sliced-off trailing entity fragment is dropped."""
    text = re.sub(r"<[^>]+>", "", block)
    text = text[: TELEGRAM_MESSAGE_LIMIT - 12]
    text = re.sub(r"&[a-zA-Z#0-9]*$", "", text)
    return text + "\n…truncated"


def format_telegram_message(recs: list[dict]) -> list[str]:
    """Pure: recommendation dicts -> ready-to-send message chunks. Whole
    game blocks are greedily packed up to Telegram's 4096-char limit,
    never split mid-game; a single block somehow over the limit (can't
    happen at this format) is truncated."""
    if not recs:
        return []
    team_names = build_team_id_to_name()
    blocks = [format_game_block(rec, team_names) for rec in recs]

    messages: list[str] = []
    current = ""
    for block in blocks:
        if len(block) > TELEGRAM_MESSAGE_LIMIT:
            block = _truncate_block(block)
        candidate = f"{current}{_BLOCK_SEPARATOR}{block}" if current else block
        if len(candidate) > TELEGRAM_MESSAGE_LIMIT:
            messages.append(current)
            current = block
        else:
            current = candidate
    messages.append(current)
    return messages


def _redact(text: str, token: str) -> str:
    """requests exception messages embed the full request URL — token
    included — and these errors land in the daily job's launchd log."""
    return text.replace(token, "<token>") if token else text


def _post_message(transport, url: str, payload: dict, timeout_seconds: int, token: str) -> bool:
    for attempt in (1, 2):
        try:
            response = transport.post(url, json=payload, timeout=timeout_seconds)
        except requests.exceptions.ConnectTimeout:
            # The request never left, so resending can't duplicate. A READ
            # timeout is deliberately not retried: Telegram may already
            # have processed the send, and sendMessage has no idempotency
            # key — a rare missing message beats a duplicated slate guess,
            # and it falls through to the generic handler below.
            if attempt == 2:
                logger.error("Telegram send: connect timeout twice; giving up on this message")
                return False
            continue
        except Exception as e:
            logger.error(f"Telegram send failed: {_redact(str(e), token)}")
            return False

        if response.status_code == 200:
            return True
        if response.status_code == 429 and attempt == 1:
            # Rate limited: honor retry_after (capped) and resend once.
            retry_after = 1
            try:
                retry_after = int(response.json()["parameters"]["retry_after"])
            except Exception:
                pass
            time.sleep(min(retry_after, MAX_RATE_LIMIT_WAIT_SECONDS))
            continue
        # Telegram 400 bodies name the offending entity/escape
        logger.error(f"Telegram send failed ({response.status_code}): {response.text}")
        return False
    return False


def send_telegram(
    messages: list[str],
    token: str,
    chat_id: str,
    timeout_seconds: int = 10,
    transport=None,
) -> bool:
    """POSTs each message to the Bot API (parse_mode HTML, matching the
    formatter). Returns True only if every message was accepted. Never
    raises — delivery is a convenience, the recommendations are already
    computed and printed by the caller.

    `transport` is anything with .post(url, json=..., timeout=...)
    (default: the requests module); tests inject a fake."""
    transport = transport or requests
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    all_ok = True
    for text in messages:
        payload = {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}
        if not _post_message(transport, url, payload, timeout_seconds, token):
            all_ok = False
    return all_ok


def _resolve_send_config(config):
    """Config gate + env secrets, shared by every top-level sender.
    Returns (tg_config, token, chat_id) when sending is possible, or a
    SendStatus (SKIPPED/FAILED) explaining why not."""
    config = config or load_config()
    tg = getattr(getattr(config, "notifications", None), "telegram", None)
    if tg is None or not tg.enabled:
        logger.info("Telegram notifications disabled — skipping send")
        return SendStatus.SKIPPED

    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        logger.error(
            "notifications.telegram.enabled is true but TELEGRAM_BOT_TOKEN/"
            "TELEGRAM_CHAT_ID missing from the environment (.env) — skipping send"
        )
        return SendStatus.FAILED
    return tg, token, chat_id


def send_notice(text: str, config=None) -> SendStatus:
    """One short operational message (failure notice, empty slate) — the
    daily job's dead-man's-switch channel. Plain text, escaped here, so
    callers can pass exception strings without HTML concerns. Never
    raises; same status semantics as send_recommendations."""
    resolved = _resolve_send_config(config)
    if isinstance(resolved, SendStatus):
        return resolved
    tg, token, chat_id = resolved
    ok = send_telegram(
        [html.escape(text)], token=token, chat_id=chat_id, timeout_seconds=tg.timeout_seconds
    )
    return SendStatus.SENT if ok else SendStatus.FAILED


def send_recommendations(recs: list[dict], config=None) -> SendStatus:
    """Top-level entry for callers holding recommend_game() dicts: checks
    the config gate and env secrets, formats, sends. Never raises:
    SKIPPED when disabled or nothing to send, FAILED on missing secrets
    or any rejected message."""
    resolved = _resolve_send_config(config)
    if isinstance(resolved, SendStatus):
        return resolved
    tg, token, chat_id = resolved

    messages = format_telegram_message(recs)
    if not messages:
        logger.info("No recommendations to send")
        return SendStatus.SKIPPED
    ok = send_telegram(
        messages,
        token=token,
        chat_id=chat_id,
        timeout_seconds=tg.timeout_seconds,
    )
    return SendStatus.SENT if ok else SendStatus.FAILED
