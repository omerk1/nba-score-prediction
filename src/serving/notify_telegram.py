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
import urllib3

from src.serving.recommended_bets import STRONG_EDGE, recommended_bets
from src.serving.team_lookup import build_team_id_to_name
from src.utils.config_loader import load_config

logger = logging.getLogger(__name__)

TELEGRAM_MESSAGE_LIMIT = 4096
MAX_RATE_LIMIT_WAIT_SECONDS = 30
# ~5.5 min of total backoff for sends that never left the machine
NETWORK_RETRY_WAITS_SECONDS = (5, 15, 30, 60, 120, 120)
# A visible rule, not a blank line: blocks have blank lines between their own sections.
_BLOCK_SEPARATOR = "\n\n──────────\n\n"


class SendStatus(str, Enum):
    """Tri-state so callers can tell a real delivery failure from the
    normal disabled no-op — in the scheduled daily job the two must not
    look the same, or a bad chat_id fails silently forever."""

    SENT = "sent"
    SKIPPED = "skipped"
    FAILED = "failed"


def _pct(p: float) -> str:
    return f"{p:.1%}"


def _bet_line(pick: str, model_prob: float, odds=None, market_prob=None, edge=None) -> str:
    """One bettable side: what to bet, the price, the model's chance and
    the break-even chance the price requires (1/odds), then the edge
    between them. A positive edge gets a check mark and bold: that's the
    side the model thinks the price undervalues. Missing odds degrade to
    the model chance alone (recommend_game sets odds, market probability
    and edge together; a half-present set must not crash the send)."""
    text = pick
    if odds is not None:
        text += f" @{odds:.2f}"
    text += f" — model chance {model_prob:.0%}"
    if market_prob is not None and edge is not None:
        edge_text = f"{edge * 100:+.1f}%".replace("-", "\u2212")
        text += f" · break-even {market_prob:.0%} · edge {edge_text}"
        if edge > 0:
            return f"✅ <b>{text}</b>"
    return f"▫️ {text}"


def format_game_block(rec: dict, team_names: dict) -> str:
    """Compact HTML block for one game. Every market is optional, mirroring
    recommend_game()'s dict. Team names pass through html.escape — they
    come from an external source (nba_api), however tame in practice."""
    home = html.escape(str(team_names.get(rec["home_team_id"], rec["home_team_id"])))
    away = html.escape(str(team_names.get(rec["away_team_id"], rec["away_team_id"])))
    diff = rec["predicted_diff"]
    leader = f"{home} by {diff:.1f}" if diff >= 0 else f"{away} by {-diff:.1f}"
    home_win = rec["home_win_probability"]

    lines = [
        f"<b>{home} vs {away}</b> (home vs away)",
        f"Model: {home} {rec['predicted_home_score']} – {rec['predicted_away_score']} {away}",
        f"{leader} · total {rec['predicted_total']:.1f}",
        f"Win chance: {home} {home_win:.0%} · {away} {1 - home_win:.0%}",
    ]

    moneyline = []
    if "home_moneyline_market_probability" in rec:
        moneyline.append(
            _bet_line(
                f"{home} to win",
                rec["home_win_probability"],
                rec.get("home_moneyline_odds"),
                rec["home_moneyline_market_probability"],
                rec.get("home_moneyline_edge"),
            )
        )
    if "away_moneyline_market_probability" in rec:
        moneyline.append(
            _bet_line(
                f"{away} to win",
                rec["away_win_probability"],
                rec.get("away_moneyline_odds"),
                rec["away_moneyline_market_probability"],
                rec.get("away_moneyline_edge"),
            )
        )
    if moneyline:
        lines += ["", "<b>Moneyline</b>", *moneyline]

    if "home_spread" in rec:
        lines += [
            "",
            "<b>Spread</b>",
            _bet_line(
                f"{home} {rec['home_spread']:+g}",
                rec["home_cover_probability"],
                rec.get("home_spread_odds"),
                rec.get("home_spread_market_probability"),
                rec.get("home_spread_edge"),
            ),
        ]
        if "away_spread_market_probability" in rec:
            lines.append(
                _bet_line(
                    f"{away} {-rec['home_spread']:+g}",
                    rec["away_cover_probability"],
                    rec.get("away_spread_odds"),
                    rec["away_spread_market_probability"],
                    rec.get("away_spread_edge"),
                )
            )

    if "total_line" in rec:
        lines += [
            "",
            "<b>Total points</b>",
            _bet_line(
                f"Over {rec['total_line']:g}",
                rec["over_probability"],
                rec.get("over_odds"),
                rec.get("over_market_probability"),
                rec.get("over_edge"),
            ),
        ]
        if "under_market_probability" in rec:
            lines.append(
                _bet_line(
                    f"Under {rec['total_line']:g}",
                    rec["under_probability"],
                    rec.get("under_odds"),
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


def _never_sent(e: Exception) -> bool:
    """True when the request provably never reached Telegram — DNS
    failure, refused connection, connect timeout — so resending can't
    duplicate it. A connection dropped AFTER connecting (RemoteDisconnected,
    read timeout) may already have been processed and is not covered."""
    if isinstance(e, requests.exceptions.ConnectTimeout):
        return True
    if isinstance(e, requests.exceptions.ConnectionError):
        reason = getattr(e.args[0], "reason", None) if e.args else None
        return isinstance(reason, urllib3.exceptions.NewConnectionError)
    return False


def _post_message(
    transport, url: str, payload: dict, timeout_seconds: int, token: str, files: dict | None = None
) -> bool:
    network_waits = iter(NETWORK_RETRY_WAITS_SECONDS)
    rate_limited = False
    while True:
        try:
            if files:
                response = transport.post(url, data=payload, files=files, timeout=timeout_seconds)
            else:
                response = transport.post(url, json=payload, timeout=timeout_seconds)
        except Exception as e:
            # Pre-connect failures are retried with backoff: right after a
            # scheduled wake the network (DNS especially) can lag by
            # minutes, and this channel is the daily job's only way to
            # report anything. Everything else — notably a READ timeout,
            # where Telegram may already have processed the send and
            # sendMessage has no idempotency key — is not retried: a rare
            # missing message beats a duplicated slate guess.
            wait = next(network_waits, None) if _never_sent(e) else None
            if wait is None:
                logger.error(f"Telegram send failed: {_redact(str(e), token)}")
                return False
            logger.warning(
                f"Telegram send: network unavailable ({type(e).__name__}); retry in {wait}s"
            )
            time.sleep(wait)
            continue

        if response.status_code == 200:
            return True
        if response.status_code == 429 and not rate_limited:
            # Rate limited: honor retry_after (capped) and resend once.
            rate_limited = True
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
            "TELEGRAM_CHAT_ID missing from the environment (.env) — cannot send, "
            "reporting FAILED"
        )
        return SendStatus.FAILED
    return tg, token, chat_id


def send_notice(text: str, config=None) -> SendStatus:
    """One short operational message (failure notice, empty slate) — the
    daily job's dead-man's-switch channel. Plain text, escaped here, so
    callers can pass exception strings without HTML concerns. Truncated
    to the message limit: exception texts can embed an entire model
    response, and a 400-rejected oversize notice would silence the
    channel on exactly the failure it exists to report. Never raises;
    same status semantics as send_recommendations."""
    resolved = _resolve_send_config(config)
    if isinstance(resolved, SendStatus):
        return resolved
    tg, token, chat_id = resolved
    escaped = html.escape(text)
    if len(escaped) > TELEGRAM_MESSAGE_LIMIT:
        escaped = escaped[: TELEGRAM_MESSAGE_LIMIT - 12]
        # don't end on a sliced entity fragment ("&am") — Telegram's HTML
        # parser rejects the whole message over it
        escaped = re.sub(r"&[a-zA-Z#0-9]*$", "", escaped) + "\n…truncated"
    ok = send_telegram(
        [escaped], token=token, chat_id=chat_id, timeout_seconds=tg.timeout_seconds
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


def format_recommended_summary(recs: list[dict], run_label: str, min_edge: float, top: int = 5) -> str:
    """Caption for the HTML page (Telegram captions cap at 1024 chars): the
    headline and the best few bets; the page holds everything."""
    team_names = build_team_id_to_name()
    bets = sorted(
        (b for rec in recs for b in recommended_bets(rec, min_edge, team_names)),
        key=lambda b: b["edge"],
        reverse=True,
    )
    strong = sum(b["edge"] >= STRONG_EDGE for b in bets)
    lines = [
        f"<b>NBA picks · {html.escape(run_label)}</b>",
        f"{len(recs)} games · {len(bets)} recommended bets ({strong} strong, edge ≥ {STRONG_EDGE:.0%})",
        "",
    ]
    for b in bets[:top]:
        odds = f" @{b['odds']:.2f}" if b["odds"] is not None else ""
        lines.append(
            f"{'✅' if b['edge'] >= STRONG_EDGE else '▫️'} <b>{html.escape(b['pick'])}{odds}</b> "
            f"({html.escape(b['game'])}) edge {b['edge'] * 100:+.1f}%"
        )
    if len(bets) > top:
        lines.append(f"…and {len(bets) - top} more in the page")
    if not bets:
        lines.append("No side has a positive edge today.")
    return "\n".join(lines)[:1000]


def send_document(
    path, caption: str, token: str, chat_id: str, timeout_seconds: int = 10, transport=None
) -> bool:
    """Uploads a file to the chat (sendDocument). Same retry/redaction
    rules as send_telegram; the bytes are read once so a retry can resend."""
    from pathlib import Path

    path = Path(path)
    transport = transport or requests
    url = f"https://api.telegram.org/bot{token}/sendDocument"
    payload = {"chat_id": chat_id, "caption": caption, "parse_mode": "HTML"}
    files = {"document": (path.name, path.read_bytes(), "text/html")}
    return _post_message(transport, url, payload, max(timeout_seconds, 60), token, files=files)


def send_html_report(path, caption: str, config=None) -> SendStatus:
    """Config gate + secrets, then one sendDocument with a caption — the
    whole message is a single Telegram post. Never raises."""
    resolved = _resolve_send_config(config)
    if isinstance(resolved, SendStatus):
        return resolved
    tg, token, chat_id = resolved
    try:
        ok = send_document(path, caption, token, chat_id, tg.timeout_seconds)
    except Exception as e:
        logger.error(f"report send failed: {_redact(str(e), token)}")
        ok = False
    return SendStatus.SENT if ok else SendStatus.FAILED


def send_slate(
    recs: list[dict], run_label: str, config=None, reports_dir="outputs/slates", injuries=None
) -> SendStatus:
    """Daily delivery as ONE Telegram message: the HTML page (recommended
    bets first, per-game drill-down below) with a short caption. If building
    or sending the page fails, falls back to the full text messages so the
    slate is never lost to the nicer format. Never raises."""
    from pathlib import Path

    from src.serving.slate_html import render_slate_html

    resolved = _resolve_send_config(config)
    if isinstance(resolved, SendStatus):
        return resolved
    tg, token, chat_id = resolved
    if not recs:
        logger.info("No recommendations to send")
        return SendStatus.SKIPPED
    min_edge = (config or load_config()).notifications.recommend_min_edge

    try:
        slug = re.sub(r"\W+", "_", run_label)
        path = Path(reports_dir) / f"{slug}.html"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_slate_html(recs, build_team_id_to_name(), run_label, min_edge, injuries), encoding="utf-8")
        caption = format_recommended_summary(recs, run_label, min_edge)
        if send_html_report(path, caption, config) is SendStatus.SENT:
            return SendStatus.SENT
        logger.error("HTML slate send failed — sending the full text slate instead")
    except Exception as e:
        logger.error(f"slate HTML failed ({_redact(str(e), token)}) — sending the full text slate")
    return send_recommendations(recs, config)
