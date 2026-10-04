# Serving — Telegram notification channel (scope)

Goal: a daily job runs the recommendation pipeline
(`src/serving/recommend.py`, via the screenshot CLI or a future scheduled
entry point) and pushes the results to the owner's personal Telegram, so
recommendations arrive without opening a terminal. One-way sends only — no
bot commands, no interaction, no multi-user support.

## Message content + format

`recommend_game()` already returns everything needed; `format_recommendation()`
is terminal-oriented (team IDs, ~25+ lines/game incl. heatmap rows). The
Telegram message is a separate, compact renderer — per game:

- Teams by name, home vs away (the rec dict carries only `home_team_id`/
  `away_team_id`; add an id -> name reverse of `src/serving/team_lookup.py`'s
  mapping — same `nba_api.stats.static.teams` source, don't hand-build).
- Predicted diff and total (`predicted_diff`, `predicted_total`), one line.
- Per market present in the rec (all optional, same as the dict): line,
  model probability, market-implied probability, edge — e.g.
  `Spread -9.0: cover 23% | mkt 55% | edge -32%`.
- Win probability always (the only unconditional probability field).

**Margin heatmap: not sent in v1.** 81 margin:probability pairs is noise in
a phone notification; the actionable numbers (cover/win prob, edge) already
summarize it. Later option: render the pmf as an image (matplotlib bar
chart -> `sendPhoto`), as its own follow-up decision, not part of v1.

**Parse mode: HTML**, not MarkdownV2. MarkdownV2 reserves 18 characters
including `.`, `-`, `(`, `)`, `+` — i.e. every number, spread, and
"(home)" in these messages needs escaping, and a missed escape makes the
API reject the whole message (400). HTML needs only `&`, `<`, `>` escaped
(`html.escape`), and team names / numbers contain none of them in practice;
escape anyway since team names come from an external source. Formatting
used: `<b>` for matchup headers, monospace `<code>` optional for number
alignment — nothing else.

**Length limit: 4096 chars/message.** A compact game block is ~300-400
chars, so a 10+ game slate can exceed one message. Rule: build per-game
blocks, greedily pack whole blocks into messages up to the limit, never
split mid-game. A single block somehow over 4096 is truncated with a
marker (shouldn't happen at this format).

## Module placement

`src/serving/notify_telegram.py` — two functions:
`format_telegram_message(recs: list[dict]) -> list[str]` (pure, returns
ready-to-send chunks) and `send_telegram(messages, token, chat_id,
transport=None)`. One module per channel, named for it; if a second
channel ever lands (email, Slack), add a thin dispatcher then — don't
pre-build one for a single channel. Callers: the end of
`scripts/recommend_from_screenshot.py`'s loop (collect recs, send once at
the end), and whatever the daily-job entry point becomes.

## Config + secrets

- `configs/config.yaml`: new `notifications.telegram` section, following
  the experimental-module pattern (`style_matchup`, `season_motivation`):
  `enabled: bool = False` default, plus `parse_mode: "HTML"` and
  `timeout_seconds: 10`. Pydantic `TelegramNotifyConfig` /
  `NotificationsConfig` added to `src/utils/config_loader.py` like the
  existing per-module schemas.
- Secrets in `.env`, never in config.yaml: `TELEGRAM_BOT_TOKEN`,
  `TELEGRAM_CHAT_ID`, read via the existing `python-dotenv` +
  `os.environ` pattern (`GOOGLE_API_KEY` in `src/serving/extract_picks.py`).
  Note: `python-dotenv` is installed in the venv but missing from
  `requirements.txt` — add it there as part of this feature.
- `enabled: true` with missing env vars → log a clear error and skip
  sending; never crash the run.

## Transport + dependencies

Telegram Bot API is one HTTPS POST:
`https://api.telegram.org/bot<token>/sendMessage` with
`{chat_id, text, parse_mode}`. Use `requests` — already a dependency
(`requests>=2.31.0`, used by `src/news_scraping/` and
`src/polymarket_prices/`). No `python-telegram-bot` SDK: it brings
async/event-loop machinery for bot *interaction*; one-way sends need none
of it, and a new dependency for one POST call isn't justified.

## Error handling

Notification failure must never fail the recommendation run — the
recommendations are already computed and printed; Telegram is a delivery
convenience. Wrap each send: catch `requests` exceptions and non-200
responses, log the error + response body (Telegram 400s say exactly which
entity/escape failed), continue. `timeout_seconds` on every request. No
retry loop in v1 (a missed daily message is low-stakes; the CLI output
still exists) — at most one immediate retry on timeout, decide at
implementation.

## Testing

Mock the transport, never hit the real API — same pattern as
`extract_picks_from_screenshot`'s injectable `client` param and
`src/availability/llm_estimator.py`. `send_telegram(transport=...)`
accepts anything with `.post(url, json=..., timeout=...)`; tests inject a
fake recording transport. Cover: formatting (fields present/absent per
market, HTML escaping), chunking at the 4096 boundary, error swallowing
(transport raises → function logs and returns False, no exception), and
the disabled/missing-secrets no-op paths. One manual real send to verify
the token/chat_id wiring, not a test.

## One-time setup (owner)

1. Telegram → message `@BotFather` → `/newbot` → name it → copy the token
   into `.env` as `TELEGRAM_BOT_TOKEN`.
2. Open the new bot's chat, send `/start` (a bot can't message a user who
   hasn't initiated).
3. `https://api.telegram.org/bot<token>/getUpdates` in a browser → the
   `/start` update's `message.chat.id` is `TELEGRAM_CHAT_ID` → `.env`.
4. Set `notifications.telegram.enabled: true`, run one manual send.

## Open questions

- Daily-job trigger itself (cron? manual?) is out of scope here — this
  scope only defines the channel the job calls.
- Send every game, or only games whose |edge| clears a threshold? v1:
  send everything, flag positive-edge lines (e.g. bold); a
  `min_edge_to_highlight` config knob can come later with real usage.
- Heatmap-as-image (`sendPhoto`): revisit only if the text summary proves
  insufficient in actual use.
