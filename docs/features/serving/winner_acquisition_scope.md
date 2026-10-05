# Serving — Winner odds acquisition (scope)

Goal: automated daily capture of winner.co.il's NBA offering (spreads,
totals, moneylines as available) in a form the existing pipeline consumes —
feeding `src/serving/extract_picks.py` (vision extraction) →
`recommend.py` → `notify_telegram.py` without changes to any of them. This
is the last component of the daily chain with real unknowns; everything
downstream is built and tested.

## Route decision: screenshot, not HTML/API parsing

Headless-browser screenshot (Playwright + bundled Chromium) of the NBA
page, fed to the existing vision extraction:

- `extract_picks.py` was deliberately built bookmaker-agnostic (loose,
  mostly-optional schema; Hebrew team names and 3-way spread-with-push
  markets — Winner's exact format — are the cases it anticipates). The
  screenshot route reuses it unchanged.
- Resilient to markup churn: the vision model reads whatever the page
  looks like today; an HTML parser couples to selectors that a site
  redesign silently breaks.
- Cost: one page load + one Gemini Flash call per day — fractions of a
  cent (within the cost-discipline rules; iteration during development
  uses one or two deliberately chosen captures, not a loop).

Fallback if the probe kills this route (hard bot-wall against headless
browsers): parse the site's underlying JSON/HTML with the same
`requests` stack. Deterministic but couples to markup and needs a
Hebrew→nickname team mapping that the vision model currently provides
for free. Not scoped further unless needed.

## New dependency

`playwright` (+ `playwright install chromium`, ~130MB one-time). The
project's first non-pure-Python tool dependency; added to
`requirements.txt` with the install step documented. No system Chrome
dependency — the bundled Chromium pins the browser version.

## Module placement

`src/serving/capture_winner.py` — `capture_nba_page(output_path, ...) ->
Path`, plus a thin `scripts/capture_winner.py` CLI (same split as
`daily_update`). Responsibilities: navigate, dismiss the cookie/consent
banner if present, wait for the odds content to render (explicit
selector wait, not a fixed sleep), full-page PNG. One retry on
navigation timeout. Screenshots land in `outputs/winner_captures/`
(dated filenames, gitignored like the rest of `outputs/`), kept ~14 days
— the screenshot is the debugging artifact when extraction goes wrong.

## Unknowns that require a live probe (cannot be scoped from code)

1. The NBA page URL and whether it's stable (vs. session-tokenized).
2. Whether the full slate renders in one page (full-page screenshot
   covers scroll) or needs per-game interaction to reveal lines — if
   spreads show on the list view but totals need a drill-down, v1 ships
   list-view markets only (the extraction schema is all-optional;
   `recommend_game` degrades per-market by design).
3. Bot posture: does a headless Chromium from a residential Israeli IP
   get the real page, a CAPTCHA, or a block? (Datacenter IPs are already
   ruled out for other reasons — nba_api — so only residential matters.)
4. Cookie-banner/consent selector, page language handling, and whether
   tonight's lines are up by the 16:00 IL run (the scheduling scope's
   assumption — verify, and if lines post later, the capture step moves
   to a second, later invocation while the data refresh stays at 16:00).
5. Whether Winner lists NBA preseason games — if yes, opening-week
   rehearsal can start immediately; if no, first live test waits for
   ~Oct 21.

**Probe plan (one session, minimal requests):** manually load the page in
a real browser to find the URL/selectors; then one headless-Chromium
capture; then `scripts/recommend_from_screenshot.py` on the capture (one
Gemini call) to verify extraction end-to-end. A handful of page loads
total — indistinguishable from normal browsing volume.

## Failure handling

- Capture failure (timeout, block, selector never appears) → non-zero
  exit after the one retry; the daily orchestrator (next slice) reports
  it via Telegram, same dead-man's-switch model as the rest of the chain.
  A best-effort screenshot of whatever did render is still written on
  failure — it's the diagnostic.
- "No NBA games recognized" from extraction is a *normal* outcome
  (off-day, offseason), not a failure: capture succeeded, the slate is
  empty. The orchestrator decides whether to send a "no games today"
  note or stay silent — its scope, not this one.

## Testing

The capture module is thin I/O around Playwright; its unit-testable
surface (dated-path construction, retry-on-timeout, failure exit codes)
gets tests with a faked page object. No automated test hits the live
site — the real verification is the probe plus the opening-week
rehearsal. Extraction correctness is already covered by
`tests/test_extract_picks.py`'s fixture screenshots; if Winner's layout
trips extraction during the probe, a cropped fixture of the problem area
joins those fixtures (strip anything account-identifying first).

## ToS note

Scraping winner.co.il likely violates their terms; one page load per day
for personal, non-redistributed use is low practical risk. Known and
accepted at project start — recorded here, not re-litigated.

## Open questions

- All of "Unknowns" above — the probe answers them; the implementation
  PR should update this doc with the answers.
- Playwright pin: latest at implementation time, pinned in
  requirements.txt like the rest.
- Whether to also save the page HTML alongside the PNG on capture (cheap,
  helps the fallback route if it's ever needed) — decide at
  implementation.
