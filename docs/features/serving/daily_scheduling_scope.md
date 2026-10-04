# Serving — daily scheduling on macOS (scope)

Goal: the daily recommendation job (data refresh -> scrape/screenshot ->
recommend -> Telegram; the orchestrator script itself is scoped in
`daily_data_refresh_scope.md` / `telegram_notify_scope.md`) runs unattended
on the owner's Mac, including lid-closed/asleep. Decision already made:
launchd user LaunchAgent + `pmset` scheduled wake. This doc scopes only the
scheduling layer — no plist or install script is written here.

## Job shape

One LaunchAgent: `~/Library/LaunchAgents/com.omerkoren.nba-daily-recommendations.plist`,
invoking the single wrapper entry point. Key plist decisions:

- `ProgramArguments`: absolute paths only —
  `/Users/omerkoren/dev/nba-score-prediction/venv/bin/python3` + the
  absolute wrapper-script path. Never bare `python3` (project rule, and
  launchd's default PATH wouldn't resolve the venv anyway).
- `WorkingDirectory`: repo root (`/Users/omerkoren/dev/nba-score-prediction`).
  Required — imports (`from src...`) and relative paths (`configs/`,
  `data/`, `.env`) only resolve from repo root.
- `StartCalendarInterval`: **16:00 Israel time** (launchd uses the Mac's
  local timezone — no conversion needed in the plist). Reasoning: NBA games
  tip off US evenings ≈ 02:00–05:30 Israel; final box scores are in by
  Israel morning and the next slate's Winner lines are posted by Israel
  afternoon — so 16:00 catches both fresh yesterday-results and today's
  lines, with hours of margin before the earliest tipoff.
- `StandardOutPath` / `StandardErrorPath`: `logs/` under the repo root
  (doesn't exist yet — create at install; `.gitignore` already covers it:
  `logs/` and `*.log` entries exist).
- `EnvironmentVariables`: none needed for secrets. launchd jobs do **not**
  inherit shell env (no `.zshrc`), but the serving scripts load `.env` via
  `python-dotenv` — verified: `scripts/recommend_from_screenshot.py` calls
  `load_dotenv()` at import time, which with `WorkingDirectory` = repo root
  picks up the repo's `.env` (`GOOGLE_API_KEY`, `TELEGRAM_*`). The wrapper
  must keep that pattern.

## Missed-run semantics (launchd facts)

- Mac **asleep** at 16:00: launchd coalesces the missed
  `StartCalendarInterval` event and fires the job **once on next wake**.
- Mac **powered off** at 16:00: the event is **not** run retroactively on
  boot — that day's run is simply lost (hence the Telegram dead-man's
  switch below).

## pmset scheduled wake

- `sudo pmset repeat wakeorpoweron MTWRFSU 15:55:00` — wake ~5 min before
  the launchd time so the system is fully up when the job fires.
- Needs sudo **once**; the schedule persists in SMC/PMU across reboots.
- Lid-closed wake requires **AC power connected**; on battery a closed
  MacBook won't wake (degraded-mode note under risks).
- Verify with `pmset -g sched`.
- **Single-slot limitation**: `pmset repeat` holds exactly one repeating
  wake/poweron schedule system-wide — setting this overwrites any existing
  repeat schedule (e.g. a prior Energy Saver "wake for backup" setting).
  Check `pmset -g sched` for an existing entry before installing.

## Overlap / locking

Wake-coalesced run + a manual run can overlap. One-line guard in the
wrapper: a lockfile (`flock`-style, or an atomic `mkdir`/pid-file check) —
second invocation exits immediately with a log line.

## Observability

- Logs: `logs/` in the repo (gitignored). launchd appends to
  `StandardOutPath` forever — rotate via either a `/etc/newsyslog.d/` entry
  or (simpler, no sudo) the wrapper writing dated files
  (`logs/daily_YYYYMMDD.log`) and pruning ones older than ~30 days.
- Failure surfacing: the **Telegram message is the dead-man's switch** —
  no message by Israel evening = investigate `logs/` and
  `launchctl print gui/$UID/com.omerkoren.nba-daily-recommendations`
  (shows last exit status). Optional hardening: the wrapper catches any
  stage failure and sends the error itself to the same Telegram chat, so
  failures are a message rather than silence.

## Install / uninstall

Darwin 22 supports both syntaxes; use the modern one:

- Install: `launchctl bootstrap gui/$UID ~/Library/LaunchAgents/com.omerkoren.nba-daily-recommendations.plist`
- Inspect: `launchctl print gui/$UID/com.omerkoren.nba-daily-recommendations`
- Remove: `launchctl bootout gui/$UID/com.omerkoren.nba-daily-recommendations`
- Manual trigger for testing: `launchctl kickstart gui/$UID/com.omerkoren.nba-daily-recommendations`

(`load`/`unload` still work on Darwin 22 but are deprecated.) A small
`scripts/install_launchd.sh` could template the plist (paths, time) and run
bootstrap + the pmset command — out of scope to write here.

## Known risks / open questions

- **Battery at wake time**: `wakeorpoweron` fires, but a lid-closed MacBook
  on battery won't actually wake. The run isn't lost — launchd coalescing
  runs it at the next manual wake — just late. Degraded, not fatal; keep
  the Mac on AC for reliable 16:00 runs.
- **Login requirement**: a user LaunchAgent (gui domain) runs only while
  the user is logged in. Logged-out Mac = no run. A LaunchDaemon would
  survive logout but is **not recommended**: it runs as root/other user,
  away from the user's home-dir repo, venv, and `.env`. Accept the
  logged-in requirement.
- **Mac leaves home network/country**: the pipeline's Winner-site scrape
  is geo-dependent and `nba_api` is sensitive to non-residential IPs —
  both covered in `daily_data_refresh_scope.md` (and the screenshot path
  in `telegram_notify_scope.md`). Scheduling can't fix this; the job
  should fail loudly (Telegram error message) rather than silently skip.
- **Timezone edge**: `StartCalendarInterval` follows the Mac's local
  clock, but the SMC `pmset` schedule is set once — after a DST shift or
  travel, re-check `pmset -g sched` still precedes 16:00 local.
