"""
Daily capture of winner.co.il's basketball lines page — a full-page
PNG that feeds src/serving/extract_picks.py unchanged. Scope and the
screenshot-over-parser decision: docs/features/serving/
winner_acquisition_scope.md.

The site is a JS app shell behind Imperva bot protection; the probe
findings recorded in the scope doc dictate the shape here: Playwright
driving the INSTALLED Google Chrome, headed (every headless variant is
blocked, headed real Chrome passes). No `playwright install chromium`
needed — channel="chrome" uses the system browser, whose auto-updates
also keep its fingerprint current. The page gets a fixed settle time
after DOM load rather than a content-selector wait: the odds grid's
class names are build artifacts, and selector churn is exactly what the
vision-extraction route exists to avoid.

The whole basketball page is captured, not an NBA-filtered view:
extract_picks only returns games it can resolve to NBA teams, so other
leagues in the screenshot filter themselves out. League-filter
navigation (and the NBA-specific rendering) is deliberately deferred to
an opening-week fine-tune, when NBA lines actually exist to verify
against.

Screenshots are the debugging artifact when extraction misbehaves:
dated files in outputs/winner_captures/ (gitignored), pruned after
RETENTION_DAYS. On a failed capture a best-effort PNG of whatever did
render is written next to the dated path with a `.failed` suffix.
"""

import datetime
import logging
import re
import time
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)

# Winner Line's basketball section; NBA games appear under it (league
# filters are client-side routes — the probe section of the scope doc
# records what the page actually shows).
WINNER_BASKETBALL_URL = "https://www.winner.co.il/mainbook/sport-כדורסל"
DEFAULT_OUTPUT_DIR = Path("outputs/winner_captures")
PAGE_LOAD_TIMEOUT_MS = 45_000
SCREENSHOT_TIMEOUT_MS = 90_000
RENDER_SETTLE_SECONDS = 8.0  # SPA: odds render well after domcontentloaded (probe-verified)
VIEWPORT_RELAYOUT_SECONDS = 2.0  # SPA re-renders rows after a viewport resize
RETENTION_DAYS = 14
VIEWPORT = {"width": 1440, "height": 2400}
# Chromium can't rasterize arbitrarily tall surfaces (~16384px texture
# ceiling); beyond the clamp the page bottom is lost rather than the
# capture hanging.
MAX_CAPTURE_HEIGHT_PX = 16_000

_CAPTURE_NAME = re.compile(r"^winner_nba_(\d{4}-\d{2}-\d{2})(\.failed)?\.png$")


def dated_capture_path(output_dir: Path = DEFAULT_OUTPUT_DIR, today=None) -> Path:
    today = today or datetime.date.today()
    return Path(output_dir) / f"winner_nba_{today.isoformat()}.png"


def failure_path(output_path: Path) -> Path:
    return output_path.with_suffix(".failed.png")


def prune_old_captures(
    output_dir: Path = DEFAULT_OUTPUT_DIR, retention_days: int = RETENTION_DAYS, today=None
) -> int:
    """Deletes dated captures older than the retention window (matching
    files only — anything else in the directory is left alone). Returns
    the number removed."""
    today = today or datetime.date.today()
    cutoff = today - datetime.timedelta(days=retention_days)
    removed = 0
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return 0
    for f in output_dir.iterdir():
        m = _CAPTURE_NAME.match(f.name)
        if not m:
            continue
        try:
            file_date = datetime.date.fromisoformat(m.group(1))
        except ValueError:
            # a stray regex-matching-but-invalid date must not take the
            # capture down over housekeeping
            continue
        if file_date < cutoff:
            f.unlink()
            removed += 1
    return removed


def _grow_viewport_to_content(page) -> int:
    """Resizes the viewport to the page's content height (clamped to
    MAX_CAPTURE_HEIGHT_PX) so a plain viewport screenshot captures the
    whole page. full_page=True is deliberately not used anywhere here:
    once real season content loaded (2026-10-07, ~6.5k-px page) it hung
    past the 90s timeout on every attempt — and left the page unable to
    serve the follow-up diagnostic shot — while a viewport shot of the
    identical content took ~4s. Re-measures after each resize because the
    odds list mounts more rows once they enter the viewport — the loop
    only settles on a measurement that confirmed no further growth, so a
    resize is never the last word on the content height."""
    height = VIEWPORT["height"]
    content = 0
    for _ in range(4):
        content = int(page.evaluate("document.documentElement.scrollHeight") or 0)
        target = min(max(content, VIEWPORT["height"]), MAX_CAPTURE_HEIGHT_PX)
        if target <= height:
            break
        page.set_viewport_size({"width": VIEWPORT["width"], "height": target})
        height = target
        time.sleep(VIEWPORT_RELAYOUT_SECONDS)
    else:
        logger.warning(f"content still growing after 4 viewport resizes; capturing at {height}px")
    if content > MAX_CAPTURE_HEIGHT_PX:
        logger.warning(
            f"content height {content}px exceeds the {MAX_CAPTURE_HEIGHT_PX}px "
            "capture ceiling — page bottom will be cut off"
        )
    return height


def _capture(page, url: str, output_path: Path, settle_seconds: float = RENDER_SETTLE_SECONDS):
    """Drives one page-like object (Playwright's, or a test fake) through
    the capture. On any failure, writes a best-effort screenshot of
    whatever rendered — that picture is the diagnostic — then re-raises."""
    try:
        page.goto(url, timeout=PAGE_LOAD_TIMEOUT_MS, wait_until="domcontentloaded")
        time.sleep(settle_seconds)
        # Force below-the-fold lazy content to render before measuring the
        # content height — capturing with it still streaming in is what
        # stalled first-attempt screenshots during the probe.
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        time.sleep(2)
        page.evaluate("window.scrollTo(0, 0)")
        time.sleep(1)
        _grow_viewport_to_content(page)
        page.screenshot(
            path=str(output_path),
            full_page=False,
            timeout=SCREENSHOT_TIMEOUT_MS,
            animations="disabled",
        )
    except Exception:
        try:
            # The diagnostic must not repeat the configuration that just
            # failed: a grown viewport may itself be why the main shot
            # stalled, so shrink back to the default before shooting.
            page.set_viewport_size(VIEWPORT)
            page.screenshot(
                path=str(failure_path(output_path)),
                full_page=False,
                timeout=SCREENSHOT_TIMEOUT_MS,
                animations="disabled",
            )
            logger.warning(f"capture failed; partial render saved to {failure_path(output_path)}")
        except Exception:
            pass
        raise


@contextmanager
def _playwright_page():
    # Deferred import: playwright is only needed when actually capturing,
    # and tests drive _capture with a fake page instead.
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        # HEADED, real installed Chrome — both are load-bearing, not
        # preferences: Winner sits behind Imperva, and the 2026-10-05
        # probe showed it blocks Playwright's headless shell AND real
        # Chrome in headless mode, while headed real Chrome renders the
        # full lines page (a window appears for ~15s per capture).
        browser = p.chromium.launch(channel="chrome", headless=False)
        try:
            page = browser.new_page(viewport=VIEWPORT, locale="he-IL")
            yield page
        finally:
            browser.close()


def capture_nba_page(
    output_path: Path = None,
    url: str = WINNER_BASKETBALL_URL,
    page_factory=None,
) -> Path:
    """Captures the lines page to a dated PNG. One retry on any failure
    (fresh browser each attempt); raises RuntimeError after both attempts
    fail. `page_factory` is a context manager yielding a page-like object
    (default: real Playwright).

    Retention is enforced only on the managed DEFAULT_OUTPUT_DIR, and only
    on default-path runs — a custom output_path must never cause deletions
    in a directory the caller owns. A prune error can't fail the run: the
    capture on disk is the product, housekeeping is not."""
    if output_path is None:
        output_path = dated_capture_path(DEFAULT_OUTPUT_DIR)
        prune_dir = DEFAULT_OUTPUT_DIR
    else:
        prune_dir = None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    factory = page_factory or _playwright_page

    last_err = None
    for attempt in (1, 2):
        try:
            with factory() as page:
                _capture(page, url, output_path)
        except Exception as e:
            last_err = e
            logger.warning(f"capture attempt {attempt} failed: {e}")
            continue
        # Success: a .failed.png from an earlier attempt (or an earlier
        # run today) is now a misleading diagnostic — remove it.
        failure_path(output_path).unlink(missing_ok=True)
        if prune_dir is not None:
            try:
                prune_old_captures(prune_dir)
            except Exception as e:
                logger.warning(f"capture succeeded but pruning failed (ignored): {e}")
        return output_path
    raise RuntimeError(f"Winner capture failed after 2 attempts: {last_err}") from last_err
