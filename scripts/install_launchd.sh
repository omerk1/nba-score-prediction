#!/bin/bash
# Installs (or removes, with --uninstall) the daily recommendations
# LaunchAgent: scripts/daily_job.py at 16:00 local time, per
# docs/features/serving/daily_scheduling_scope.md. Run as the login
# user, no sudo. The pmset wake schedule DOES need sudo and is printed
# as an instruction rather than run here — check `pmset -g sched`
# first: pmset holds a single system-wide repeating schedule, and
# installing over an existing one would silently replace it.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# REPO_ROOT is templated into plist XML and a zsh -c string below; a
# path with XML- or quote-special characters would produce a malformed
# plist or a mangled command, so refuse it outright.
case "$REPO_ROOT" in
    *[\&\<\>\"\']*)
        echo "Repo path '$REPO_ROOT' contains XML/quote-special characters; refusing to template the plist." >&2
        exit 1
        ;;
esac
LABEL="com.omerkoren.nba-daily-recommendations"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
GUI_DOMAIN="gui/$(id -u)"

if [[ "${1:-}" == "--uninstall" ]]; then
    launchctl bootout "$GUI_DOMAIN/$LABEL" 2>/dev/null || true
    rm -f "$PLIST"
    echo "Uninstalled $LABEL."
    echo "To also clear the wake schedule: sudo pmset repeat cancel"
    exit 0
fi

mkdir -p "$REPO_ROOT/logs" "$HOME/Library/LaunchAgents"

# The job runs through zsh -c so each run day gets its own dated log
# (the scheduling scope's no-sudo log rotation; daily_job.py prunes
# files older than 30 days). StandardErrorPath still catches anything
# that fails before the redirect exists (zsh itself, bad paths).
cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>$LABEL</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/zsh</string>
        <string>-c</string>
        <string>exec "$REPO_ROOT/venv/bin/python3" "$REPO_ROOT/scripts/daily_job.py" >> "$REPO_ROOT/logs/daily_job_\$(date +%Y-%m-%d).log" 2>&1</string>
    </array>
    <key>WorkingDirectory</key>
    <string>$REPO_ROOT</string>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>16</integer>
        <key>Minute</key>
        <integer>0</integer>
    </dict>
    <key>StandardErrorPath</key>
    <string>$REPO_ROOT/logs/launchd_daily_job.err.log</string>
</dict>
</plist>
EOF

launchctl bootout "$GUI_DOMAIN/$LABEL" 2>/dev/null || true
launchctl bootstrap "$GUI_DOMAIN" "$PLIST"
echo "Installed $LABEL (daily 16:00 local). Status:"
launchctl print "$GUI_DOMAIN/$LABEL" | grep -E "state|path|last exit" || true

echo
echo "Current wake schedule (pmset holds ONE repeating schedule system-wide):"
pmset -g sched
echo
echo "To wake the Mac before the run (once, needs sudo; lid-closed wake needs AC power):"
echo "  sudo pmset repeat wakeorpoweron MTWRFSU 15:55:00"
echo
echo "Test the job now without waiting for 16:00:"
echo "  launchctl kickstart $GUI_DOMAIN/$LABEL"
