#!/usr/bin/env bash
if git diff --quiet -- docs/NEXT_PHASE_SESSIONS.md && \
   git diff --cached --quiet -- docs/NEXT_PHASE_SESSIONS.md; then
  echo "Reminder: docs/NEXT_PHASE_SESSIONS.md wasn't updated this session. Check whether a status log entry was needed." >&2
fi
exit 0