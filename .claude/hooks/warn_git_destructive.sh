#!/usr/bin/env bash
INPUT=$(cat)
CMD=$(echo "$INPUT" | jq -r '.tool_input.command // empty')
[ -z "$CMD" ] && exit 0

if echo "$CMD" | grep -qE '(^|[;&|]|[[:space:]])git[[:space:]]+(checkout|reset[[:space:]]+--hard|clean[[:space:]]+-f)'; then
  DIRTY=$(git status --short 2>/dev/null)
  if [ -n "$DIRTY" ]; then
    N=$(printf '%s\n' "$DIRTY" | wc -l | tr -d ' ')
    {
      echo "WARNING: about to run a git command that can discard uncommitted work:"
      echo "  $CMD"
      echo "This repo currently has $N uncommitted change(s) -- if any predate this session (e.g. a prior session's unfinished/uncommitted work), this command may destroy them:"
      echo "$DIRTY"
      echo "Consider committing or stashing first. Not blocking -- proceeding."
    } >&2
    SUMMARY="git-destructive-command warning: '$CMD' with $N uncommitted change(s) present -- see stderr for the file list."
    printf '{"systemMessage": %s}\n' "$(printf '%s' "$SUMMARY" | jq -Rs .)"
  fi
fi
exit 0
