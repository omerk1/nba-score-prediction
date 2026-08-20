#!/usr/bin/env bash
INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // .tool_input.path // empty')
if [ -n "$FILE_PATH" ] && [ -f "$FILE_PATH" ]; then
  echo "Verified: $FILE_PATH" >&2
  ls -la "$FILE_PATH" >&2
  wc -l "$FILE_PATH" >&2
fi
exit 0