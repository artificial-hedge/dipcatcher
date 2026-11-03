#!/bin/sh
# Load API keys from *_FILE for the Caddy process. Refuse a plain value when
# the file is set, and under the production profile refuse a plain value
# even when the file is unset. Do not print the values.
set -eu

token_ok() {
  token=$1
  if [ -z "$token" ]; then
    return 1
  fi
  case "$token" in
    *[[:space:]]*)
      return 1
      ;;
  esac
  hash='#'
  open='{'
  close='}'
  dq='"'
  sq="'"
  for fragment in "$hash" "$open" "$close" "$dq" "$sq"; do
    rest=${token#*"$fragment"}
    if [ "$rest" != "$token" ]; then
      return 1
    fi
  done
  return 0
}

load_secret() {
  name=$1
  eval "plain=\${$name-}"
  eval "path=\${${name}_FILE-}"
  if [ -n "$plain" ] && [ -n "$path" ]; then
    echo "$name must not be set in the environment when ${name}_FILE is set" >&2
    exit 1
  fi
  profile=$(printf '%s' "${DIPCATCHER_PROFILE:-}" | tr '[:upper:]' '[:lower:]' | tr -d '[:space:]')
  if [ -n "$plain" ] && { [ "$profile" = "production" ] || [ "$profile" = "prod" ]; }; then
    echo "$name must be loaded from ${name}_FILE when DIPCATCHER_PROFILE is production" >&2
    exit 1
  fi
  if [ -z "$path" ]; then
    return 0
  fi
  if [ -L "$path" ] || [ ! -f "$path" ]; then
    echo "${name}_FILE must contain one or two tokens" >&2
    exit 1
  fi
  count=$(grep -cve '^[[:space:]]*$' "$path" || true)
  if [ "$count" -lt 1 ] || [ "$count" -gt 2 ]; then
    echo "${name}_FILE must contain one or two tokens" >&2
    exit 1
  fi
  value=$(awk 'NF { print; exit }' "$path" | tr -d '\r')
  second=$(awk 'NF { n++; if (n == 2) { print; exit } }' "$path" | tr -d '\r')
  if ! token_ok "$value"; then
    echo "${name}_FILE must contain one or two tokens" >&2
    exit 1
  fi
  if [ "$count" -eq 2 ] && ! token_ok "$second"; then
    echo "${name}_FILE must contain one or two tokens" >&2
    exit 1
  fi
  export "$name=$value"
  unset "${name}_FILE"
  unset value second
}

load_secret QUANT_API_KEY
load_secret RESEARCH_API_KEY
if [ "${1:-}" = "caddy" ]; then
  shift
fi
exec /usr/bin/caddy "$@"
