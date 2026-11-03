#!/bin/sh
# Load scrape tokens from *_FILE when that is how the secret is provided,
# then exec Prometheus. The token values are not printed and are not stored
# in the image. A plain value is refused when the file is set, and under
# the production profile a plain value is refused on its own.
set -eu

token_ok() {
  value=$1
  if [ -z "$value" ]; then
    return 1
  fi
  case "$value" in
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
    rest=${value#*"$fragment"}
    if [ "$rest" != "$value" ]; then
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
    echo "${name}_FILE must contain one token" >&2
    exit 1
  fi
  value=$(tr -d '\r\n' < "$path")
  if ! token_ok "$value"; then
    echo "${name}_FILE must contain one token" >&2
    exit 1
  fi
  export "$name=$value"
  unset "${name}_FILE"
  unset value
}

load_secret QUANT_API_KEY
load_secret RESEARCH_API_KEY
if [ -z "${QUANT_API_KEY:-}" ] || [ -z "${RESEARCH_API_KEY:-}" ]; then
  echo "QUANT_API_KEY and RESEARCH_API_KEY must be set for the Prometheus scrape" >&2
  exit 1
fi
umask 077
printf '%s' "$QUANT_API_KEY" > /tmp/lab.token
printf '%s' "$RESEARCH_API_KEY" > /tmp/research.token
unset QUANT_API_KEY RESEARCH_API_KEY
exec /bin/prometheus \
  --config.file=/etc/prometheus/prometheus.yml \
  --storage.tsdb.path=/prometheus \
  --web.console.libraries=/usr/share/prometheus/console_libraries \
  --web.console.templates=/usr/share/prometheus/consoles
