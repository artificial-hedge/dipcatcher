#!/bin/sh
# Write scrape tokens from the environment, then exec Prometheus.
# The token values are not printed and are not stored in the image.
set -eu
if [ -z "${QUANT_API_KEY:-}" ] || [ -z "${RESEARCH_API_KEY:-}" ]; then
  echo "QUANT_API_KEY and RESEARCH_API_KEY must be set for the Prometheus scrape" >&2
  exit 1
fi
umask 077
printf '%s' "$QUANT_API_KEY" > /tmp/lab.token
printf '%s' "$RESEARCH_API_KEY" > /tmp/research.token
exec /bin/prometheus \
  --config.file=/etc/prometheus/prometheus.yml \
  --storage.tsdb.path=/prometheus \
  --web.console.libraries=/usr/share/prometheus/console_libraries \
  --web.console.templates=/usr/share/prometheus/consoles
