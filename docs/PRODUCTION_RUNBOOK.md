# Production runbook

This is the process contract for the lab API (`dipcatcher api`) and the
read-only research API (`python -m quant_fund.api.research_api`). Both are
research and simulated-paper surfaces. They do not place live orders, and a
green healthcheck is not a live-trading claim.

`dipcatcher api --host` and `--port` override `DIPCATCHER_BIND_HOST` and
`DIPCATCHER_BIND_PORT`. The research API has no flags; it reads the
environment only. The process validates this table before it binds. A failed
check exits without listening, and the message names the variable, not the
value.

| Variable | Default | Rule |
|---|---|---|
| `DIPCATCHER_BIND_HOST` | `127.0.0.1` | Lab bind host. `0.0.0.0` requires `QUANT_API_KEY`. |
| `DIPCATCHER_BIND_PORT` | unset, then `PORT`, then `8000` | Lab bind port, integer 1–65535. |
| `DIPCATCHER_CONFIG` | `configs/research.yaml` | Lab only. A path outside `configs/` is rejected. |
| `DIPCATCHER_LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, or `CRITICAL`. |
| `DIPCATCHER_SHUTDOWN_GRACE_S` | `20` | Seconds to finish requests after SIGTERM. `0`–`600`. |
| `DIPCATCHER_OBSERVE` | unset (off) | `1` turns on metrics, JSON audit events, and the alert loop. |
| `DIPCATCHER_METRICS_HOST` | `127.0.0.1` | Used only when `DIPCATCHER_METRICS_PORT` is set. |
| `DIPCATCHER_METRICS_PORT` | unset | Requires `DIPCATCHER_OBSERVE=1`. `/metrics` has no authentication. Do not publish it. |
| `DIPCATCHER_OTEL_ENDPOINT` | unset | `http` or `https` URL with a host and no userinfo. Requires `DIPCATCHER_OBSERVE=1`. |
| `DIPCATCHER_SLO_PATH` | shipped `deploy/observability/slo.yml` when observe is on and this is unset | Requires `DIPCATCHER_OBSERVE=1`. A symlink or invalid file refuses startup. |
| `DIPCATCHER_ALERT_INTERVAL_S` | `60` | Alert loop period. `1`–`86400`. |
| `DIPCATCHER_KILL_SWITCH_PATH` | unset | See below. Missing file means `ENABLED`. |
| `PORT` | `8000` lab / `8010` research image | Fallback when the service's named port is unset. |
| `QUANT_API_KEY` | unset | Required to bind the lab API off loopback. Sent as `X-API-Key`. |
| `RESEARCH_API_HOST` | `127.0.0.1` (image sets `0.0.0.0`) | Research bind host. Off-loopback requires `RESEARCH_API_KEY`. |
| `RESEARCH_API_PORT` | unset, then `PORT`, then `8010` | Research bind port. |
| `RESEARCH_API_KEY` | unset | Required to bind the research API off loopback. |

The lab image sets `PORT=8000` and `DIPCATCHER_BIND_PORT=8000`. The research
image sets `PORT=8010` and `RESEARCH_API_PORT=8010`. The container probe
reads `DIPCATCHER_BIND_PORT`, then `RESEARCH_API_PORT`, then `PORT`, and
requests `http://127.0.0.1:<port>/health`.

## Start

Local lab API, loopback:

```bash
uv run dipcatcher api
```

Container, published only on the host loopback. Export `QUANT_API_KEY` in
the shell first. Do not put the value in the command line of a recorded
shell history if you can avoid it; the compose file reads it from the
environment.

```bash
docker build -t dipcatcher:local .
docker compose -f deploy/compose.prod.yml up -d
```

Equivalent `docker run`:

```bash
docker run --detach --name dipcatcher \
  -p 127.0.0.1:8000:8000 \
  -e QUANT_API_KEY \
  -e DIPCATCHER_BIND_HOST=0.0.0.0 \
  -e DIPCATCHER_BIND_PORT=8000 \
  -e DIPCATCHER_OBSERVE=1 \
  -e DIPCATCHER_METRICS_PORT=9464 \
  -v dipcatcher-data:/app/data \
  dipcatcher:local
```

`GET /health` is liveness and stays public. `GET /ready` is readiness. On a
fresh tree it returns 503 until a data manifest and a research receipt
exist. That is fail-closed, not a crashed process. `/health` returns 200
while `/ready` is 503.

Research API image:

```bash
docker build -f docker/research-api.Dockerfile -t dipcatcher-research-api .
docker run --rm -p 127.0.0.1:8010:8010 \
  -e RESEARCH_API_KEY \
  -v /path/to/data:/app/data:ro \
  dipcatcher-research-api
```

## Stop

`docker stop` sends SIGTERM. The process marks itself not ready, stops
accepting connections, waits `DIPCATCHER_SHUTDOWN_GRACE_S` (compose allows
30 seconds), then stops the metrics server and the alert loop. After that
drain the process re-raises SIGTERM, so a shell reports exit status 143.
That status means the shutdown finished.

```bash
docker compose -f deploy/compose.prod.yml stop
docker stop dipcatcher
```

SIGKILL skips the drain. Use it only when SIGTERM does not finish inside
the grace period.

## Rollback

There is no model checkpoint in this image. Roll back by running the
previous image tag and keeping the data volume:

```bash
docker compose -f deploy/compose.prod.yml down
docker compose -f deploy/compose.prod.yml up -d
```

Pin the image tag you intend to return to before you pull a replacement.
Receipts already written are immutable. Retract a bad research receipt
with `dipcatcher tombstone`. Do not edit the receipt file.

## Kill switch

Write one token into the file named by `DIPCATCHER_KILL_SWITCH_PATH`:

`ENABLED`, `HALT_NEW_ORDERS`, `CANCEL_OPEN_ORDERS`, or `FLATTEN_OPTIONAL`.

Any token other than `ENABLED` makes `/ready` return 503 on the next probe.
A missing file is `ENABLED`. A symlink, directory, or unknown body also
fails closed. The body is never copied into logs or the readiness payload.

This file does not cancel paper orders and does not flatten a book. Those
actions stay on `dipcatcher paper --halt` and the operator desk. It is also
not the fx-1 harness drain.

Clear the halt by writing `ENABLED`, or by removing the file.

## Logs

Service events and uvicorn logs are JSON lines on stderr. Docker keeps them
on the container's log stream:

```bash
docker logs dipcatcher
```

There is no log file inside the image. Values of known secret environment
variables are replaced with `[redacted]` before a line is written. Do not
add a new log field that copies a credential.

Startup emits `service_start`. SIGTERM emits `service_draining`, then
`service_shutdown`. With `DIPCATCHER_OBSERVE=1`, SLO and kill-switch
transitions emit `alert`.

## What this does not do

`/metrics` is unauthenticated. Leave it on `127.0.0.1` and do not publish
the port. The image's `uv` installer is version-pinned, not hash-pinned.
Crown-jewel and epoch pins are re-signed by the key holder after review.
This runbook does not authorize live trading.
