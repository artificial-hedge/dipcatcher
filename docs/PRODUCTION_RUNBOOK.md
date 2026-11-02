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
| `DIPCATCHER_METRICS_PORT` | unset | Requires `DIPCATCHER_OBSERVE=1`. `/metrics` requires the service API key as `X-API-Key` or `Authorization: Bearer`, unless the loopback scrape opt-out is set. `dipcatcher smoke-up` installs that server at import, so the lab CLI defaults `DIPCATCHER_METRICS_KEY_ENV` to `QUANT_API_KEY` when it is unset. The research process sets the name to `RESEARCH_API_KEY` before it binds. |
| `DIPCATCHER_OTEL_ENDPOINT` | unset | `http` or `https` URL with a host and no userinfo. Requires `DIPCATCHER_OBSERVE=1`. |
| `DIPCATCHER_RATE_LIMIT_RPS` | unset (off) | Finite rate from 0 to 1000000 for both APIs. `0` or unset disables the limit outside production. `/health` is exempt. Overload returns HTTP 429. An invalid value refuses startup. A production profile refuses to boot unless this is set and above 0. |
| `DIPCATCHER_PROFILE` | unset | `production` or `prod` requires `DIPCATCHER_RATE_LIMIT_RPS` set and above 0, and forces the data-dir write probe even when `DIPCATCHER_REQUIRE_DATA_WRITE=0`. Any other value keeps the non-production defaults. Set it in `deploy/prod.env`, not in the example env file. |
| `DIPCATCHER_SLO_PATH` | shipped `deploy/observability/slo.yml` when observe is on and this is unset | Requires `DIPCATCHER_OBSERVE=1`. A symlink or invalid file refuses startup. |
| `DIPCATCHER_ALERT_INTERVAL_S` | `60` | Alert loop period. `1`–`86400`. |
| `DIPCATCHER_METRICS_LOOPBACK_SCRAPE` | unset (off) | Allows an unauthenticated `/metrics` scrape only when `DIPCATCHER_METRICS_HOST` is loopback. Any other host refuses startup. |
| `DIPCATCHER_KILL_SWITCH_PATH` | unset | See below. Missing file means `ENABLED`. |
| `PORT` | `8000` lab / `8010` research image | Fallback when the service's named port is unset. |
| `QUANT_API_KEY` | unset | Required to bind the lab API off loopback. Sent as `X-API-Key` or `Authorization: Bearer`. |
| `RESEARCH_API_HOST` | `127.0.0.1` (image sets `0.0.0.0`) | Research bind host. Off-loopback requires `RESEARCH_API_KEY`. |
| `RESEARCH_API_PORT` | unset, then `PORT`, then `8010` | Research bind port. |
| `RESEARCH_API_KEY` | unset | Required to bind the research API off loopback. Sent as `X-API-Key` or `Authorization: Bearer`. |

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
docker build -f docker/research-api.Dockerfile -t dipcatcher-research-api .
docker compose -f deploy/compose.prod.yml up -d
```

That compose file starts the lab API, the research API, Prometheus, and
Grafana. Published ports stay on `127.0.0.1`. The defaults are lab `8000`,
research `8010`, Prometheus `9090`, and Grafana `3000`. Set
`LAB_PUBLISH_PORT`, `RESEARCH_PUBLISH_PORT`, `PROMETHEUS_PUBLISH_PORT`, or
`GRAFANA_PUBLISH_PORT` when a default is already taken. Do not publish
`9464`. Prometheus reads bearer tokens from files it writes at start and
scrapes `lab-api:9464` and `research-api:9464`. Those files are not in git.
Grafana anonymous access is a local viewer. The compose file does not set
an admin password.

Equivalent `docker run`:

```bash
docker run --detach --name dipcatcher \
  -p 127.0.0.1:8000:8000 \
  -e QUANT_API_KEY \
  -e DIPCATCHER_BIND_HOST=0.0.0.0 \
  -e DIPCATCHER_BIND_PORT=8000 \
  -e DIPCATCHER_OBSERVE=1 \
  -e DIPCATCHER_METRICS_PORT=9464 \
  dipcatcher:local
```

The image creates `/app/data` owned by uid 10001. The entrypoint
(`DIPCATCHER_REQUIRE_DATA_WRITE=1`, `DIPCATCHER_DATA_ROOT=/app/data`) refuses
to start when that directory is missing, is a symlink, or is not writable by
uid 10001. The message names the path and the uid. A root-owned named volume
fails that check. The smoke compose file uses the image directory. The
research image sets `DIPCATCHER_REQUIRE_DATA_WRITE=0`, so a `:ro` mount still
starts when `DIPCATCHER_PROFILE` is unset, and it still refuses a symlink.
`deploy/prod.env` sets `DIPCATCHER_PROFILE=production`,
`DIPCATCHER_RATE_LIMIT_RPS=50`, and `DIPCATCHER_REQUIRE_DATA_WRITE=1`.
The entrypoint also forces the write probe for `production` or `prod`, so
a read-only research data mount fails closed the same way the lab mount
does. A full data directory fails that probe before bind. The message says
the directory is full and names uid 10001. A production boot with a missing,
zero, or invalid rate exits before exec. The message names
`DIPCATCHER_RATE_LIMIT_RPS` and does not print the value.

`GET /health` is liveness and stays public. `GET /ready` is readiness. On a
fresh tree it returns 503 until a data manifest and a research receipt
exist. That is fail-closed, not a crashed process. `/health` returns 200
while `/ready` is 503.

`dipcatcher ready-bootstrap` writes a labeled SYNTHETIC manifest and receipt
under the config data root when those files are missing. `dipcatcher smoke-up`
runs that bootstrap and then the lab API. The smoke compose file uses
`smoke-up`. An invalid file that is already present is left unchanged.

Research API image:

```bash
docker build -f docker/research-api.Dockerfile -t dipcatcher-research-api .
docker run --rm -p 127.0.0.1:8010:8010 \
  -e RESEARCH_API_KEY \
  -v /path/to/data:/app/data:ro \
  dipcatcher-research-api
```

That `:ro` mount is the image default: the write flag is off and the
production profile is unset. `deploy/prod.env` refuses the same mount.

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

## Rolling restart

`scripts/rolling_restart.py` drains one container, waits until `/ready` is
not 200, stops it, starts it, and waits until `/ready` is 200. SIGTERM sets
the drain latch, so `/ready` returns 503 before the process exits. The load
check uses `/health`, which stays 200 until the socket closes. A 503 from
`/ready` is the drain signal. A single replica has a gap of transport
errors while it is stopped. Those are not HTTP 5xx. The script exits 1
when `scripts/http_load.py` reports any HTTP 5xx. The key is not printed.

```bash
python scripts/rolling_restart.py \
  --container dipcatcher-ops6-lab-api-1 \
  --ready-url http://127.0.0.1:8000/ready \
  --env-file /secure/lab.env \
  --key-env QUANT_API_KEY \
  --load-url http://127.0.0.1:8000/health \
  --load-seconds 45 \
  --load-concurrency 10
```

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

`ENABLED`, `HALT_NEW_ORDERS`, `CANCEL_OPEN_ORDERS`, `FLATTEN_OPTIONAL`, or
`FLATTEN`.

Any token other than `ENABLED` makes `/ready` return 503 on the next probe.
A missing file is `ENABLED`. A symlink, directory, or unknown body also
fails closed. The body is never copied into logs or the readiness payload.

`dipcatcher live kill` writes `HALT_NEW_ORDERS` to this file (or to
`--kill-switch-file`, or to `<desk>/kill_switch` when neither is set) and
then latches the desk. It does not cancel or flatten a paper book, and it
does not write `FLATTEN`. `dipcatcher live reset` clears the desk and then
writes `ENABLED`. `dipcatcher paper --halt` and `--clear-halt` write the
same file when `DIPCATCHER_KILL_SWITCH_PATH` is set.

New paper orders and the paper fill ledger call the same reader. A blocking
token, or a symlink, directory, or unknown body, refuses the order and does
not change the book.

`HALT_NEW_ORDERS` cancels resting paper orders only when
`DIPCATCHER_HALT_CANCELS_OPEN_ORDERS` is `1`, `true`, `yes`, or `on`. The
default leaves those orders in place. `CANCEL_OPEN_ORDERS` always cancels
resting paper orders and does not close positions. `FLATTEN_OPTIONAL` does
not cancel or flatten; the in-memory human gate is unchanged.
`FLATTEN` cancels resting paper orders and then closes the simulated book at
the last positive finite mark, in symbol order, with zero costs. A missing
mark cancels the orders and leaves every position. This is a paper
mark-to-cash close, not a live order. The live broker is not on this path.

`fx1 harness serve` watches the file. A blocking token sets the harness
drain immediately. After the file stays `ENABLED` for
`DIPCATCHER_KILL_DRAIN_DEBOUNCE_S` (default 1 second, maximum 300), the
watcher clears only the latch it set and logs `operator_kill_drain_cleared`.
An invalid debounce never clears that latch and logs
`operator_kill_drain_debounce_invalid`. `POST /harness/drain` stays one-way.
`python -m fx1.serve.api` watches the same file. The pinned module is
unchanged; `fx1.serve` installs a loader hook only when `sys.argv[0]` is
`-m`, and that hook routes `uvicorn.run` through the same watcher. A normal
import does not install it. The lab process SIGTERM drain is also one-way.

Clear the halt by writing `ENABLED`, or by removing the file.

Paper ledger and checkpoint backup:

```bash
python scripts/paper_state_backup.py backup --source /path/to/data --dest /path/to/archive
python scripts/paper_state_backup.py restore --archive /path/to/archive --dest /path/to/data
```

The archive covers `metadata/paper/` and `checkpoints/` under the data root,
or a single run directory that contains `broker_state.json`. Restore checks
every hash before it writes. A symlink, a tampered file, or a destination
file with different bytes is refused. The script does not archive the rest
of the tree.

## Incident response

These commands are the operator path for a simulated-paper incident. They do
not place a live order. Set `DIPCATCHER_KILL_SWITCH_PATH` to a regular file
the process can read. The writer below refuses a symlink. Do not edit
receipts, `gate_pins.sig`, or corpus epochs.

Stop the lab process with SIGTERM. `docker stop` waits for the grace period.
Exit code 143 on the container means the drain finished (the process
re-raised SIGTERM). SIGKILL skips the drain.

```bash
docker stop dipcatcher
docker inspect --format '{{.State.ExitCode}}' dipcatcher
```

Halt new paper orders and close `/ready`. This writes `HALT_NEW_ORDERS` and,
when you use the live command, also trips the desk. It does not cancel a
paper book unless `DIPCATCHER_HALT_CANCELS_OPEN_ORDERS` is `1`, `true`,
`yes`, or `on`, and it does not flatten.

```bash
python -c "import os; from pathlib import Path; from quant_fund.utils.operator_kill import write_operator_kill; write_operator_kill(Path(os.environ['DIPCATCHER_KILL_SWITCH_PATH']), 'HALT_NEW_ORDERS')"
dipcatcher live kill --reason incident-halt --kill-switch-file "$DIPCATCHER_KILL_SWITCH_PATH"
```

Cancel resting paper orders and leave positions:

```bash
python -c "import os; from pathlib import Path; from quant_fund.utils.operator_kill import write_operator_kill; write_operator_kill(Path(os.environ['DIPCATCHER_KILL_SWITCH_PATH']), 'CANCEL_OPEN_ORDERS')"
```

Flatten the simulated paper book. The next paper loop cancels resting
orders, then closes positions at the last positive finite mark, in symbol
order, with zero costs. A missing mark cancels orders and leaves every
position. `dipcatcher live flatten` addresses the desk or gateway and is not
this paper close.

```bash
python -c "import os; from pathlib import Path; from quant_fund.utils.operator_kill import write_operator_kill; write_operator_kill(Path(os.environ['DIPCATCHER_KILL_SWITCH_PATH']), 'FLATTEN')"
```

The fx-1 harness (`fx1 harness serve` and `python -m fx1.serve.api`) latches
its drain on the blocking token immediately. After the file stays `ENABLED`
for `DIPCATCHER_KILL_DRAIN_DEBOUNCE_S`, it clears only the latch it set.
`POST /harness/drain` and the lab SIGTERM latch stay set.

Clear the file after the book is where you want it:

```bash
python -c "import os; from pathlib import Path; from quant_fund.utils.operator_kill import write_operator_kill; write_operator_kill(Path(os.environ['DIPCATCHER_KILL_SWITCH_PATH']), 'ENABLED')"
dipcatcher live reset --reason incident-clear --kill-switch-file "$DIPCATCHER_KILL_SWITCH_PATH"
```

`dipcatcher paper --halt` and `dipcatcher paper --clear-halt` write the same
file when `DIPCATCHER_KILL_SWITCH_PATH` is set, and they also start a paper
run. Use the `write_operator_kill` commands above when you only need the file.

Back up the paper ledger and checkpoints before you restore or roll back.
Stop the process first so it is not writing the same files.

```bash
python scripts/paper_state_backup.py backup --source /var/lib/dipcatcher/data --dest /var/lib/dipcatcher/backup
python scripts/paper_state_backup.py restore --archive /var/lib/dipcatcher/backup --dest /var/lib/dipcatcher/data
```

Restore checks every archived hash before it writes. It refuses a symlink,
a tampered payload, and a destination file whose bytes differ.

Roll back by starting the previous image tag with the same data directory.
The directory must be writable by uid 10001. Keep port 8000 on loopback.
Do not publish `9464`.

```bash
docker tag dipcatcher:local dipcatcher:previous
docker stop dipcatcher
docker rm dipcatcher
docker run -d --name dipcatcher \
  --env-file /secure/dipcatcher.env \
  -p 127.0.0.1:8000:8000 \
  -v /var/lib/dipcatcher/data:/app/data \
  --stop-signal SIGTERM \
  dipcatcher:previous
docker inspect --format '{{.State.Health.Status}}' dipcatcher
```

With compose, pin the tag you saved and do not rebuild the bad tree:

```bash
docker compose -f deploy/compose.prod.yml stop lab-api
docker tag dipcatcher:previous dipcatcher:local
docker compose -f deploy/compose.prod.yml up -d --no-build lab-api
```

Retract a bad research receipt. This appends a tombstone. It does not edit
the receipt.

```bash
dipcatcher tombstone receipts/bad-receipt.json --reason "retracted after rollback"
```

With `DIPCATCHER_OBSERVE=1` and `DIPCATCHER_METRICS_PORT=9464`, the lab scrape
exposes `dipcatcher_ready`, `dipcatcher_draining`, `dipcatcher_kill_switch_blocking`,
and `dipcatcher_kill_switch_info`. Stage latency, error rate, and freshness
use the same names as `deploy/observability/slo.yml`. The alert rules in
`deploy/observability/alerts.yml` page when readiness is 0, drain is 1, the
kill file is blocking, p99 is above 2 seconds, the error ratio is at least
0.01, freshness is at least 86400 seconds, or the `lab-api` or
`research-api` scrape target is down. A non-zero error rate is a warning.
Tripping the kill file keeps the process up, so
`DipcatcherKillSwitchBlocking` and `DipcatcherNotReady` fire. A full data
directory makes the entrypoint exit before bind, the scrape target goes
down, and `DipcatcherTargetDown` fires. The metrics port stays unpublished.

## Resilience

Load either API with the stdlib runner. Each invocation stays at or under
120 seconds and 200 concurrency:

```bash
python scripts/http_load.py \
  --url http://127.0.0.1:8000/health --concurrency 50 --seconds 30
python scripts/http_load.py \
  --url http://127.0.0.1:8010/health --concurrency 50 --seconds 30
```

Set `DIPCATCHER_RATE_LIMIT_RPS` to a positive rate and repeat against
`/ready` with `--env-file` and `--key-env`. The key is not printed. Clients
over the limit receive 429 and `Retry-After`. That response is not a 5xx.

A directory that becomes read-only after start leaves the last complete
ledger file in place. The next lab start fails the write preflight. Paper
ledger writes use a same-directory temporary file, fsync, and replace.
SIGKILL during the write leaves the previous complete file. Torn names
start with a dot and are not archived.

```bash
python scripts/paper_state_backup.py backup --source /app/data --dest /backup/paper
python scripts/paper_state_backup.py restore --archive /backup/paper --dest /app/data
```

Restore checks every hash before it writes. A destination file with
different bytes is left unchanged.

## Logs

Service events and uvicorn logs are JSON lines on stderr. Docker keeps them
on the container's log stream:

```bash
docker logs dipcatcher
```

The process does not open a log file. Request volume does not grow files
under `/app`, `/tmp`, `/var/log`, or `/home/dipcatcher`. Docker records the
stderr stream on the host, outside the container writable layer. Values of
known secret environment variables are replaced with `[redacted]` before a
line is written. Do not add a new log field that copies a credential.

Startup emits `service_start`. SIGTERM emits `service_draining`, then
`service_shutdown`. With `DIPCATCHER_OBSERVE=1`, SLO and kill-switch
transitions emit `alert`.

## Crown-jewel edits in the ops series

These are the crown-jewel paths that differ from `afaee12d4` through the
ops commits. The pins are not re-signed here.

| File | PR | Why |
|---|---|---|
| `.dockerignore` | 3058 | Image context note includes the healthcheck and the shipped SLO file. |
| `.env.example` | 3058 | Document the lab bind, log, shutdown, kill-file, and SLO variables. |
| `.env.example` | 3064 | Document the loopback metrics scrape opt-out and the live-kill file. |
| `.env.example` | 3069 | Document paper halt cancel, harness drain debounce, and the data-dir write flag. |
| `.env.example` | 3079 | Document `DIPCATCHER_RATE_LIMIT_RPS`. |
| `Dockerfile` | 3058 | Non-root lab image, in-container healthcheck, SIGTERM, default bind port. |
| `Dockerfile` | 3064 | Install uv 0.11.23 with `pip --require-hashes`. |
| `Dockerfile` | 3069 | Entrypoint preflight and `DIPCATCHER_REQUIRE_DATA_WRITE=1`. |
| `mkdocs.yml` | 3058 | Publish this runbook in the docs nav. |

No other pinned crown-jewel path changed in that range. This slice does
not edit `.env.example` or `Dockerfile`.

## What this does not do

`/metrics` requires the service API key unless
`DIPCATCHER_METRICS_LOOPBACK_SCRAPE=1` and the metrics bind is loopback.
Do not publish `9464`. The image installs `uv` 0.11.23 with
`pip --require-hashes` from `docker/uv-requirements.txt`. Crown-jewel and
epoch pins are re-signed by the key holder after review. This runbook does
not authorize live trading.
