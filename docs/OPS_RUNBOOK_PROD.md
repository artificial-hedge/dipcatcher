# Production ops drills

These drills cover the compose stack in `deploy/compose.prod.yml`. The lab
API and the research API are research and simulated-paper surfaces. They do
not place live orders. A green check is not a live-trading claim.

Do not print `QUANT_API_KEY` or `RESEARCH_API_KEY`. Do not commit
certificates, private keys, or bearer tokens. Dev material stays under
`TLS_DIR` (default `/tmp/dipcatcher-ops7-tls`) and the data directories
(defaults `/tmp/dipcatcher-ops7-data/lab` and
`/tmp/dipcatcher-ops7-data/research`).

`docs/PRODUCTION_RUNBOOK.md` is the process contract. This file is the
operator sheet for a restart, a rollback, a certificate renewal, and a
replica failure.

Preflight, from the repo root. Export both API keys in the environment
first. Set `PROMETHEUS_PUBLISH_PORT` and `GRAFANA_PUBLISH_PORT` when `9090`
or `3000` is already taken. Set `SUDO=` on the command line when docker
does not need sudo.

```bash
make -f deploy/prod.mk prod-smoke
```

Stdout is one JSON object with `ok`, `up`, `down`, and `checks`. The exit
status is 0 only when `prod-up` exited 0, every check passed, and
`prod-down` exited 0. It is 1 otherwise. The checks are TLS `GET /health`
on both listeners (200 and HSTS `max-age=31536000`), a small POST that is
not 413, a POST of 65537 bytes that is 413, `write_owner_busy` from a
second claim in `lab-a`, `is not the writer` from a kill-file write in
`lab-b`, errno `EROFS` from a create on the research data mount, a request
id in the proxy log and a replica log for each API, and `rollback` after
`prod-rollback`. Make and docker logs go to stderr with the key values
redacted. If either key is missing, the object is
`{"error": "QUANT_API_KEY and RESEARCH_API_KEY are required", "ok": false}`
and `make` is not invoked. After a `prod-up` attempt the script always
attempts `prod-down`. `prod-up` runs `prod-preflight` first. A failed
preflight does not build images and does not start the stack.

## Preflight

```bash
make -f deploy/prod.mk prod-certs
make -f deploy/prod.mk prod-data
make -f deploy/prod.mk prod-preflight
```

Stdout is one JSON object with `ok` and `checks`. The exit status is 0
only when every check passed. The checks are the shell keys (present, one
token, values not printed), `dev.crt` and `dev.key` (key mode 600, cert
valid for at least a day), the four loopback publish ports free and
distinct, at least 64 MiB free on the lab, research, and certificate
directories, the compose write-owner layout (`lab-a` is `1`, the other
three replicas are `0`, research data is `:ro`), and the host clock
(year 2024–2035 and inside the certificate window). `9464` must not be
one of the publish ports.

When preflight is not root, the JSON also carries `warnings` for
unattended upgrades, the firewall, SSH key-only authentication, and time
sync. Each warning is `"not root"`. Those four do not change the exit
status. When preflight is root, the same four are checks and a failure
exits 1. `docs/HOST_HARDENING.md` is the host contract. `prod-up` does
not run preflight as root, so a deploy warns.

`deploy/prod.env` keeps `DIPCATCHER_RATE_LIMIT_RPS=50`. If that variable
is unset while `DIPCATCHER_PROFILE` is `production` or `prod`, the process
uses 50 rather than leaving the limit off. `deploy/prod.env.example`
lists every production variable, the default, and why. The root
`.env.example` does not.

## Restart

```bash
make -f deploy/prod.mk prod-restart
```

Success is exit 0. Each pair runs `scripts/http_load.py` at concurrency 10
for 110 seconds through TLS. The report for each pair has `transport` 0
and `http_5xx` 0.

The script rolls one replica at a time: `lab-a`, then `lab-b`, then the
research pair. SIGUSR1 sets the drain latch and leaves the socket open.
`/ready` becomes 503. `/health` stays 200. The script waits
`--settle-seconds` (5) so Caddy's one-second `/ready` check removes that
replica, then `docker stop` and `docker start`. It waits until `/ready` is
200 and settles again before the next replica. The peer keeps serving.
SIGTERM is what `docker stop` sends after the replica is out of rotation.
Do not roll both lab replicas in one step. Do not send SIGKILL during the
roll.

The ready probe is `docker exec` inside the replica. It reads the API key
from the container environment. The key is not an argument and the script
does not print it.

On exit 1, read the load report on stderr. A `transport` count above 0
means a client lost the connection. Leave the stack up and inspect
`docker ps` before a second roll. `prod-restart` does not run `prod-down`.

`lab-a` is the only writer. The flock is released when that process exits,
so the restarted `lab-a` can claim it again. Do not set
`DIPCATCHER_WRITE_OWNER=1` on `lab-b` while `lab-a` might still be running.
A second owner exits before bind with `write_owner_busy`.

## Secrets

`prod-up` writes the shell keys to `SECRET_DIR` (default
`/tmp/dipcatcher-ops7-secrets`) and mounts them as docker secrets. The
directory is mode `0700` and each file is mode `0644`. Compose bind-mounts
that inode and ignores a secret `mode`, and the API process is uid 10001,
so a mode `0600` file owned by the host user is not readable in the
container. The containers set `QUANT_API_KEY_FILE` and
`RESEARCH_API_KEY_FILE`. They do not receive the plain values. Under `DIPCATCHER_PROFILE=production` a
plain `QUANT_API_KEY`, `RESEARCH_API_KEY`, or
`DIPCATCHER_ALERT_WEBHOOK_SECRET` refuses the boot. The message names the
variable and does not print the value. The same refusal applies when the
plain value and the file are both set.

`DIPCATCHER_ALERT_WEBHOOK_URL` is optional. When it is set, the webhook
secret file is required. The URL is http or https with a host and no
userinfo. `prod-secrets` keeps an existing webhook token file when the
shell variable is unset, or writes a random token. It does not print the
token. Do not commit `SECRET_DIR`.

A secret file may hold two lines during rotation. Line 1 is what the proxy
and Prometheus load at start. Line 2 is the previous token. The API reads
the file on each request, so both tokens work without an API restart.
`prod-secrets` keeps those two lines when the shell token is already one
of them. A shell token that is on neither line replaces the file with that
one token. Do not run that replacement in the middle of an overlap. The
rotation section below is the procedure. `prod-rotate-drill` is the check.

## Request ids

The proxy forwards `X-Request-Id` to the replica. When the header is
absent it uses the request uuid. Access logs on the proxy are JSON on
stdout. Each API replica writes one JSON object per request, with
`request_id`, and echoes the id on the response. An id that is not a
single token of letters, digits, `.`, `_`, or `-` is replaced before it
is logged. `prod-smoke` sends one id to each listener and requires that
id in the proxy log and in one replica log.

## Service level objectives

`deploy/observability/slo_rules.yml` is a Prometheus rule file. It records
5 minute and 30 day availability, 5 minute p99 latency, and the fraction
of the 30 day error budget that remains. The availability objective is
99.9 percent of requests not answered with 5xx. The latency objective is
p99 under 0.25 seconds. `DipcatcherErrorBudgetBurn` pages when the fast
burn (14.4 times the budget, on both the 5 minute and 1 hour windows) or
the slow burn (6 times the budget, on both the 30 minute and 6 hour
windows) is true for 2 minutes. These are process objectives, not
research scores. `promtool test rules deploy/observability/slo_rules_test.yml`
checks the burn-rate alert.

## Rollback

`prod-build` tags the current `dipcatcher:local` image as
`dipcatcher:previous`, and the current `dipcatcher-research-api:latest`
image as `dipcatcher-research-api:previous`, before it builds. Roll back
with:

```bash
make -f deploy/prod.mk prod-rollback
```

The target tags those previous names back onto the running tags, recreates
the four API replicas and the proxy without building, and then requires
both TLS `/health` checks to be 200 with HSTS and the four replicas to be
healthy. Stdout of the gate is one JSON object. Exit 0 is healthy. Exit 1
leaves the new containers in place for inspection. Do not delete the lab
data directory. The research mount stays `:ro`.

`prod-smoke` points the previous tags at the image it just deployed and
then runs `prod-rollback`, so the drill recreates that image and checks
health. A rollback outside that drill uses the tags `prod-build` saved.

`make -f deploy/prod.mk prod-up` rebuilds both images from the working
tree. Do not use it to roll back. Tag the images you trust before you
replace them:

```bash
docker tag dipcatcher:local dipcatcher:known-good
docker tag dipcatcher-research-api:latest dipcatcher-research-api:known-good
```

Return to those tags without rebuilding. Keep the lab data directory. The
research host directory stays the same and stays mounted `:ro`. Do not
delete either directory. Do not edit receipts, `gate_pins.sig`, or corpus
epochs. Retract a bad research receipt with `dipcatcher tombstone`.

```bash
make -f deploy/prod.mk prod-down
docker tag dipcatcher:known-good dipcatcher:local
docker tag dipcatcher-research-api:known-good dipcatcher-research-api:latest
sudo --preserve-env=QUANT_API_KEY,RESEARCH_API_KEY,DIPCATCHER_LAB_DATA,DIPCATCHER_RESEARCH_DATA,DIPCATCHER_TLS_DIR,LAB_TLS_PORT,RESEARCH_TLS_PORT,PROMETHEUS_PUBLISH_PORT,GRAFANA_PUBLISH_PORT \
  docker compose -f deploy/compose.prod.yml up -d --no-build --remove-orphans
```

Export `DIPCATCHER_LAB_DATA`, `DIPCATCHER_RESEARCH_DATA`, and
`DIPCATCHER_TLS_DIR` to the paths the stack was using before `prod-down`.
The makefile defaults are the `/tmp/dipcatcher-ops7-*` paths above. Pass
`SUDO=` and call `docker compose` directly when sudo is not required. Do
not echo the API keys. `--no-build` is required: a build from the bad tree
overwrites the tags you just restored.

If `known-good` was never tagged, check out the previous commit in a
second work tree, build `dipcatcher:known-good` and
`dipcatcher-research-api:known-good` there, then retag and `up --no-build`
from the compose file you intend to run. Do not rebuild the bad tree.

## Certificate renewal

The proxy reads `dev.crt` and `dev.key` from `TLS_DIR`. `prod-certs`
mints a self-signed pair when either file is missing: CN `localhost`, SAN
`DNS:localhost` and `IP:127.0.0.1`, key mode 600. Neither file is in git.

Replace that pair with a certificate whose SAN covers the name clients
use. The key is unencrypted PEM so Caddy can read it. Mode 600 on the key.
The pinned `caddy:2.8.4` image runs as root, so a mode 600 key on the host
is readable there. Recreate only the proxy:

```bash
sudo --preserve-env=QUANT_API_KEY,RESEARCH_API_KEY,DIPCATCHER_LAB_DATA,DIPCATCHER_RESEARCH_DATA,DIPCATCHER_TLS_DIR,LAB_TLS_PORT,RESEARCH_TLS_PORT,PROMETHEUS_PUBLISH_PORT,GRAFANA_PUBLISH_PORT \
  docker compose -f deploy/compose.prod.yml up -d --no-build --force-recreate --no-deps proxy
curl --cacert "$DIPCATCHER_TLS_DIR/dev.crt" -sI "https://127.0.0.1:${LAB_TLS_PORT:-8443}/health"
curl --cacert "$DIPCATCHER_TLS_DIR/dev.crt" -sI "https://127.0.0.1:${RESEARCH_TLS_PORT:-8444}/health"
```

Both responses are 200 and include
`Strict-Transport-Security: max-age=31536000; includeSubDomains`. The
`Server` header is absent. A browser that accepts the certificate
remembers that host for a year. Turn HSTS on for a public name only when
that name is always served over HTTPS. Keys substituted into the Caddyfile
must be a single token: no spaces, quotes, braces, or `#`.

Do not commit the key. Delete `dev.crt` and `dev.key` and run
`make -f deploy/prod.mk prod-certs` to mint a fresh dev pair. `prod-up`
does that too when the files are missing.

## Replica failure

Stop one replica. The peer keeps serving.

```bash
docker stop dipcatcher-ops7-lab-b-1
curl --cacert "$DIPCATCHER_TLS_DIR/dev.crt" -sS -o /dev/null -w '%{http_code}\n' \
  "https://127.0.0.1:${LAB_TLS_PORT:-8443}/health"
```

The curl line prints `200` while `lab-a` is up. Prometheus
`DipcatcherTargetDown` is `up{job=~"lab-api|research-api"} == 0` for 1
minute. It fires for the stopped target. The peer target stays `up`.

```bash
docker start dipcatcher-ops7-lab-b-1
```

Wait until `/ready` inside that container is 200. The alert clears after
the next successful scrape. Repeat the same stop and start for
`dipcatcher-ops7-research-b-1` against port 8444. Research replicas are
availability only: the mount is `:ro` and both set
`DIPCATCHER_WRITE_OWNER=0`. They cannot create the lock file and they
cannot flip the kill file. Stop the replica or replace the mount to change
the research tree. Do not make that mount writable under the production
profile. The entrypoint exits before exec and the message says the
directory must be read-only.

`lab-a` is the only writer. Do not set `DIPCATCHER_WRITE_OWNER=1` on
`lab-b` while `lab-a` might return. The flock releases when the `lab-a`
process dies, so a restarted `lab-a` can claim it. A second live owner
exits with `write_owner_busy`. Under the production profile an unset
`DIPCATCHER_WRITE_OWNER` refuses ledger and kill-file writes and refuses
to boot with `write_owner_required`. Dev and test leave it unset and still
write. Do not unset it on a production replica to clear a lock error.

If the proxy container is down, clients see transport errors while the
replicas are healthy. Start the proxy and check both TLS listeners:

```bash
docker start dipcatcher-ops7-proxy-1
```

Container names follow `PROJECT` (default `dipcatcher-ops7`) plus
`-<service>-1`.

## Backup and restore

The lab data directory holds the paper ledger and the kill file. Research
data is mounted read-only and is not the writer. Stop `lab-a` before a
backup so the owner is not replacing a file. The archive stores hashes.
It includes `metadata/paper/`, `checkpoints/`, and `kill_switch` when that
file exists. A missing kill file is `ENABLED` and is not an error. Dotfiles
and a symlink are refused.

```bash
docker stop dipcatcher-ops7-lab-a-1
make -f deploy/prod.mk prod-backup
```

The archive lands in `BACKUP_DIR` (default `/tmp/dipcatcher-ops7-backup`).
Restore checks every hash before it writes. A destination file whose bytes
differ is left unchanged and the command exits 1. Byte-identical files are
left in place. Restore writes as root when `SUDO` is `sudo`, because the
data directory is owned by uid 10001.

Drill, on a copy, before pointing restore at the live directory:

```bash
make -f deploy/prod.mk LAB_DATA=/tmp/dipcatcher-drill/lab BACKUP_DIR=/tmp/dipcatcher-drill/archive prod-backup
rm -rf /tmp/dipcatcher-drill/lab
mkdir -p /tmp/dipcatcher-drill/lab
make -f deploy/prod.mk LAB_DATA=/tmp/dipcatcher-drill/lab BACKUP_DIR=/tmp/dipcatcher-drill/archive prod-restore
```

Compare the restored `kill_switch` and `metadata/paper/` bytes with the
archive payload. Do not edit receipts. Start `lab-a` again after the live
directory is the one you intend to run. This is a simulated-paper restore.
It does not place a live order.

## Disaster recovery

```bash
make -f deploy/prod.mk prod-dr-drill
```

The drill stops the stack, writes a SYNTHETIC `broker_state.json` under
`metadata/paper/dr-drill/` when the lab directory has none, and runs
`prod-backup`. It then deletes `DIPCATCHER_LAB_DATA` and
`DIPCATCHER_RESEARCH_DATA`, runs `prod-up`, runs `prod-restore`, and
hashes the restored files. A path is deleted only when it resolves under
`/tmp` and a component starts with `dipcatcher-`. These are the
bind-mounted data directories. The compose file has no named volume to
remove.

`rto_seconds` is the wall time from the start of that delete through a
matching ledger digest and both TLS `/health` checks at 200 with HSTS
`max-age=31536000`. The stop before the backup is not included. The
ledger digest is the sha256 of the one `broker_state.json`. When the
archive has more than one, it is the sha256 of the sorted `path digest`
lines. The drill exits 0 only when every check passed, the digest
matches, and the final `prod-down` exited 0. Stdout is one JSON object.
It does not print API keys. It does not place an order.

Measured RTO on this host: 16.033 seconds. The phases were destroy
0.007 seconds, `prod-up` 13.839 seconds, restore 2.144 seconds, and TLS
health 0.044 seconds. `prod-up` reused cached image layers from the smoke
run on the same host, and that boot also started the unpublished
node-exporter. The restored ledger digest was
`08e1c3bc35d053fb228ef8ce4ef1caf2fff3114d9aae99848fc6f0b3c688263e`.
`rto_includes` is `destroy, prod-up, restore, ledger digest, TLS health`.

## Epoch drift

```bash
uv run python scripts/owner_restamp.py
```

Stdout is one JSON object. `drifted` lists which of `deploy`, `docs`,
`docker`, `scripts`, `src`, and `tests` differ from the tree root in
`quality/epoch_heads.json`. `owner_steps` is the key-holder order:

```bash
make stamp-epochs
make sign-pins
make anchor-pins
make checkpoint
```

The script is read-only. It does not stamp or sign. `signed` is false.
Do not run the four commands to make a local tree look clean.

## Kill-switch receipt

`docs/KILL_AUDIT_RECONCILIATION.md` records why
`receipts/kill_audit.json` says 57 checks and the unit test expects 63,
which checks were added and which three unknown-`FLATTEN` checks were
removed, and the owner command that reseals the receipt. Do not edit
`receipts/` in a deploy slice. The receipt stays sealed at 57 until that
command runs.

## Log retention

App and proxy processes keep writing to stderr and stdout. Compose caps
those streams with the json-file driver (`DIPCATCHER_LOG_MAX_SIZE`,
default `10m`, and `DIPCATCHER_LOG_MAX_FILE`, default `5`). json-file has
no age cap. `prod-log-retain` deletes rotated `*-json.log.N` files that
are older than the age cap or past the file cap, and it does not rename
the live file the daemon still has open.

The same command rotates regular files under `LOG_DIR/app`,
`LOG_DIR/proxy`, and `LOG_DIR/audit`. The defaults are 10 MiB, 7 days,
and 5 files (`DIPCATCHER_LOG_MAX_BYTES`, `DIPCATCHER_LOG_MAX_AGE_S`,
`DIPCATCHER_LOG_MAX_FILES`). A stream can override them with
`DIPCATCHER_APP_LOG_*`, `DIPCATCHER_PROXY_LOG_*`, or
`DIPCATCHER_AUDIT_LOG_*`. Non-decimal values and values below 1 make the
command exit 1. The hash-chained ledger names `entries.jsonl` and
`checkpoints.jsonl`, including numbered copies, are never renamed or
deleted.

Each replica appends its own audit file when
`DIPCATCHER_AUDIT_LOG_PATH` is an absolute path inside the container.
The production compose file sets `/logs/audit/<replica>.jsonl`. Unset,
the service stays stderr-only. A symlink is ignored. The in-process
size cap is `DIPCATCHER_AUDIT_LOG_MAX_BYTES` (default 10485760) and
`DIPCATCHER_AUDIT_LOG_MAX_FILES` (default 5). Age is the retain script.

```bash
make -f deploy/prod.mk prod-log-retain
```

`DipcatcherDiskPressure` pages when
`node_filesystem_avail_bytes / node_filesystem_size_bytes` stays under
`0.15` for 5 minutes on a writable filesystem. `0.15` is the knob in
`deploy/observability/alerts.yml`. Prometheus does not substitute an
environment variable there. The series comes from the unpublished
`node-exporter` service. If that process is down, the series is absent
and the rule stays silent.

## Off-host backup

`prod-backup` still writes one archive at `BACKUP_DIR`. `prod-backup-ship`
writes a new directory `BACKUP_ROOT/<UTC stamp>`, rehashes every payload
against `manifest.json`, and only then copies. `BACKUP_ROOT` defaults to
`/tmp/dipcatcher-ops7-backups`. A root that already contains
`manifest.json` is refused so a stamp is not nested inside the
`prod-backup` archive. Previous stamps are kept. A torn read fails the
hash check, skips the copy, and exits 1. The timer does not stop
`lab-a`. Stop it yourself when you need a quiet writer:

```bash
docker stop dipcatcher-ops7-lab-a-1
make -f deploy/prod.mk prod-backup-ship
```

`DIPCATCHER_BACKUP_COPY` is `none` (verify only), `rsync`, or `s3`.
The rsync target is a local absolute path or `user@host:/absolute`.
The s3 target is `s3://bucket/prefix`. Use an instance role or
`aws configure`. Do not put AWS keys in `backup.env`. Paths must stay
under `/tmp/dipcatcher*`, `/var/lib/dipcatcher*`, `/var/log/dipcatcher*`,
or `/var/backups/`.

```bash
sudo install -d -m 0700 /etc/dipcatcher
sudo tee /etc/dipcatcher/backup.env >/dev/null <<'EOF'
DIPCATCHER_BACKUP_COPY=rsync
DIPCATCHER_BACKUP_RSYNC_TARGET=backup@backup.example:/var/backups/dipcatcher
EOF
sudo chmod 600 /etc/dipcatcher/backup.env
sudo cp deploy/systemd/dipcatcher-prod-backup.service /etc/systemd/system/
sudo cp deploy/systemd/dipcatcher-prod-backup.timer /etc/systemd/system/
sudo cp deploy/systemd/dipcatcher-log-retain.service /etc/systemd/system/
sudo cp deploy/systemd/dipcatcher-log-retain.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now dipcatcher-prod-backup.timer dipcatcher-log-retain.timer
```

The backup timer is `OnCalendar=*-*-* 03:15:00` with `Persistent=true`.
The log timer is hourly. `deploy/cron/` has the same two schedules for
hosts that use cron instead of systemd. Both units run
`/opt/dipcatcher/.venv/bin/python`. They do not call `uv`.

## Canary

`prod-build` tags the running image as `dipcatcher:previous` and then
builds `dipcatcher:local`. Containers that are already up keep the old
image until they are recreated. `prod-canary` does not build. It starts
`lab-canary` with the canary profile, waits until that container is
healthy, and probes `http://127.0.0.1:8000/health` from inside it: GET
200, a small POST below 500 and not 413, and a 65537-byte POST at 413.
The canary is one lab reader (`DIPCATCHER_WRITE_OWNER=0`). It is not a
Caddy upstream.

A passing probe recreates `lab-a`, `lab-b`, `research-a`, `research-b`,
and `proxy`, then removes the canary. A failing probe removes the canary.
Exit 0 means the roll happened. Exit 1 means abort, or the roll command
failed. Set `DIPCATCHER_AUTO_ROLLBACK=1` to retag `dipcatcher:previous`
and `dipcatcher-research-api:previous` back onto the running tags and
recreate the fleet when the probe fails. `prod-rollback-drill` sets that
and measures the time through the health gate. `prod-smoke` is a separate
drill and it always stops the stack. Do not point the canary at it.

```bash
make -f deploy/prod.mk prod-build
make -f deploy/prod.mk prod-canary
```

## Alert drill

`prod-alert-drill` starts Prometheus and Alertmanager on the loopback
ports `19090` and `19093`. It does not start the lab stack and it does
not page an operator. A local webhook records each notification. The
rules are the files in `deploy/observability/alerts.yml` and
`slo_rules.yml`, including the real `for` durations. The longest `for`
is 5 minutes, so the drill waits through firing and through resolve.

The fixture publishes one series per alert. Gauges cover readiness,
drain, the kill switch, stale data, reconcile drift, backup age, canary
abort, and disk pressure. Counters cover stage errors, stage latency,
HTTP errors, and HTTP latency. A second scrape target stays closed until
resolve, which is `DipcatcherTargetDown`. After every name has fired,
the drill waits 20 seconds and requires no further webhook post. It then
publishes the healthy gauges, stops moving the counters, and opens the
down target. Each fingerprint must show one firing notification and a
later resolved notification.

```bash
make -f deploy/prod.mk prod-alert-drill
```

Textfile gauges are `dipcatcher_reconcile_drift`,
`dipcatcher_backup_last_success_timestamp`, and
`dipcatcher_canary_abort`. `prod-backup-ship` writes the backup stamp
after a verified copy. `prod-canary` writes abort `1` when the roll does
not finish and `0` when it does. Reconcile is not published on a timer:
`python scripts/ops_signals.py reconcile` reads the same six corpora as
`owner_restamp` and does not stamp or sign. `refresh` republishes that
gauge plus the backup stamp and the canary state. `prod-up` runs it, and
`dipcatcher-ops-signals.timer` runs it every five minutes, so the three
series are present. A missing backup stamp stays 0, which pages
`DipcatcherBackupAge` until a ship writes a unix time. Node-exporter
reads the directory and publishes no port of its own. Each alert's
`runbook` annotation points at `docs/incidents/`.

Measured on this host: 15 alerts, 15 fingerprints, 30 webhook posts.
Each fingerprint was delivered once while firing and once when it
resolved. A 20 second quiet interval added no post. Deduped and
resolved were both true. The drill took 655.973 seconds.

## Secret rotation

Rotate the API keys, the broker pair, and the webhook HMAC without
taking the API down. Use the files and env names that already exist.
Do not add names, and do not edit `.env.example`.

API and webhook files:

- `DIPCATCHER_QUANT_API_KEY_FILE` (container path `QUANT_API_KEY_FILE`)
- `DIPCATCHER_RESEARCH_API_KEY_FILE`
- `DIPCATCHER_ALERT_WEBHOOK_SECRET_FILE`

Put the new token in the shell variable of the same name and run
overlap. Line 1 stays the current token. Line 2 becomes the new token.
Clients on either token get through the API. A third token is rejected.
The webhook check tries every line and does not stop at the first match.
The signature is `sha256=<hex>` over `timestamp.body`, with a five
minute window.

```bash
make -f deploy/prod.mk prod-rotate-drill
python scripts/prod_rotate.py overlap
python scripts/prod_rotate.py promote
python scripts/prod_rotate.py retire
```

`promote` swaps the lines so a restarted proxy or Prometheus loads the
new token. `retire` keeps line 1. Overlap refuses to run when
`QUANT_API_KEY` or `RESEARCH_API_KEY` is empty. The webhook file is
included when it already exists. Overlap does not print the values.

The proxy and Prometheus read line 1 into the process at start. After
promote, recreate only those two so they load the new line. Leave the
API replicas up. One proxy recreate drops the connections that were on
that process. It does not restart the API.

```bash
sudo --preserve-env docker compose -f deploy/compose.prod.yml up -d \
  --no-build --no-deps --force-recreate proxy prometheus
python scripts/prod_rotate.py retire
```

Before the next `prod-secrets`, `prod-up`, or `prod-down`, export the
token that remains on line 1. Those targets call `prod-secrets`. During
overlap it keeps both lines. After retire a shell token that is not the
remaining line replaces the file.

Broker overlap uses `ALPACA_API_KEY_ID` and `ALPACA_API_SECRET_KEY` as
the current pair and `APCA_API_KEY_ID` and `APCA_API_SECRET_KEY` as the
previous pair. There is no broker file. A mixed pair, one key from one
name and the secret from the other, is refused. The session reads both
pairs on each call. A 401 or 403 from the current pair is retried once
with the previous pair. A 200 is not retried. `alpaca-live` is still
refused before any HTTP. The lab containers are not a broker. A process
does not see environment edits made from outside it. Export both pairs
before the gateway starts, or change them inside that process. Drop
`APCA_*` to retire the previous pair. The next call sees the change.

```bash
make -f deploy/prod.mk prod-rotate-drill
```

Measured on this host: during overlap the previous and new API tokens
and both webhook signatures were accepted, and the broker exposed two
pairs. After retire the previous API token was rejected, the previous
webhook signature no longer verified, and one broker pair remained.
122 requests during the overlap write all returned 200. `gap` was false.

## Capacity soak

`prod-soak` holds `https://127.0.0.1:$LAB_TLS_PORT/health` for 1200
seconds at concurrency 20. `/health` is outside the rate limit. The
script samples `lab-a` RSS and open file descriptors every 30 seconds.
After a 120 second warmup, a file-descriptor slope above 0.05 per second
or an RSS slope above 20 KiB per second is a leak and the command exits
1. The JSON reports `p50_ms`, `p99_ms`, and `error_rate`. Set
`DIPCATCHER_SOAK_SECONDS` only to shorten a rerun. The default stays
1200, and a value above 1200 is refused.

If `/health` is down, the script runs `prod-up` and, only in that case,
`prod-down` afterwards. A stack that was already up is left up.

```bash
make -f deploy/prod.mk prod-soak
```

Measured on this host: 1200 seconds at concurrency 20, 2158990 requests,
error rate 0. p50 was 10.164 ms and p99 was 31.884 ms. After the 120
second warmup, lab-a RSS slope was -50.420 KiB/s and the open-descriptor
slope was -0.000487 per second across 40 samples. `leak` was false.

## Rollback drill

`prod-rollback-drill` records the running image ids and the ledger digest,
tags those images as previous, then builds a `/tmp` image whose entrypoint
is `/bin/false`. That image is tagged as N+1. The canary never becomes
healthy. With `DIPCATCHER_AUTO_ROLLBACK=1` the previous tags are restored
and the fleet is recreated. Time-to-rollback includes that recreate and
the health gate. The seeded `broker_state.json` digest must be unchanged.

```bash
make -f deploy/prod.mk prod-rollback-drill
```

Measured on this host: 14.273 seconds from the start of the retag through
the health gate. The canary action was abort, `rollback_ok` was true, and
the lab and research tags and the four API containers were back on the
captured image ids. The ledger digest was
`3d70cdcc0e854a49f1c350ad484e05356b372d8d0149a1551c6be9753de892e0` and it
did not change.

## Signal drill

`prod-signal-drill` writes reconcile drift 1, backup timestamp 0, and
canary abort 1 through `ops_signals.publish`, then runs `promtool test
rules` on `alerts.yml` inside the pinned Prometheus image. The three
alerts must be firing at 26 hours of evaluation time, which covers the
5 minute backup-age `for` and `time() - 0 > 86400`.

```bash
make -f deploy/prod.mk prod-signal-drill
```

Measured on this host: promtool exit 0. The published gauges were
reconcile drift 1, backup timestamp 0, and canary abort 1.
DipcatcherReconcileDrift, DipcatcherBackupAge, and DipcatcherCanaryAbort
were firing.

## Order-path load

`prod-order-load` reads `DIPCATCHER_RATE_LIMIT_RPS` from `deploy/prod.env`
and drives 10 times that many decisions. The paper book must refuse an
over-notional order and, once the kill file is `HALT_NEW_ORDERS`, refuse
every later order. The loopback broker must refuse the over-notional
order, keep the signed 20-order window, and accept nothing after the
kill latch. The result is simulated paper.

```bash
make -f deploy/prod.mk prod-order-load
```

Measured on this host: the production rate in `deploy/prod.env` is 50
requests/s, so the burst is 500 decisions. The paper book refused the
over-notional order (`max_order_notional`) and rejected that path at
418130.823 decisions/s over 0.001196 seconds, with zero accepted. After
`HALT_NEW_ORDERS`, accepted orders were 0. The loopback broker refused
the over-notional order, allowed 8 decisions inside the signed window,
denied 481 for the order rate, accepted 0 after the kill latch, and held
the position. Claims were 0. The evidence label is simulated-paper.

## Redaction audit

`prod-redaction-audit` runs the rotation drill and the 1200 second soak,
then scans the log directory, metrics text, and Prometheus alert payload.
A configured secret, an email address, an SSN-shaped number, a bearer
token, or an account-style metrics label fails the command. The finding
records the kind and the path.

```bash
make -f deploy/prod.mk prod-redaction-audit
```

Measured on this host: rotation exit 0 and soak exit 0. The soak was
1200 seconds at concurrency 20, 1722954 requests, error rate 0. p50 was
13.448 ms and p99 was 39.367 ms. After the warmup, lab-a RSS slope was
10.180 KiB/s and the open-descriptor slope was -0.001318 per second
across 40 samples. `leak` was false. The scan of the log directory, the
metrics text, the alert payload, and the drill output had no findings.
`alerts.json`, `metrics-lab-a.prom`, and `metrics-research-a.prom` were
present.

Incident pages for every alert are linked from the rule annotations and
listed in `docs/incidents/README.md`.
