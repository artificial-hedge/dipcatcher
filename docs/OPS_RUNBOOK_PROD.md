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
