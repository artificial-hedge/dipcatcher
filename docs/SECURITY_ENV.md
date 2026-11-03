# Security environment variables

Names introduced from security-hardening2 through security-hardening5
(`0a3fff4de` .. `c3849128d`). `.env.example` is hash-pinned and does not
list them. Copy the block at the bottom into `.env.example` when the owner
runs `make stamp-epochs`, then `make sign-pins`, then `make anchor-pins`.

A production value is what the process should see. `unset` means leave the
variable out. Do not put key material in `.env.example` or in this file.
`0`, `off`, `false`, `no`, or an empty value disables a rate pair or the
sensitive-read flag, as named below.

The anchors path is not an environment variable. `scripts/anchor_audit_head.py`
and `scripts/prod_backup.py` take it as `--anchors`.

## Read keys (security-hardening2)

| Variable | Production value |
|---|---|
| `QUANT_API_READ_KEY` | unset |
| `RESEARCH_API_READ_KEY` | unset |

Unset means there is no read credential. The admin key authorizes every
method. Set a secret distinct from the admin key only for a client that
may call `GET`, `HEAD`, and `OPTIONS`. Any other method with the read key
is 403 `read key cannot modify`. The read key does not satisfy the CLI
bind guard.

## Key files and grace (security-hardening3)

For each of `QUANT_API_KEY`, `QUANT_API_READ_KEY`, `RESEARCH_API_KEY`, and
`RESEARCH_API_READ_KEY`:

| Variable | Production value |
|---|---|
| `{NAME}_FILE` | unset |
| `{NAME}_PREVIOUS_FILE` | unset |
| `{NAME}_GRACE_UNTIL_FILE` | unset |

Unset `{NAME}_FILE` means the environment value of `{NAME}` is the current
secret. `deploy/compose.hardened.yml` still injects `QUANT_API_KEY` and
`RESEARCH_API_KEY` and does not mount these files. While `{NAME}_FILE` is
set, the environment value of `{NAME}` is ignored. The file is one UTF-8
line, mode `0600`. A missing, empty, multi-line, or unreadable current file
stays configured and matches nothing.

`{NAME}_PREVIOUS_FILE` is the retiring secret, same file shape. It is
accepted only while the clock is strictly before the timestamp in
`{NAME}_GRACE_UNTIL_FILE`. That timestamp is one ISO-8601 line with a
timezone, for example `2026-11-03T12:00:00Z`. An unreadable deadline closes
the window. Steady-state production leaves all three unset.

## Loopback rate (security-hardening4)

| Variable | Production value |
|---|---|
| `QUANT_API_LOOPBACK_RATE_PER_SEC` | `100` |
| `QUANT_API_LOOPBACK_RATE_BURST` | `1000` |
| `RESEARCH_API_LOOPBACK_RATE_PER_SEC` | `100` |
| `RESEARCH_API_LOOPBACK_RATE_BURST` | `1000` |

These are the code defaults, so an unset variable has the same effect.
`127.0.0.1`, `::1`, `localhost`, and the ASGI peer `testclient` use this
pair. A published Docker port is a remote peer and uses the older remote
pair below. Keep the loopback pair at these values in production. `/health`
is not counted.

## Audit log (security-hardening3, rotation in 4, reads in 5)

| Variable | Production value |
|---|---|
| `QUANT_API_AUDIT_LOG` | a mode-`0600` path on the durable volume |
| `RESEARCH_API_AUDIT_LOG` | a mode-`0600` path on the durable volume |
| `QUANT_API_AUDIT_LOG_MAX_BYTES` | `1048576` |
| `RESEARCH_API_AUDIT_LOG_MAX_BYTES` | `1048576` |
| `QUANT_API_AUDIT_READS` | `on` |
| `RESEARCH_API_AUDIT_READS` | `on` |

The code default for the log path is unset, which emits the same JSON
through the audit logger in `quant_fund.api.admin_audit` and writes no file. Production
that wants a chain sets both paths. The process creates the file, appends,
fsyncs, and chmods it to `0600`. A path that cannot be opened turns an
admin mutation, a denied attempt, and a sensitive read into 503
`audit log unavailable` before the route runs.

`1048576` is the code default for rotation. `off` disables rotation. When
the next line would pass the cap, the active file becomes `NAME.1`.

`on` is the code default for sensitive reads (`GET` and `HEAD` of
`/backtest/{id}`, and `GET` of `/risk/portfolio`, `/portfolio/target`, and
`/portfolio/exposures`). The research API has no such route, so its flag
changes nothing until one exists. `0`, `off`, `false`, or `no` disables
the read lines.

## Shared rate store (security-hardening5)

| Variable | Production value |
|---|---|
| `DIPCATCHER_RATE_BACKEND` | `memory` for one process; `sqlite` when several workers serve the API |
| `DIPCATCHER_RATE_STORE` | unset with `memory`; a path on the shared volume with `sqlite` or `file` |

`memory` is the code default. It keeps the bucket in the process and
refills from the monotonic clock. `sqlite` and `file` refill from the wall
clock. `file` takes an exclusive `fcntl` lock. A missing path, an unknown
backend, or a store that cannot be opened is 503 `rate store unavailable`.
Choose `sqlite` for more than one worker. Use `file` on a local POSIX disk.

## Remote rate pair, already present before hardening2

These four are not new in hardening2–5. `.env.example` does not list them
either. Production values are the code defaults:

| Variable | Production value |
|---|---|
| `QUANT_API_RATE_PER_SEC` | `10` |
| `QUANT_API_RATE_BURST` | `30` |
| `RESEARCH_API_RATE_PER_SEC` | `10` |
| `RESEARCH_API_RATE_BURST` | `30` |

A scanner may set the per-second variables to `off` in an uncommitted
override. `deploy/compose.hardened.yml` does not pass `off`.

## Block for `.env.example`

```text
# ── Lab and research API security (hardening2–5) ────────────────────────
# [optional] Read keys. Unset means the admin key is the only credential.
# QUANT_API_READ_KEY=
# RESEARCH_API_READ_KEY=
# [optional] Current secret file. While set, the env value of the key is ignored.
# One UTF-8 line, mode 0600. Steady state leaves these unset.
# QUANT_API_KEY_FILE=
# QUANT_API_READ_KEY_FILE=
# RESEARCH_API_KEY_FILE=
# RESEARCH_API_READ_KEY_FILE=
# [optional] Retiring secret and the grace deadline (ISO-8601 with timezone).
# Accepted only while now is strictly before the deadline.
# QUANT_API_KEY_PREVIOUS_FILE=
# QUANT_API_KEY_GRACE_UNTIL_FILE=
# QUANT_API_READ_KEY_PREVIOUS_FILE=
# QUANT_API_READ_KEY_GRACE_UNTIL_FILE=
# RESEARCH_API_KEY_PREVIOUS_FILE=
# RESEARCH_API_KEY_GRACE_UNTIL_FILE=
# RESEARCH_API_READ_KEY_PREVIOUS_FILE=
# RESEARCH_API_READ_KEY_GRACE_UNTIL_FILE=
# [optional] Loopback token bucket. Defaults are 100/s and burst 1000.
QUANT_API_LOOPBACK_RATE_PER_SEC=100
QUANT_API_LOOPBACK_RATE_BURST=1000
RESEARCH_API_LOOPBACK_RATE_PER_SEC=100
RESEARCH_API_LOOPBACK_RATE_BURST=1000
# [optional] Remote peers. Defaults are 10/s and burst 30. A scanner may use off.
QUANT_API_RATE_PER_SEC=10
QUANT_API_RATE_BURST=30
RESEARCH_API_RATE_PER_SEC=10
RESEARCH_API_RATE_BURST=30
# [conditional] Durable audit chain. Unset keeps lines on the audit logger only.
# QUANT_API_AUDIT_LOG=
# RESEARCH_API_AUDIT_LOG=
# [optional] Rotate at this size. off disables rotation. Default 1048576.
QUANT_API_AUDIT_LOG_MAX_BYTES=1048576
RESEARCH_API_AUDIT_LOG_MAX_BYTES=1048576
# [optional] Audit ledger and position reads. on is the default. off disables them.
QUANT_API_AUDIT_READS=on
RESEARCH_API_AUDIT_READS=on
# [optional] memory is one process. sqlite shares DIPCATCHER_RATE_STORE.
DIPCATCHER_RATE_BACKEND=memory
# DIPCATCHER_RATE_STORE=
```
