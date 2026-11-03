# HTTP threat model

Scope: the lab forecast API (`dipcatcher api`, `quant_fund.api.app`) and the
read-only research evidence API (`quant_fund.api.research_api`). The fx-1
harness service (`src/fx1/serve/api.py`) is a separate process. This document
records its current posture and the pending change. That file is not modified
here.

Research payloads from these APIs stay research-only. They are not live P&L
claims. `quant_fund.api.app._stamp_research_honesty` overwrites
`research_only` and `live_pnl_claim` on metric-bearing responses.

## Assets

- Operator API keys in the process environment and, when set, in the
  mode-`0600` files named by `QUANT_API_KEY_FILE`, `QUANT_API_READ_KEY_FILE`,
  `RESEARCH_API_KEY_FILE`, `RESEARCH_API_READ_KEY_FILE`, and the matching
  `*_PREVIOUS_FILE` values. The harness key remains `FX1_API_KEY`. The
  request path sees a key only as an `X-API-Key` header. Bucket ids and
  audit key ids store a SHA-256 digest, not the key.
- The append-only admin audit log (`QUANT_API_AUDIT_LOG` and
  `RESEARCH_API_AUDIT_LOG`). A line holds time, route, key id, request id,
  outcome, kind, and the SHA-256 of the previous line. It does not hold the key.
- Research evidence on disk: `receipts/`, `verifier/`, `artifacts/`, and the
  run notebooks under the research data root. The research API is read-only.
- Lab configs under `configs/`. `resolve_allowed_config_path` refuses a path
  that does not resolve inside that directory.
- Forecast, ranking, regime, risk, and backtest responses. These are research
  outputs. A successful risk or backtest response can still name a local file
  (see residual risks).
- Signing keys (`GATE_SIGNING_KEY` and the other keys in `docs/PRODUCTION.md`)
  are not request credentials for these two APIs and are not read by their
  middleware.

## Trust boundaries

| Boundary | What is trusted | What is not |
|---|---|---|
| Direct peer | `request.client.host` as the immediate TCP peer | `X-Forwarded-For`, `Forwarded`, and any client-supplied host header |
| Loopback | `127.0.0.1`, `::1`, `localhost`, and the ASGI test peer `testclient` | `::ffff:127.0.0.1` and every other address, including IPv4-mapped loopback |
| Environment and key files | The current file, or the env value when no file is set, read on that request. A previous file only before its grace deadline | A key copied into the app at import, for the env-backed apps. An env value is ignored while `{NAME}_FILE` is set |
| Filesystem | Paths that stay inside the configured root after resolve | User path segments that escape the root (`research_api._contained`, `resolve_allowed_config_path`) |
| Reverse proxy | Nothing, until a key is set | A proxy on the same host makes every browser look like loopback |

`X-Forwarded-For` is not read. A reverse proxy bound to loopback collapses
every client into the loopback peer: the no-key fallback allows them, and
they share the higher loopback bucket. Set the API key before placing either
service behind a proxy.

## STRIDE

Each row names the control and the test or gate that exercises it. A gap is
a threat this tree does not close.

### Direct peer

| Threat | Control | Covered by |
|---|---|---|
| Spoofing | The peer is `request.client.host`. `X-Forwarded-For` is not a peer. | `test_forwarded_for_does_not_become_loopback` |
| Tampering | 64 KiB body cap, JSON depth cap, header checks, config path allowlist, research id checks | `tests/unit/api/test_http_controls.py`, `tests/unit/security/test_pentest_surface.py`, `tests/unit/api/test_research_api.py`, `tests/unit/security/test_auth_rate_fuzz.py` |
| Repudiation | An admin-key unsafe request appends one JSON line. A rejected key on an unsafe method is outcome `denied`. A ledger or position read is outcome `read` | `tests/unit/security/test_admin_audit.py`. Other safe methods are not lines. That remains a gap |
| Information disclosure | Unhandled errors are `{"detail":"internal error"}`. Package version is on `GET /health` only | `test_lab_internal_error_hides_the_exception`, `test_health_is_the_only_lab_version_surface` |
| Denial of service | Per-IP and per-key token buckets, body cap, 128-header cap. `sqlite` and `file` share one store across workers | `tests/unit/api/test_http_controls.py`, `tests/unit/api/test_rate_store.py` |
| Elevation of privilege | Admin key, read key, wrong key, and no key have a fixed status on every route | `tests/unit/security/test_authz_matrix.py` |

### Loopback

| Threat | Control | Covered by |
|---|---|---|
| Spoofing | Loopback is `127.0.0.1`, `::1`, `localhost`, and `testclient`. `::ffff:127.0.0.1` is remote | `test_ipv4_mapped_loopback_is_a_remote_peer`, `middleware_audit` |
| Tampering | Same body and path controls as any peer | The peer tests above |
| Repudiation | A loopback request with no admin key does not write an audit line for an unsafe method. A sensitive read still appends outcome `read` when the reads flag is on | `test_admin_audit` (the unauthenticated mutation is absent), `test_loopback_without_a_key_audits_a_sensitive_read` |
| Information disclosure | `/health` on loopback returns the package version | `test_health_is_the_only_lab_version_surface` |
| Denial of service | Loopback uses `{PREFIX}_LOOPBACK_RATE_PER_SEC` (default 100) and `{PREFIX}_LOOPBACK_RATE_BURST` (default 1000). `/health` stays unlimited | `test_loopback_uses_its_own_higher_limit` |
| Elevation of privilege | With no key configured, loopback is allowed and every other peer is 403 | `middleware_audit` |

### Key files and environment

| Threat | Control | Covered by |
|---|---|---|
| Spoofing | A previous key works only before `*_GRACE_UNTIL_FILE`. After that instant it is 401 | `test_previous_admin_key_works_only_inside_the_grace_window`, `test_grace_boundary_is_strict` |
| Tampering | `{NAME}_FILE` replaces the env value on the next request. A missing or oversized file stays configured and matches nothing | `test_file_replaces_the_env_value_without_a_restart`, `test_unreadable_key_file_fails_closed`, `test_oversized_key_file_matches_nothing` |
| Repudiation | The audit key id is `sha256` of the presented key. Each line's `prev` is the SHA-256 of the previous raw line | `test_admin_mutation_is_appended_and_the_key_stays_out_of_the_log`, `tests/unit/security/test_audit_chain.py` |
| Information disclosure | The key is absent from the audit line, including when `X-Request-ID` is the key, and from the read-key 403 body | `test_admin_audit`, `test_read_key_cannot_modify` |
| Denial of service | A key file over 4096 bytes is not loaded | `test_oversized_key_file_matches_nothing` |
| Elevation of privilege | The read key, current or previous, cannot call an unsafe method. A previous admin key can | `test_previous_read_key_is_safe_methods_only`, `test_read_key_cannot_modify` |

### Filesystem

| Threat | Control | Covered by |
|---|---|---|
| Spoofing | Config paths must resolve under `configs/`. Research ids must match a closed expression and stay under the evidence root | `test_lab_config_path_traversal`, research path tests in `test_research_api.py` |
| Tampering | Backtest artifact paths cannot leave the artifact root | `get_backtest` path checks. Gap: no separate fuzz corpus beyond `test_pentest_surface.py` |
| Repudiation | `POST /backtest` is kind `ledger`. `GET /backtest/{id}` is kind `ledger` with outcome `read`. Position reads are kind `positions` with outcome `read`. `POST /portfolio/optimize` is kind `optimize`. `dump_resolved`, `PaperLedger.flush`, `LiveDesk.kill`, and the gateway `kill` append `config`, `ledger`, and `kill-file`. With no HTTP actor the route is `cli:<user>@<host>` | `test_process_mutations_bind_a_cli_actor`, `test_portfolio_optimize_kind_is_optimize`, `test_sensitive_reads_append_outcome_read` |
| Information disclosure | A 200 can still name `source` or `artifact_path` | Residual below. No gate fails that response |
| Denial of service | Artifact writes are the backtest route, which is admin-only and body-capped | `test_authz_matrix` |
| Elevation of privilege | A path that escapes the root is 400 or 422 | `test_pentest_surface.py` |

### Reverse proxy

| Threat | Control | Covered by |
|---|---|---|
| Spoofing | Forwarded headers are not trusted | `test_forwarded_for_does_not_become_loopback` |
| Elevation of privilege | A proxy that connects from loopback inherits the no-key fallback | No test can see a proxy the process does not observe. Gap. Set a key before adding a proxy |

### fx-1 harness

`src/fx1/serve/api.py` is a separate process and is hash-pinned. This tree does
not change it. Spoofing of `::ffff:127.0.0.1`, per-request key reload, and
the constant-time compare described under Pending harness change are gaps.
No gate in this slice covers them. `make -f deploy/security.mk security-gate`
does not scan that process.

### Supply chain and the running images

| Threat | Control | Covered by |
|---|---|---|
| Tampering of dependencies, images, Dockerfiles, or git history | `make -f deploy/security.mk security-gate` | `scripts/check_security_gate_summary.py` requires `ok` true, `dirty` false, an empty `new_findings` list, and `head` equal to `git rev-parse HEAD` |
| Information disclosure on the published ports | ZAP baseline plus the lab `/ready` 503 rule | `deploy/security/zap-rules.tsv`, `tests/unit/security/test_ready_not_ready.py` |
| Denial of service via the SQL rules' request rate | Production keeps 10 requests/second. A scanner sets the rate to `off` only in an uncommitted override | `docs/SECURITY_GATE.md` |

## Gaps

- Safe methods other than ledger and position reads are not audit lines. `GET /backtest/{id}` and `HEAD /backtest/{id}` are kind `ledger` with outcome `read`. `GET /risk/portfolio`, `GET /portfolio/target`, and `GET /portfolio/exposures` are kind `positions` with outcome `read`. `QUANT_API_AUDIT_READS` and `RESEARCH_API_AUDIT_READS` default on. `0`, `off`, `false`, or `no` disables those lines. The config dump is `dump_resolved`, which appends outcome `written`. There is no HTTP config-dump route. A full disk after the handler has started can return 503 `audit log unavailable` while a backtest artifact from that request already exists. A denied line that cannot be appended becomes that same 503, and the route does not run. A sensitive read does too, before the handler, when the log path cannot be opened.
- The default rate backend is in-process memory, so workers do not share buckets until `DIPCATCHER_RATE_BACKEND` is `sqlite` or `file` and `DIPCATCHER_RATE_STORE` points at the shared volume. A missing or unreadable store is 503 `rate store unavailable`. Loopback and remote peers use separate buckets on every backend.
- The hash chain detects a torn, deleted, or reordered line. `scripts/anchor_audit_head.py` appends the head digest and the time to a separate anchors file. That line is unsigned. Signing it is an owner step, the same as `gate_pins.sig`. A rewrite of the log and the anchors file still verifies. `scripts/verify_audit_chain.py` checks the chain. `scripts/prod_backup.py` copies the chain and the anchors file. `gate_pins.sig` is not re-signed here.
- The harness gaps in the pending section remain.
- `.env.example` does not list the file, grace, audit-log, loopback-rate, audit max-byte, rate-backend, audit-reads, or anchors variables.

## Entrypoints

| Entrypoint | Bind guard | Request gate |
|---|---|---|
| `dipcatcher api` (`quant_fund.cli.ops_cmds.api`) | Refuses a host other than `127.0.0.1`, `::1`, or `localhost` when `QUANT_API_KEY` and `QUANT_API_KEY_FILE` are both missing or empty. Default `127.0.0.1:8000`. | `api_auth_middleware` |
| `uvicorn quant_fund.api.app:app` | No bind guard. The request gate still refuses a non-loopback peer when no key is set. | `api_auth_middleware` |
| `python -m quant_fund.api.research_api` | Refuses a non-loopback `RESEARCH_API_HOST` when `RESEARCH_API_KEY` and `RESEARCH_API_KEY_FILE` are both missing or empty. Default `127.0.0.1:8010`. | `research_api_auth` |
| `uvicorn quant_fund.api.research_api:app` | No bind guard. Same request gate. | `research_api_auth` |
| `fx1 harness serve`, `python -m fx1.serve.api` | Refuses a non-loopback bind when `FX1_API_KEY` is missing or empty. Loopback still boots with no key. See the pending proposal. | `harness_api_auth` in `src/fx1/serve/api.py` |

## Authentication and authorization

Each app has an admin key and an optional read key.

| App | Admin key | Read key |
|---|---|---|
| Lab | `QUANT_API_KEY` | `QUANT_API_READ_KEY` |
| Research | `RESEARCH_API_KEY`, or `ResearchApiSettings.api_key` when `key_from_env` is false | `RESEARCH_API_READ_KEY` |

The admin key authorizes every method. The read key authorizes `GET`, `HEAD`,
and `OPTIONS`. Any other method presented with the read key is 403
`{"detail": "read key cannot modify"}`. That body does not echo the key.
The research API registers only GET routes, so the two keys authorize the
same registered routes. A POST with the read key is still that 403. A POST
with the admin key reaches the router and is 405 when the path has no POST
route.

`/health` is public on both apps and is not rate limited. `/ready` is public
on the research API only. The lab `/ready` probe requires a key when one is
configured. Neither app mounts `/docs` or `/redoc` (`docs_url` and
`redoc_url` are `None`). `/openapi.json` is mounted and follows the key rule.

| Keys on this request | Peer | Other routes |
|---|---|---|
| Either key non-empty | any | `X-API-Key` must match the admin key, or match the read key on a safe method. Otherwise 401 `{"detail": "invalid or missing X-API-Key"}`. A matching read key on another method is the 403 above. |
| Both missing or empty | loopback set above | Allowed |
| Both missing or empty | anything else | 403. The detail names the admin env var and tells the operator to set it or bind loopback. It does not echo a key. |

`_authenticate_request` loads both lab slots on every non-public request.
An empty string is unset. `load_key_slot` reads `{NAME}_FILE` when that
variable is set and otherwise reads the environment value. The research
admin slot does the same when `key_from_env` is true. An explicit
`ResearchApiSettings.api_key` is a frozen override used by tests; it does
not read `RESEARCH_API_KEY` or `RESEARCH_API_KEY_FILE`. The research read
slot still uses `load_key_slot("RESEARCH_API_READ_KEY")`.

`credential_refusal` runs `keys_match` four times, in order: current admin,
previous admin, current read, previous read. A missing header or an unset
slot is digested as the stand-in `absent-api-key`. The previous compare runs
even when the current key matched. Both sides of each call are SHA-256
digested and the digests are compared with `hmac.compare_digest`, so each
compare is fixed length. A surrogate that cannot be encoded as UTF-8 is a
mismatch, not a 500. Hash time still depends on the length of the header
value.

`{NAME}_FILE` is read on each request. While it is set, the environment
value of `{NAME}` is not accepted. `{NAME}_PREVIOUS_FILE` is accepted only
while the clock is strictly before the timestamp in
`{NAME}_GRACE_UNTIL_FILE`. The runbook is in `docs/SECURITY_GATE.md`. An
admin-key unsafe method appends a line to `QUANT_API_AUDIT_LOG` or
`RESEARCH_API_AUDIT_LOG` when that variable is set, and always emits the
same JSON through the `quant_fund.api.audit` logger. A rejected key on an
unsafe method appends outcome `denied`. If the configured file cannot be
opened, the mutation or the denial is 503 `audit log unavailable` and the
route does not run. `GET` and `HEAD` of `/backtest/{id}`, and `GET` of
`/risk/portfolio`, `/portfolio/target`, and `/portfolio/exposures`, append
outcome `read` when `QUANT_API_AUDIT_READS` is unset or any value other
than `0`, `off`, `false`, or `no`. The research flag is
`RESEARCH_API_AUDIT_READS`. The research API has no ledger, position, or
config-dump route, so it writes no read line. `dump_resolved` appends
outcome `written`. `PaperLedger.flush`, `LiveDesk.kill`, and the gateway
session `kill` append when that file is set. With no HTTP
actor the route is `cli:<user>@<host>`. Each line's `prev` is the SHA-256
of the previous raw line. The active file rotates to `NAME.1` when the next
line would pass `{LOG}_MAX_BYTES` (default 1048576).
`scripts/anchor_audit_head.py` appends that head digest and the time to a
separate anchors file. The line is unsigned. `scripts/prod_backup.py` copies
the chain and the anchors file.

The CLI bind guard still requires the admin key. `QUANT_API_KEY_FILE`
satisfies it. `QUANT_API_READ_KEY` and `QUANT_API_READ_KEY_FILE` do not.
The research module's bind guard is the same for `RESEARCH_API_KEY` and
`RESEARCH_API_KEY_FILE`.

## Controls on both APIs

Implemented in `quant_fund.api.http_controls` and called from
`quant_fund.api.app` and `quant_fund.api.research_api`.

### Body size

The cap is 64 KiB (`MAX_BODY_BYTES`, and `_MAX_REQUEST_BYTES` on the lab).

- A `Content-Length` that does not parse, or is negative, is 400
  `{"detail": "invalid Content-Length"}`.
- A declared or streamed body over 64 KiB is 413
  `{"detail": "request body exceeds the 64 KiB limit"}`.
- The lab counts bytes in `api_auth_middleware` before the route runs.
- The research app uses `reject_declared_body` and `attach_body_limit` inside
  `_research_dispatch`. The middleware turns `RequestBodyTooLarge` into the
  413 response.

The check runs for `/health` as well, so a public liveness route cannot be
used to stream an unbounded body.

### Rate limit

Token bucket. `DIPCATCHER_RATE_BACKEND` selects the store. Unset, empty, or `memory` keeps the bucket in the process and refills from `time.monotonic`. `sqlite` stores the bucket in a SQLite file at `DIPCATCHER_RATE_STORE`. `file` stores JSON at that path and takes an exclusive `fcntl` lock for the update. The shared backends refill from `time.time`, so two workers on one volume share a clock. A missing path, an unknown backend, or a store that cannot be opened is 503 `{"detail": "rate store unavailable"}`.

- `QUANT_API_RATE_PER_SEC` and `RESEARCH_API_RATE_PER_SEC` default to `10`.
- `QUANT_API_RATE_BURST` and `RESEARCH_API_RATE_BURST` default to `30`.
- `QUANT_API_LOOPBACK_RATE_PER_SEC` and `RESEARCH_API_LOOPBACK_RATE_PER_SEC` default to `100`.
- `QUANT_API_LOOPBACK_RATE_BURST` and `RESEARCH_API_LOOPBACK_RATE_BURST` default to `1000`.
- `0`, `off`, `false`, `no`, or an empty value disables the bucket named by that variable. Turning off the remote pair leaves the loopback pair in place.
- A value that does not parse keeps that pair's default.
- Loopback peers, including `testclient`, take the loopback pair. Their key bucket is separate from the remote key bucket. The bucket id is still a SHA-256 digest.
- `/health` is not counted.
- Other routes take an IP token first, including a request that will 401 or
  403. After the key matches, a second bucket is taken. Its id is
  `sha256(key)`, not the key. A read key refused for a mutating method does
  not take that second bucket.
- At 1024 identities the oldest bucket is dropped. The shared store drops the oldest row the same way.
- Over the limit: 429 `{"detail": "rate limit exceeded"}` and `Retry-After: 1`.

`reset_rate_gates` clears the in-process buckets. It does not delete a shared store. `tests/unit/api/test_rate_store.py` runs two processes against one sqlite or file store. The memory backend keeps a separate bucket in each process.

### CORS

Neither app installs `CORSMiddleware`. `enforce_closed_cors` removes
`Access-Control-Allow-Origin`, `Access-Control-Allow-Credentials`,
`Access-Control-Allow-Methods`, `Access-Control-Allow-Headers`, and
`Access-Control-Expose-Headers` if something else set them. An `Origin`
request header is not reflected. There is no allowlist.

### Security headers

`apply_security_headers` sets:

- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `Referrer-Policy: no-referrer`
- `Content-Security-Policy` with `frame-ancestors 'none'`
- `Permissions-Policy` denying camera, microphone, and geolocation
- `Cache-Control: no-store`
- `Strict-Transport-Security` when the request scheme is `https`

The lab middleware sets HSTS again on https so the pinned complexity of
`api_auth_middleware` stays at 11 (`quality/mccabe_baseline.txt`).

### Error responses

`install_plain_errors` registers the `Exception` handler. Starlette runs that
handler on the outermost server-error middleware, so it only sees errors that
route handlers did not turn into `HTTPException` or a validation error.
The body is `{"detail": "internal error"}`. The handler does not interpolate
the exception. It also applies the security headers and strips CORS headers,
because this response is produced outside user middleware.

`HTTPException` and request-validation errors keep their status and their
existing detail. A 422 from a point-in-time failure can include `str(exc)`.
That path is not the 500 handler.

Starlette re-raises after the 500 handler so the server process can log the
traceback. The HTTP body does not contain it. `debug=True` would replace the
handler with an HTML traceback. Both apps are constructed with debug off.

## Route catalog

### Lab API (`quant_fund.api.app`)

Public: `GET /health`.

Key required when configured, loopback-only when not. Safe methods accept
either key. `POST /portfolio/optimize` and `POST /backtest` accept the admin
key. `/docs` and `/redoc` are not mounted.

- `GET /ready`
- `GET /models`
- `GET /forecast/{symbol}`
- `GET /forecast/{symbol}/distribution`
- `GET /ranking`
- `GET /regime`
- `GET /risk/portfolio`
- `GET /portfolio/target`
- `GET /portfolio/exposures`
- `POST /portfolio/optimize`
- `POST /backtest`
- `GET /backtest/{backtest_id}`
- `GET /monitoring/drift`
- `GET /doctor`
- `GET /research/latest`
- `GET /openapi.json`

### Research API (`quant_fund.api.research_api`)

Every declared route is GET. Public: `GET /health` and `GET /ready`.
`/ready` is 200 when the evidence roots are directories and 503 otherwise.
`/docs` and `/redoc` are not mounted.

The same key rule covers:

- `GET /runs`, `GET /runs/latest`, `GET /runs/{run_id}`,
  `GET /runs/{run_id}/markdown`, `GET /runs/{run_id}/verification`
- `GET /receipts`, `GET /receipts/by-hash/{digest}`, `GET /receipts/{receipt_id}`,
  `GET /receipts/{receipt_id}/verification`
- `GET /results`, `GET /results/{kind}/{name}`,
  `GET /results/{kind}/{name}/verification`
- `GET /verifier/versions`, `GET /verifier/versions/{version}`
- `GET /verifier/runs`, `GET /verifier/runs/{run_log_id}`
- `GET /artifacts`, `GET /artifacts/{artifact_id:path}`
- `GET /openapi.json`

Ids are checked against closed regular expressions, and every file read goes
through `_contained`.

## Pending harness change

`src/fx1/serve/api.py` is hash-pinned. This branch does not modify it, and it
does not modify `src/fx1/cli.py`. Current behavior, read from those files:

- `create_app` copies `FX1_API_KEY` once (`api_key = os.environ.get("FX1_API_KEY") or None`) and `_resolve_auth` compares that closed-over value. Rotating the variable does nothing until the process builds a new app.
- The compare is `hmac.compare_digest` on the raw UTF-8 bytes. Python's compare returns early when the lengths differ, so the length of the configured key can leak through timing. A surrogate uses `surrogatepass` so a malformed key does not become a 500.
- With no key and no managed keys, only `_LOOPBACK_HOSTS` (`127.0.0.1`, `::1`, `localhost`, `testclient`) is served. `::ffff:127.0.0.1` is not unwrapped, so it is refused. A missing `request.client` is refused.
- `/health` is public. With a key set, other routes require `X-API-Key` (Bearer on OpenAI-shaped paths). Managed keys add a scope check.
- `fx1 harness serve` and `python -m fx1.serve.api` refuse a non-loopback bind when `FX1_API_KEY` is missing or empty. A loopback bind still starts with no key. `import fx1.serve.api` does not exit.
- Body cap is 1 MiB. `FX1_API_RATE_LIMIT_RPS` defaults to `0`, which leaves the limiter off. When set, the bucket is per client host, not per key. `FX1_API_CORS_ORIGINS` defaults to empty, which installs no CORS middleware. A non-empty list is an explicit origin allowlist with credentials off.

The pending proposal, written with the workflow-hardening slice and not
applied, changes only the serving entrypoints and the auth lookup:

- `uvicorn fx1.serve.api:app`, `python -m fx1.serve.api`, and `fx1 harness serve` refuse to boot unless `FX1_API_KEY` is non-empty, including a loopback bind.
- `create_app()` stays callable with no key so in-process tests keep working.
- Middleware reads `FX1_API_KEY` on each request instead of closing over the import-time value.
- The client host strips brackets and a zone id, then unwraps IPv4-mapped IPv6. `::ffff:127.0.0.1` becomes loopback. `::ffff:169.254.169.254` stays link-local and is refused when no key is configured.

Applying that proposal re-signs `src/fx1/serve/api.py`. The lab and research
apps in this branch do not unwrap IPv4-mapped addresses. On those two apps,
`::ffff:127.0.0.1` stays a remote peer: it needs a key, and it is rate limited.

## Residual risks

- A reverse proxy in front of loopback inherits the no-key fallback and skips
  the rate limit. Set the key first.
- The memory backend is per process. Several workers on that backend multiply the budget, and a restart clears it. `sqlite` and `file` share `DIPCATCHER_RATE_STORE`. The limit is not a quota. A shared volume that does not honor SQLite locks or `fcntl` fails closed with 503.
- `/health` is unauthenticated and unlimited. It returns status, firm,
  product, version, and mode. It does not return a key.
- SHA-256 of the presented key takes time proportional to the header length.
  The digest compare does not.
- Successful lab responses can include a local path: `source` on
  `GET /risk/portfolio` and `artifact_path` on a stored backtest. Those are
  200 bodies, not the 500 handler.
- Some 422 details are `str(exc)` (`PointInTimeError` on the portfolio
  routes). They are not passed through `install_plain_errors`.
- Process logs can still contain a traceback, because the server-error
  middleware re-raises after writing the generic body.
- `debug=True` on a future constructor would HTML-render the traceback.
- The admin key still authorizes every method on that app. The read key does
  not. A deployment that sets only the admin key has no separate read credential.
- Direct `uvicorn` of either module skips the CLI bind guard. The request
  gate still applies.
- `.env.example` is hash-pinned, so the rate-limit variables and
  `QUANT_API_READ_KEY` / `RESEARCH_API_READ_KEY` are documented here and are
  not added to that file.
- `docs/THREAT_MODEL.md` is a new corpus file. A later evidence audit with
  `require_stamped` will report it until the docs epoch is restamped. This
  change does not edit `quality/*.json` or `gate_pins.sig`.
- The harness gaps above remain until the pending proposal is applied and
  `src/fx1/serve/api.py` is re-signed. This branch does not change that file,
  its signature, crown-jewel verifiers, or workflow `runs-on` labels.

## Outbound clients

`quant_fund.utils.outbound.vetted_addresses` is the shared selector.
`literal_connect_address` is the same classification without DNS, for a
check that must not resolve yet. A decimal, octal, hex, short, trailing-dot,
or compatibility-folded IPv4 literal is parsed before `getaddrinfo`. 6to4,
Teredo, NAT64 `64:ff9b::/96`, IPv4-translated, and deprecated IPv4-compatible
embeddings are refused whole. The caller dials one returned numeric address.

| Caller | How it dials |
|---|---|
| `open_pinned`, `read_http`, `read_http_redirects`, `public_addresses` | `vetted_addresses` with the boolean loopback/private flags |
| BYOK (`backends._byok_resolved_addresses`) | `vetted_addresses` and `FX1_PRIVATE_ALLOW_CIDRS` before DNS. Connect uses `addresses[0]` |
| Unflagged OpenAI and local-engine completion (`_openai_urlopen`) | Same pin. Loopback and RFC1918/ULA literals, and the name `localhost`, stay dialable. Any other name must be public. `http_proxy` and `https_proxy` are not consulted |
| Webhooks | `literal_connect_address` at `check_callback_url`. `vetted_addresses` again at delivery. `_post_once` dials that address |
| `quant_fund.data.concurrent_io` | `public_addresses` on every pooled hop. A checkout without an address resolves the same way, then `pinned_connection` |
| `quant_fund.live.brokers.urllib_transport` | `open_pinned`. Redirects are not followed. Alpaca live still refuses before the socket |
| `quant_fund.observe.tracing._export` | `vetted_addresses` with loopback literals allowed, then `pinned_connection`. A refused collector is an `otel_export` error and does not raise |
| `src/fx1/serve/client.py` | `open_pinned` (`allow_loopback`, `allow_private`) |
| `src/fx1/webresearch` | `open_pinned` / `read_http` |
| Research witnesses and `scripts/vendor_http.py` | `read_http` / `read_http_redirects` |
| `src/fx1/serve/e2e_audit.py`, `scripts/fx1_goldenpath.py` | `open_pinned` with loopback allowed. The URLs are `127.0.0.1` plus a port this process bound |

### Exemptions

- `scripts/verify_auditor_bundle.py` calls `urllib.request.urlopen` on the fixed URL `https://rekor.sigstore.dev/api/v1/log/publicKey`. The script is a second verifier and imports nothing from `quant_fund`. The host is not caller input. A DNS rebind of that name is the residual.
- `third_party/kronos/examples/get_akshare_date_2024-2025_x.py` and `get_date_new.py` call `requests.get` on fixed AkShare hosts. They are vendored examples. Product code does not import them.
- `src/fx1/serve/jobs_audit.py`, `src/fx1/serve/idem_audit.py`, and `tests/unit/fx1_serve/test_harness_idempotency_races.py` construct `httpx.AsyncClient` with `ASGITransport`. The client stays in process. There is no DNS and no socket.
- `tests/fx1/test_goldenpath.py` and `tests/examples/test_fx1_quickstart_startup.py` call `urlopen` on `http://127.0.0.1` servers those tests bind.
- `src/quant_fund/data/adapters/hf_ohlcv_1m.py` `stream_https` takes an injected opener and is covered by tests. The production download is `http_download` → `pooled_stream` → `public_addresses`. `stream_https` is pinned at McCabe 12, so this change does not rewrite it.
- `src/fx1/serve/api.py` imports `urllib.request` and does not open a connection. The file is hash-pinned and is not modified here.
- `quant_fund.paper.trading_session` forbids importing `urllib`, `requests`, `httpx`, and `aiohttp`. It has no outbound client.
- No `aiohttp` call site exists in this repo.
