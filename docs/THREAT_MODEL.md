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

- Operator API keys in the process environment: `QUANT_API_KEY`,
  `RESEARCH_API_KEY`, and, on the harness, `FX1_API_KEY`. The request path
  sees them only as an `X-API-Key` header. Bucket ids store a SHA-256 digest,
  not the key.
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
| Environment | The key read on that request | A key copied into the app at import, for the env-backed apps |
| Filesystem | Paths that stay inside the configured root after resolve | User path segments that escape the root (`research_api._contained`, `resolve_allowed_config_path`) |
| Reverse proxy | Nothing, until a key is set | A proxy on the same host makes every browser look like loopback |

`X-Forwarded-For` is not read. A reverse proxy bound to loopback collapses
every client into the loopback peer: the no-key fallback allows them, and the
rate limit does not apply. Set the API key before placing either service
behind a proxy.

## Entrypoints

| Entrypoint | Bind guard | Request gate |
|---|---|---|
| `dipcatcher api` (`quant_fund.cli.ops_cmds.api`) | Refuses a host other than `127.0.0.1`, `::1`, or `localhost` when `QUANT_API_KEY` is missing or empty. Default `127.0.0.1:8000`. | `api_auth_middleware` |
| `uvicorn quant_fund.api.app:app` | No bind guard. The request gate still refuses a non-loopback peer when no key is set. | `api_auth_middleware` |
| `python -m quant_fund.api.research_api` | Refuses a non-loopback `RESEARCH_API_HOST` when `RESEARCH_API_KEY` is missing or empty. Default `127.0.0.1:8010`. | `research_api_auth` |
| `uvicorn quant_fund.api.research_api:app` | No bind guard. Same request gate. | `research_api_auth` |
| `fx1 harness serve`, `python -m fx1.serve.api` | Refuses a non-loopback bind when `FX1_API_KEY` is missing or empty. Loopback still boots with no key. See the pending proposal. | `harness_api_auth` in `src/fx1/serve/api.py` |

## Authentication and authorization

Both lab and research apps use one shared key. There are no scopes and no
per-route roles. A matching key authorizes every non-public route, including
`/docs`, `/redoc`, and `/openapi.json`.

`/health` is the only public route on each app (`_PUBLIC_PATHS`). It is not
rate limited.

| Key on this request | Peer | Other routes |
|---|---|---|
| Non-empty | any | `X-API-Key` must match, or the response is 401 `{"detail": "invalid or missing X-API-Key"}` |
| Missing or empty | loopback set above | Allowed |
| Missing or empty | anything else | 403. The detail names the env var and tells the operator to set it or bind loopback. It does not echo a key. |

The lab reads `QUANT_API_KEY` inside `api_auth_middleware` on every request.
An empty string is treated as unset. The research app built with
`ResearchApiSettings.from_env` sets `key_from_env` and `research_api_key`
calls `nonempty_env("RESEARCH_API_KEY")` on every request. An explicit
`ResearchApiSettings.api_key` is a frozen override used by tests; it does not
read the environment.

Comparison is `quant_fund.api.http_controls.keys_match`. Both sides are
SHA-256 digested and the digests are compared with `hmac.compare_digest`, so
the compare itself is fixed length. A surrogate that cannot be encoded as
UTF-8 is a mismatch (401), not a 500. Hash time still depends on the length
of the header value.

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

In-memory token bucket, one process, not shared across workers.

- `QUANT_API_RATE_PER_SEC` and `RESEARCH_API_RATE_PER_SEC` default to `10`.
- `QUANT_API_RATE_BURST` and `RESEARCH_API_RATE_BURST` default to `30`.
- `0`, `off`, `false`, `no`, or an empty value disables that API's limit.
- A value that does not parse keeps 10 per second and a burst of 30.
- Loopback peers, including `testclient`, skip the limit.
- `/health` is not counted.
- Other routes take an IP token first, including a request that will 401 or
  403. After the key matches, a second bucket is taken. Its id is
  `sha256(key)`, not the key.
- At 1024 identities the oldest bucket is dropped.
- Over the limit: 429 `{"detail": "rate limit exceeded"}` and `Retry-After: 1`.

`reset_rate_gates` clears process state for tests.

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

Key required when configured, loopback-only when not:

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
- `GET /docs`, `GET /redoc`, `GET /openapi.json`, `GET /docs/oauth2-redirect`

### Research API (`quant_fund.api.research_api`)

Every declared route is GET. Public: `GET /health`.

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
- `GET /docs`, `GET /redoc`, `GET /openapi.json`, `GET /docs/oauth2-redirect`

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
- Buckets are per process. Several workers multiply the budget. A restart
  clears them. The limit is not a quota.
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
- One key authorizes the whole app. There is no read-only vs. optimize split
  on the lab API.
- Direct `uvicorn` of either module skips the CLI bind guard. The request
  gate still applies.
- `.env.example` is hash-pinned, so the new rate-limit variables are
  documented here and not added to that file.
- `docs/THREAT_MODEL.md` is a new corpus file. A later evidence audit with
  `require_stamped` will report it until the docs epoch is restamped. This
  change does not edit `quality/*.json` or `gate_pins.sig`.
- The harness gaps above remain until the pending proposal is applied and
  `src/fx1/serve/api.py` is re-signed. This branch does not change that file,
  its signature, crown-jewel verifiers, or workflow `runs-on` labels.
