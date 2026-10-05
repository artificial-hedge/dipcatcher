# fx-1 harness — deployment runbook

The operator's guide to running `fx1.serve.api` in front of real callers:
process launch, durable state, managed-key bootstrap and rotation,
webhook verification, graceful shutdown, the deploy gates, usage
accounting, and the failure-mode table. The wire contract itself lives in
`docs/FX1_HARNESS_API.md`; the client-side legs in `docs/FX1_CLIENTS.md`;
the BYOK provider details in `docs/FX1_BYOK_RUNBOOK.md`. A zero-config
walkthrough of every step below runs in `examples/fx1_quickstart_http.py`.

## Process launch

The canonical launcher is the CLI — it is the only path that enforces the
non-loopback auth precondition:

```bash
uv run fx1 harness serve --host 0.0.0.0 --port 8011
```

`fx1 harness serve` builds `create_app()` and hands it to
`uvicorn.run(..., reload=False, server_header=False)`. Every knob has a
flag and an env twin — flags win:

| Flag | Env | Default | Meaning |
|---|---|---|---|
| `--host` | — | `127.0.0.1` | Bind host. Non-loopback **refuses to start** without `FX1_API_KEY`. |
| `--port` | — | `8011` | Bind port. |
| `--max-inflight` | `FX1_API_MAX_INFLIGHT` | `16` | Concurrent heavy requests; saturation answers `503 over_capacity`. |
| `--job-max` | `FX1_API_JOB_MAX` | `1024` | Async job-store capacity. |
| `--idem-max` | `FX1_API_IDEM_MAX` | `1024` | Idempotency-key store capacity. |
| `--sse-keepalive-s` | `FX1_API_SSE_KEEPALIVE_S` | `15` | SSE keepalive comment interval. |
| `--rate-limit-rps` | `FX1_API_RATE_LIMIT_RPS` | `0` (off) | Per-client requests/s cap. |
| `--gzip-min-bytes` | `FX1_API_GZIP_MIN_BYTES` | `1024` | Response-compression floor. |
| `--cors-origins` | `FX1_API_CORS_ORIGINS` | off | Comma-separated explicit `scheme://host` origins; `*` is rejected. |
| `--breaker-threshold` | `FX1_API_BREAKER_THRESHOLD` | `5` | Consecutive backend faults that open the circuit (`0` = off). |
| `--breaker-cooldown-s` | `FX1_API_BREAKER_COOLDOWN_S` | `30` | Open-circuit fast-fail window before a half-open probe. |
| `--receipts-dir` | `FX1_API_RECEIPTS_DIR` | `receipts` | Sealed-receipt store backing `GET /receipts*`. |
| `--store-max` | `FX1_API_STORE_MAX` | `256` | `/v1/vector_stores` index capacity. |
| `--state-dir` | `FX1_API_STATE_DIR` | off | Durable journals — see below. |

Plus `FX1_API_FILE_MAX` / `FX1_API_FILE_BYTES` / `FX1_API_BATCH_MAX` /
`FX1_API_BATCH_LINES` for the file and batch stores (only env-configured),
and `FX1_API_BYOK_OVERRIDE` (`1` default) to allow/disallow per-request
BYOK overrides — see `docs/FX1_BYOK_RUNBOOK.md`.

Embedding the app instead of the CLI: `create_app()` is a plain FastAPI
app — `uvicorn fx1.serve.api:create_app` does **not** work (the factory
takes kwargs), but `create_app(state_dir=..., rate_limit_rps=...)` under
any ASGI server does. You lose the non-loopback refusal guard, so enforce
`FX1_API_KEY` yourself — the app accepts it, it just won't refuse to
start without it.

Liveness/readiness probes:

- `GET /health` — the only public route; no auth, cheap.
- `GET /ready` — read scope; `503` once the drain latch is set. Poll this
  before cutting traffic, and as the drain witness during shutdown.
- `GET /harness/version` — `{"api_version", "fx1_version"}` for client
  negotiation (`HarnessClient.check_compat`, `fx1 harness version`).
- `GET /harness/capabilities` — feature flags + effective limits; clients
  self-configure from this instead of hardcoding.
- `GET /metrics` — ops counters; `Accept: text/plain` (or
  `fx1 harness metrics --format prom`) returns Prometheus text
  (`fx1_requests_total`, `fx1_draining`, `fx1_rate_limited_total`,
  per-backend outcomes, …).

## Durability: `--state-dir`

Without it every store is in-process: a restart drops queued/running
jobs, evals, batches, idempotency keys, and pending webhooks. With
`--state-dir /var/lib/fx1` every store journals into that directory:

| File | Contents |
|---|---|
| `jobs.jsonl` | Async run-job transitions (`/harness/jobs`, `/v1/...` job surfaces) |
| `evals.jsonl` | Eval records (`POST /harness/evals`) |
| `eval_specs.jsonl` | Eval spec containers (`/v1/evals` spec objects) |
| `batches.jsonl` / `abatches.jsonl` | OpenAI `/v1/batches` and Anthropic `/v1/messages/batches` |
| `ft_jobs.jsonl` | Fine-tuning job transitions |
| `files.jsonl` + `files/<file_id>.bin` | `/v1/files` metadata + blob bytes (blob lands via tmp+rename **before** the journal line) |
| `keys.jsonl` | Managed-key records — sha256 fingerprints only, raw keys never touch disk |
| `vector_stores.jsonl` | Vector-store metadata + `vs_touch` membership lines |
| `idem_{runs,complete,complete_batch,openai,anthropic,legacy}.jsonl` | Idempotency-key → first-response maps, per write surface |

Journal semantics (`fx1.serve.journal`):

- Every line is `{"seq", "chain", "sha256", "payload"}` where `chain` is
  the previous line's sha256 — an append-only hash chain, fsync'd before
  the API answers.
- Boot replay verifies each line's hash + link and **stops at the first
  bad one**: a crash mid-append truncates honestly instead of corrupting
  state; a mid-journal edit invalidates the tail rather than smuggling.
- Non-terminal work recovers as `failed` with
  `error="process restarted before completion"` — no fake re-execution.
  Its `Idempotency-Key` mapping survives, so a client retrying the same
  submission gets the lost record back (`replayed=true`), not a duplicate.
- `callback_secret` is never journaled — a recovered record keeps
  `callback_url` for audit but cannot deliver a signed webhook
  post-restart (`callback_attempts` stays 0).
- Stores compact their journals when they grow past bounds; a crash
  mid-compact leaves the old journal intact.

```bash
uv run fx1 harness serve --state-dir /var/lib/fx1
# SDK twin: Fx1Harness(state_dir=...) or FX1_SDK_STATE_DIR
```

## Managed keys

Two credential classes:

- **Bootstrap credential** — the `FX1_API_KEY` env var. Full access
  (`admin`), never expires, unmetered. It exists so you can mint managed
  keys; workload traffic should not run on it.
- **Managed keys** — minted via `POST /harness/keys`. Raw key is
  `fx1k_<hex>`, shown **once** in the mint response; the server stores
  only the sha256 fingerprint. The record's `id` (first 16 hex chars of
  the sha) is the handle for list/get/usage/revoke.

Bootstrap paths:

```bash
# 1. With FX1_API_KEY set (the normal path) — mint the first admin key:
fx1 harness key-create --remote https://fx1.internal:8011 \
  --api-key "$FX1_API_KEY" --admin --name ops-admin

# 2. Loopback-dev path — no FX1_API_KEY and an empty key store: any
#    loopback client may mint (this is how first-boot provisioning works
#    before the env credential exists). Once any key exists, the
#    loopback bypass is off.
```

Per-key controls at mint (`POST /harness/keys` / `key_create` /
`fx1 harness key-create`):

| Field | Effect |
|---|---|
| `admin` | May mint/list/revoke keys itself (`/harness/keys*`, `/harness/drain`). Adds `admin` to the scope set. |
| `scopes` | `read` (GET/HEAD/OPTIONS), `write` (data-plane mutations), `admin` (control plane). Unset = `read,write` (+`admin` when `admin=True`). Out-of-scope → `403 insufficient_scope`. |
| `rpm` | Requests per 60 s window; over-limit → `429 rate_limited` + `Retry-After` + `X-RateLimit-*` headers. |
| `ttl_s` | Expiry in seconds from mint; expired keys fail closed like revoked ones. |
| `max_requests` | Hard cap on authenticated calls; exhausted → `429 quota_exceeded` (no `Retry-After` — terminal). |
| `max_tokens` | Hard cap on provider-reported token spend; charged post-response off the completion ring. |

Usage and introspection:

- `GET /harness/self` (`fx1 harness self`) — **read** scope; the calling
  credential's own card: class (`managed`/`env`/`none`), scopes, and for
  managed keys the budget headroom.
- `GET /harness/keys/{id}/usage` (`fx1 harness key-usage`) — **admin**
  scope; the operator card: `uses`, `tokens_used`, `requests_remaining`,
  `tokens_remaining`, `window_remaining`, plus the completion-ring
  `served` split per backend.
- `GET /harness/usage` (`fx1 harness usage`) — fleet-wide token/request
  accounting over the completion log; `since`/`until`/`backend`/`model`/
  `key_id` filters.

### Rotation drill

Use the mint and revoke APIs for this rotation drill: mint a replacement,
switch callers, then revoke the old key. Re-declare the intended scopes
and budgets on the replacement. This procedure lets callers verify the
new credential before the old one stops working:

```bash
# 1. mint the replacement (same scopes/budgets — re-declare them)
NEW=$(fx1 harness key-create --remote $URL --api-key "$FX1_API_KEY" \
  --name deploy-v2 --rpm 60 --max-requests 100000 | jq -r .key)

# 2. cut callers over — they pick up $NEW at next read
#    (verify the new key works before revoking the old one)
curl -sf $URL/harness/self -H "X-API-Key: $NEW" | jq .credential   # "managed"

# 3. tombstone the old key — fails closed immediately, record kept for audit
fx1 harness key-revoke --remote $URL --api-key "$FX1_API_KEY" <old-key-id>
```

`DELETE /harness/keys/{id}` never deletes — it writes
`enabled=false` + `revoked_at`; subsequent auth with that key answers
401. The record stays for audit (`fx1 harness keys` lists tombstones).

## Webhooks

Any async submission (`/harness/jobs`, `/harness/evals`, batches) that
carries `callback_url` + `callback_secret` POSTs the terminal record to
the URL with:

- `X-Fx1-Webhook-Timestamp` — unix seconds at delivery time
- `X-Fx1-Webhook-Signature` — `sha256=<hmac-sha256 hex>` over
  `"{timestamp}.{raw body}"`

Verify over the **raw** body — re-serialization changes the bytes.

Python (same code the harness ships):

```python
from fx1.serve.webhooks import verify_webhook  # 300 s freshness window

ok = verify_webhook(
    secret=os.environ["FX1_CALLBACK_SECRET"],
    timestamp=request.headers["X-Fx1-Webhook-Timestamp"],
    signature=request.headers["X-Fx1-Webhook-Signature"],
    body=await request.body(),  # raw bytes — not request.json()
)
```

TypeScript (the client exposes the twin):

```ts
const ok = await HarnessApiClient.verifyWebhook({
  body: rawBody,
  signature: req.headers["x-fx1-webhook-signature"],
  timestamp: req.headers["x-fx1-webhook-timestamp"],
  secret: process.env.FX1_CALLBACK_SECRET!,
});
```

Delivery: `deliver_signed` retries transient faults (network, 5xx) up to
3 attempts with capped exponential backoff (`0.5 s` base); a 4xx is a
definitive rejection and is never retried; per-attempt timeout is 10 s.
Undeliverable callbacks land on the record (`callback_attempts`,
`callback_error`) — poll `GET /harness/jobs/{id}` / `evals/{id}` to audit.
Do not rely on webhooks alone for durability: `callback_secret` is never
journaled, so a recovered job reports `callback_attempts: 0`.

## Drain and shutdown

`POST /harness/drain` (`fx1 harness drain --remote $URL`) latches drain
mode — one-way and idempotent:

- Gated routes immediately refuse new work with `503 draining`.
- In-flight requests finish; `/ready` starts answering 503.
- `GET /metrics` keeps reporting `inflight`; `fx1_draining` is the gauge.
- `--wait-s N` (or `drain(wait_s=N)`) blocks server-side until the pool
  empties and reports `drained: true`.

Shutdown ordering:

```bash
fx1 harness drain --remote $URL --wait-s 60   # returns drained=true
# then SIGTERM the process — uvicorn exits on the signal
```

Drain first, then signal: hard-killing a live process leaves non-terminal
jobs to be recovered as `failed` at next boot — correct, but noisy.

## Deploy gates

```bash
# Zero-config golden path — boots a stub engine + the production app on
# loopback and walks health/auth/commands/BYOK completion/SSE/idem/
# jobs/receipt/drain/SDK parity (17 checks, exit 0/2):
fx1 harness selftest
fx1 harness selftest --state-dir /tmp/fx1-state   # adds restart recovery
fx1 harness selftest --remote https://fx1.internal:8011 --api-key $K  # read-only probes

# Latency/throughput gate — n measured requests at c workers; exits 0
# only when every measured request succeeded (exit 1 on any failure):
fx1 harness bench --remote $URL --api-key $K --n 64 --concurrency 8 \
  --byok-base-url $BYOK_URL --byok-model $MODEL   # or --backend local_fx1
fx1 harness bench --remote $URL --receipt        # prints the sealed
                                                 # fx1_bench_result.v1 doc
```

`selftest` remote mode never mutates the deployment — health, version,
capabilities, commands listing only. `bench` sends real traffic: run it
against staging, or size `--n` to what the deployment can absorb.

## Failure modes

| Status | `code` | Meaning | Retry contract |
|---|---|---|---|
| 401/403 | `invalid_api_key`, `insufficient_scope`, `admin_required` | Credential unknown, out of scope, or non-admin on `/harness/keys*`/`/harness/drain` | Never — fix the credential |
| 404 | — | Unknown job/eval/completion/key id | Never |
| 422 | — | Malformed body, bad `backend`, `callback_url` not http(s), unknown eval suite | Never — fix the request |
| 429 | `rate_limited` | Managed key's `rpm` window full | Yes — `Retry-After` + `X-RateLimit-{Limit,Remaining,Reset}-Requests` say when |
| 429 | `quota_exceeded` | `max_requests`/`max_tokens` budget exhausted | **No — terminal.** No `Retry-After` is sent; SDKs must not retry |
| 429 | `too_many_requests` | Server-wide `FX1_API_RATE_LIMIT_RPS` cap | Yes — `Retry-After` is set |
| 501 | `not_supported`/`not_implemented` | Backend lacks the surface (e.g. stream/tokenize on a provider with no `/tokenize`) | Never — the capability is absent |
| 502 | `honesty_gate` | The gate refused model output before the bytes left | Never — a refusal is a verdict, not a fault |
| 502 | `backend_failure` | Backend `RuntimeError` mid-call (upstream fault surfaced verbatim) | Diagnose the provider first |
| 503 | `backend_unavailable` | `BackendNotConfiguredError` — the lane's env isn't set (`FX1_BYOK_*`, checkpoint, `MOONSHOT_API_KEY`) | After configuring the backend |
| 503 | `draining` | Drain latch set | After the next process comes up |
| 503 | `over_capacity` | `max_inflight` saturated or job executor down | Yes — `Retry-After: 1`; scale `--max-inflight` if chronic |
| 503 | `receipts_unavailable` | `--receipts-dir` missing/unreadable | After fixing the store mount |

The circuit breaker (`--breaker-threshold`/`--breaker-cooldown-s`) wraps
backend resolution: N consecutive faults open the circuit and the lane
fast-fails `503` with `Retry-After` until the cooldown lets a probe
through. `fallbacks=[...]` on any completion/eval call tries the next
link on availability faults (`backend_unavailable` only) — never on
honesty-gate refusals or 4xx.

## Accounting

- `GET /harness/usage` — fleet token/request accounting over the bounded
  completion ring. `log_dropped` on the report marks when the ring has
  overwritten old entries (totals are then a lower bound).
- `GET /harness/keys/{id}/usage` — per-key card with declared budgets and
  headroom (admin scope).
- `GET /harness/self` — the caller's own card (read scope).
- `fx1 harness usage --remote $URL`, `key-usage`, `self` — CLI twins.

Every served response stamps the credential's fingerprint on the
completion record, so `key_id`-filtered usage is exact even when callers
share a base URL. The meters (`uses`, `last_used_at`, `tokens_used`) and
the rpm window occupancy are journaled as counter snapshots on every
admitted request, and the journal compacts to one record per key on
boot. On restart the store restores the full spend — declared budgets
and their lifetime usage meters are restart-durable.
