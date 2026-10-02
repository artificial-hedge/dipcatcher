# Harness API — wire contract & consumption modes

The dipcatcher harness (registry + backends + honesty gate + receipt
verifier) is one code path exposed four ways. This page is the
integrator reference; `docs/FX1.md` has the model overview and
`docs/FX1_API_STABILITY.md` the versioning policy.

## Consumption modes

| Mode | Entry point | Use when |
|---|---|---|
| HTTP API | `fx1 harness serve` | fx-1 pods / services calling over the network |
| Typed SDK | `from fx1.sdk import Fx1Harness` | in-process Python — no socket |
| Remote client | `fx1.serve.client.HarnessClient` | Python callers on a remote harness — same result types as the SDK |
| TS client | `clients/typescript/fx1` (`HarnessApiClient`) | TypeScript/JS callers — generated from the pinned OpenAPI spec |
| OpenAI-compatible | `GET /v1/models`, `POST /v1/chat/completions` | drop-in for OpenAI SDKs / existing toolchains — set `base_url` to the harness |
| CLI | `fx1 harness …` | shell, CI, ops scripts |

The Python surfaces share one error taxonomy (`KeyError` 404 /
`ValueError` 422 / `BackendNotConfiguredError` 503 /
`NotImplementedError` 501 / `Fx1HonestyError` 502-gate /
`HarnessAuthError` 401–403 / `HarnessTransportError` for wire faults);
the TS client collapses it into `HarnessApiError` carrying `status` +
the envelope's `code`. Either way switching surfaces is a construction
change, never a semantic one. Byte-identical payloads and
error classes are pinned by `receipts/fx1_parity_audit.json`; the
real-socket lifecycle (uvicorn + urllib) by `receipts/fx1_e2e_audit.json`.

## Backends (the model side)

`backend=` selects the model lane on every completion/eval call:

- **`hosted_k3`** — the K3 endpoint (`MOONSHOT_API_KEY`), temperature 0.
- **`local_fx1`** — weights-direct. Attaches to a running engine
  (`FX1_LOCAL_SERVE_URL`) or spawns one (`FX1_LOCAL_SERVE_CMD` with
  `$checkpoint_dir`/`$python` template vars); a card'd checkpoint dir is
  required and signature-gated before any spawn.
- **`byok`** — bring-your-own-key to any OpenAI-compatible
  `/chat/completions` endpoint: `FX1_BYOK_BASE_URL` +
  `FX1_BYOK_API_KEY` + `FX1_BYOK_MODEL` (kwargs beat env). Missing
  credentials fail closed at construction; the URL must be http(s) with
  a netloc. Wire shape pinned: `{model, messages, temperature: 0.0}`,
  Bearer auth.

  **Per-request override:** `complete`, `complete/batch`, and
  `complete/stream` accept a `byok` object
  (`{base_url, api_key, model}` — all three required, `base_url` must be
  http(s)) so each caller can route to its own endpoint instead of the
  server's env config. Overrides on a non-`byok` backend are 422; a
  deployment can refuse them entirely with `create_app(byok_override=False)`
  or `FX1_API_BYOK_OVERRIDE=0` (then `byok` → 422
  `byok_override_disabled`). Each override gets its own breaker circuit
  (`byok:<sha256(base_url)[:16]>`) so one tenant's dead endpoint never
  fast-fails another's; the idempotency fingerprint is the request body's
  sha256, so stored dedupe records carry no key material. `GET
  /harness/capabilities` reports `features.byok_override`. On the SDK and
  `HarnessClient`, pass the same dict as `byok=` on
  `complete`/`complete_many`/`stream_complete`; the CLI takes
  `--byok-base-url --byok-api-key --byok-model` (all-or-none).

**Per-request backend deadline:** `timeout_s` (0, 3600] on
`complete`/`complete/batch`/`complete/stream` is handed to the backend
constructor as `timeout_s`, overriding each backend's env/config default
— including `hosted_k3`, which previously hardcoded 120s. Out-of-range
values are model-level 422s. SDK/`HarnessClient` take `timeout_s=`; the
CLI takes `--backend-timeout` (distinct from `--timeout`, the HTTP
transport deadline).

**Fallback chains:** `fallbacks` (≤2, distinct, never repeating the
primary) orders alternate backends after `backend` on every completion
route — `["local_fx1"]` with `fallbacks=["hosted_k3","byok"]` tries the
local engine, then hosted, then BYOK. The chain advances only on
availability faults: a link that fails to resolve (unconfigured,
spawn refused, endpoint down) or faults at the call
(`BackendNotConfiguredError`/`RuntimeError`, wire `backend_unavailable`
503 / `backend_failure` 502) or sits behind an open circuit is skipped;
a gate refusal (502 `honesty_gate`), a capability gap (501
`not_supported`), or any client error aborts the request — a refusal is
a verdict, not a reason to spend another backend's capacity. An
exhausted chain re-raises the last retriable verdict. Per-link kwargs:
`byok` binds only a `byok` link and `checkpoint_dir` only a `local_fx1`
link — bound to an absent link they are request-level 422/ValueError.
Batch and stream apply the chain at resolve level (one link serves the
whole batch; restarting a committed stream would be dishonest).
Every link tried lands in `attempts` on the response
(`{backend, ok, error_class, latency_ms}`, omitted for single-link
requests) and on the completion record — the failover trace is part of
the sealed evidence. SDK takes `fallbacks=[...]`,
`HarnessClient`/`complete_many`/`stream_complete` the same; the CLI
takes repeatable `--fallback`.

**Usage accounting:** when the backend reports a usage block
(`prompt_tokens`/`completion_tokens`/`total_tokens` for
OpenAI-compatible endpoints), `POST /harness/complete` returns it as
`usage` and `POST /harness/complete/batch` returns the batch's sum as
`usage_total`. Counts are never synthesized: an endpoint that stays
silent yields `usage: null`, and because one backend instance serves a
whole batch under worker threads, the batch level is a before/after
delta — per-item attribution would be a lie. On streaming, a provider
that emits a `usage` chunk (the OpenAI `stream_options` convention —
BYOK only ever captures it opportunistically; the request never asks
for it, so strict providers that would 400 on unknown fields stay
compatible) lands it on the SSE `final` frame's `usage` field — null
when the provider stays silent. The same values land on
`CompletionResult.usage` in the SDK and `HarnessClient`.

**Sampling controls:** `complete`, `complete/batch`, and
`complete/stream` accept flat decode fields `temperature` (0–2),
`top_p` (0,1], `max_tokens` (1–262144), and `seed` (≥0) — model-level
422s on range violations. They resolve into a `SamplingParams` value
whose `body_fields()` is the wire dict: `temperature` always ships
(default `0.0` — eval/teacher runs stay deterministic and replayable),
while `top_p`/`max_tokens`/`seed` ship only when declared, so providers
that don't know a field never see it. The resolved set is echoed on the
response (`sampling`), the SSE `final` frame, and the `CompletionRecord`
— the decode configuration is part of the sealed evidence. SDK
`complete`/`complete_many`/`stream_complete` and `HarnessClient` take
the same names as flat kwargs; the CLI takes
`--temperature --top-p --max-tokens --seed`.

**Completion observability:** `GET /metrics` carries per-backend outcome
counters (`fx1_complete_total{backend,outcome}`), a cumulative
latency histogram (`fx1_complete_latency_ms_bucket{le=…}`, `_sum`,
`_count`), and a usage ledger
(`fx1_complete_tokens_total{backend,kind="prompt|completion|total"}`
+ `fx1_complete_usage_calls_total{backend}`) over attempted model
calls — breaker rejections and pre-call validation never land in it.
`usage_calls` counts calls that reported usage at all, so a silent
provider reads as 0 tokens and 0 calls, distinct from a zero bill.
The JSON view exposes the same data under `complete.<backend>`.

**Per-call evidence:** every gated call (sync, stream, batch item —
success or failure) lands in a bounded in-process log of 256 records.
A record carries `completion_id`, backend, model, ok, `latency_ms`,
timestamp, usage, error class, and sha256 hashes of the request
messages and the pre-citation output — evidence handles, never
content. The id returns on `CompleteResponse.completion_id`,
per-item on batch results, on the stream's `final` frame, and as the
`X-Fx1-Completion-Id` response header (idempotency replays echo the
original id). Probes never log. In-process, `Fx1Harness.completions()`
/ `.completion(id)` return the same records.

**Sealed exports:** `GET /harness/completions/{id}/receipt` returns the
record wrapped as a `fx1_completion_record.v1` sealed document —
`{kind, schema, git_revision, data_label:"OPS", research_only,
live_pnl_claim:false, record, receipt_sha256}`. Exports are
deterministic per record and verify through `POST /receipts/verify` or
`verify-research` like any other receipt; tampering with the record
breaks the seal. The claim is "these bytes were the recorded call" —
prompt/output are sha256 evidence handles, never content. In-process,
`Fx1Harness.completion_receipt(id)` mints the twin document (each
surface seals its own record; the sha256s are the cross-surface claim).
Async jobs seal the same way: `GET /harness/jobs/{id}/receipt` returns a
`fx1_job_record.v1` document whose `record.result` carries the run with
stdout/stderr digested (never content); in-process runs seal standalone
through `Fx1Harness.run_receipt(result)` as `fx1_run_result.v1` — the
same digested shape the job record embeds.

## Routes

| Route | Purpose |
|---|---|
| `GET /health` | presence booleans only — never env values (unauthenticated) |
| `GET /ready` | readiness: `200 {ready, inflight}` until drain latches → `503` |
| `GET /metrics` | ops counters; `?format=prom` or `Accept: text/plain` renders Prometheus exposition |
| `GET /harness/version` | `{"api_version", "fx1_version"}` — negotiate before sending work |
| `GET /harness/capabilities` | `{"features", "limits", "backends", "roles"}` — self-configure batch caps, retry budgets, stream use |
| `GET /harness/backends` | per-backend liveness: `configured`, `circuit_open`, `cooldown_remaining_s`, `consecutive_failures`, plus `last_probe` — the most recent deep-health verdict (`ok`, `latency_ms`, `checked_at`, `error_class`; null before the first probe), so scrapes read health without spending a live call |
| `POST /harness/backends/{name}/probe` | deep health: one live gated completion through the real resolver → `{ok, model, latency_ms, error, error_class}`; an unconfigured backend is a verdict (`ok:false, error_class:"backend_unavailable"`), not a wire fault. BYOK probes test the caller's endpoint inline; probes bypass and never feed the breaker, and land under `probe:<name>` in metrics so they can't pollute completion SLOs |
| `POST /harness/gate/check` | pre-flight text through the honesty gate → `{ok, error}`; a refusal is a verdict, not a wire fault. Advisory: not slot-gated, stays up during drain, never metered — also `Fx1Harness.check_text` / `HarnessClient.check_text` / `fx1 harness check-text` |
| `GET /harness/completions` | newest-first window on the per-call completion log (`?limit≤256`, `?backend=`); `Fx1Harness.completions` / `HarnessClient.completions` / `fx1 harness completions` |
| `GET /harness/completions/{id}` | one logged call by `completion_id` → record or `404 not_found`; `Fx1Harness.completion` / `HarnessClient.completion` / `fx1 harness completion` |
| `GET /harness/completions/{id}/receipt` | the logged call sealed as a `fx1_completion_record.v1` document → verify via `POST /receipts/verify`; `Fx1Harness.completion_receipt` / `HarnessClient.completion_receipt` / `fx1 harness completion --receipt` |
| `GET /harness/commands` | registered commands, optional `?role=` filter |
| `POST /harness/runs` | synchronous command run |
| `POST /harness/complete` | gated model completion (sync) — carries `completion_id`, `latency_ms` (per-call wall clock; replays report the original) |
| `POST /harness/complete/batch` | up to 64 conversations over one shared backend; per-item `completion_id` + `latency_ms` |
| `POST /harness/complete/stream` | SSE `token` frames + `final` (`completion_id`, `latency_ms`, `usage` when the provider reports it) + `[DONE]` — the gate runs before any frame leaves |
| `POST /harness/jobs` | async run → `202 {job_id}` |
| `POST /harness/jobs/batch` | up to 64 submissions, per-item `{error, code}` outcomes |
| `GET /harness/jobs` | list/filter (`?status=`, `?limit=`, `?offset=`) |
| `GET /harness/jobs/{id}` | poll status/result/error |
| `GET /harness/jobs/{id}/receipt` | terminal job sealed as `fx1_job_record.v1` (streams digested, callback URL hashed) → verify via `POST /receipts/verify`; `HarnessClient.job_receipt` / `fx1 harness job --receipt`; in-process runs seal via `Fx1Harness.run_receipt` |
| `GET /harness/jobs/{id}/events` | SSE frame per state change until terminal |
| `DELETE /harness/jobs/{id}` | cancel (queued → cancelled fires the webhook) |
| `POST /harness/evals` | submit a seeded eval suite against a backend chain → `202 {eval_id}` (`Location` header, `Idempotency-Key` dedupe); `HarnessClient.submit_eval` / `fx1 harness eval --remote` |
| `GET /harness/evals` | newest-first eval inventory (`?status=`, `?suite=`, `?limit≤256`); `HarnessClient.list_evals` / `fx1 harness evals` |
| `GET /harness/evals/{id}` | poll the live record (status/report/attempts/sampling pin); `HarnessClient.eval_status` |
| `GET /harness/evals/{id}/receipt` | terminal record sealed as `fx1_eval_record.v1` → `POST /receipts/verify` (`409 eval_not_terminal` until terminal); `HarnessClient.eval_receipt` / `fx1 harness eval-status --receipt` |
| `DELETE /harness/evals/{id}` | cooperative cancel of a queued eval (running/terminal → 409); `HarnessClient.cancel_eval` / `fx1 harness eval-cancel` |
| `POST /harness/drain` | latch draining; `?wait_s=` blocks until inflight empties |
| `GET /v1/models` | OpenAI `list` envelope: `fx1` + the backend names |
| `POST /v1/chat/completions` | OpenAI-compatible gated completion (JSON or SSE `stream:true`) |
| `POST /receipts/verify` | verify one receipt payload |
| `POST /receipts/verify/batch` | up to 64 in one call, order-preserved |
| `GET /receipts` | index the store: `sha256` → filename |
| `GET /receipts/{sha256}` | fetch the sealed receipt by content hash — verbatim bytes, `ETag` = the hash, `Cache-Control: public, immutable`, `X-Fx1-Receipt-Valid` from live re-verify |

The store is content-addressed, so `If-None-Match: "<sha256>"` (or `*`)
answers `304` without a body — receipts are immutable, a cached copy is
always current. The same store is reachable on every surface, one
contract: `Fx1Harness(receipts_dir=…).receipts()` / `.receipt(sha256)`
in-process, `HarnessClient.receipts()` / `.receipt(sha256)` over the
wire (`.receipt()` returns the document plus the server's live
re-verify flag), `HarnessApiClient.receipts()` / `.receipt()` in TS,
and `fx1 harness receipts` / `fx1 harness receipt <sha256>` on the CLI
(local `--receipts-dir` or `--remote`). Misses map to the same errors
everywhere: `KeyError`/`404` unknown hash, `ValueError`/`422` malformed
digest, store-absent → `FileNotFoundError` locally / `503
receipts_unavailable` on the wire.sha.

`receipt_hashes` on a completion request cites that store: when it is
mounted, every cited hash must resolve — an unknown or malformed hash is
`422 receipt_not_found` before the backend call (never breaker-counted),
on sync, batch, and stream alike. With no store mounted the citations
stay advisory footnotes (the response footer already tells consumers to
verify externally).

`GET /openapi.json` is codegen-grade: every operation carries a stable
`operation_id` + tag (`quality/fx1_openapi_surface.json` pins the
surface — `paths` + `schema_sha256`).

## OpenAI-compatible ingress (`/v1`)

`POST /v1/chat/completions` is a translation layer, not a second
pipeline: the request is mapped onto `CompleteRequest` and run through
the same `complete` path — honesty gate, breaker, fallback chain,
metering, completion log, and sealed per-call receipt all apply
unchanged. Every response carries `X-Fx1-Completion-Id`, which links it
to `GET /harness/completions/{id}` and its sealed
`fx1_completion_record.v1` receipt.

- **Backend selection:** the `fx1` extension's `backend` field >
  `X-Fx1-Backend` > a `model` naming a backend
  (`hosted_k3`/`local_fx1`/`byok`) > `hosted_k3`. Anything else in
  `model` is the default link (or the BYOK upstream model when BYOK
  headers are present).
- **BYOK:** the `fx1` extension's `byok = {base_url, api_key, model}`,
  or the `X-Fx1-Byok-Base-Url`/`X-Fx1-Byok-Api-Key`/`X-Fx1-Byok-Model`
  headers (base-url without api-key is 400). Header BYOK with a
  non-backend `model` field uses it as the upstream model (e.g.
  `gpt-4o`).
- **Chain knobs:** the `fx1` extension's `fallbacks` /
  `X-Fx1-Fallbacks` (CSV), `checkpoint_dir` / `X-Fx1-Checkpoint-Dir`,
  `timeout_s`, and `receipt_hashes`.
- **Streaming:** `stream: true` returns SSE `chat.completion.chunk`
  frames — a `role` delta, ~64-char content deltas on whitespace
  boundaries, a `finish_reason: "stop"` frame, an optional
  `choices: []` + `usage` chunk (`stream_options.include_usage`), then
  `data: [DONE]`. The gate runs before the first delta — no ungated
  bytes ever ship; a refusal is an OpenAI-shaped 502, not a truncated
  stream.
- **Fail-closed surface:** tool calls (`tools`, `functions`,
  `tool_calls`, `tool_call_id`), `n != 1`, non-text `response_format`,
  `logprobs`, `logit_bias`, `stop`, penalties, `modalities`, `audio`,
  `prediction`, `reasoning_effort`, `service_tier`, `store`, `metadata`,
  and `None`/non-text-part content are all rejected — nothing is
  silently dropped.
- **Error envelope:** under `/v1`, every error — validation, auth
  (401/403), rate limit (429), body cap (413), over-capacity (503),
  honesty refusal (502) — returns OpenAI's
  `{error: {message, type, param, code}}` shape with OpenAI's type names
  (`invalid_request_error`, `authentication_error`,
  `rate_limit_error`, `server_error`, `service_unavailable`).
- **Auth:** when `FX1_API_KEY` is set, `/v1` also accepts the OpenAI
  `Authorization: Bearer` header in place of `X-API-Key`.

Client-side: `HarnessClient.chat_completion` /
`chat_completion_stream` / `list_models` in Python;
`HarnessApiClient.chatCompletion` / `chatCompletionStream` /
`listModels` in TS. Any OpenAI SDK works directly — point it at the
harness `base_url` and use `model: "fx1"`.

The same surface exists in-process: `Fx1Harness.openai_chat(request)`
accepts the same request body dict (or a parsed
`OpenAIChatRequest`) and optional `X-Fx1-*` header kwargs, and returns
the `chat.completion` envelope plus the completion-log id;
`openai_chat_stream(request)` returns the identical
`chat.completion.chunk` payload sequence (minus SSE framing);
`openai_models()` is the `/v1/models` inventory. Both surfaces
translate through `fx1.serve.openai_compat` — one validation object,
one backend-precedence order, one error taxonomy — and the parity
audit pins envelope, chunk stream, rejection classes, and
completion-log linkage identical across them.

## Auth & safety

- `fx1 harness serve` binds **loopback-only** unless `FX1_API_KEY` is
  set; with a key, every route except `/health` requires
  `X-API-Key` (constant-time compare). An empty key equals unset — never
  a bypass.
- Request bodies over 1 MiB are refused `413`; `/health` leaks only
  presence booleans.
- The honesty gate runs before output bytes reach the caller — a
  refusal is a structured `502`, not a truncated stream. Cited receipts
  arrive as a provenance footer.
- Every response carries `X-Request-ID`, `X-Fx1-Api-Version`,
  `X-Content-Type-Options: nosniff`, `Cache-Control: no-store` (a
  route's own deliberate caching policy — e.g. immutable receipts —
  wins over the default),
  `Referrer-Policy: no-referrer`; `Retry-After` is declared on 429/503
  and `Location` on the job-submit 202 — all of these are declared on
  the OpenAPI spec itself, so generated clients see them typed.
  `X-RateLimit-*` declarations appear only on builds where the limiter
  is enabled.
- `circuit_breaker_threshold` + `circuit_reset_s` on `HarnessClient`
  fast-fail a dead peer (`HarnessTransportError`) and half-open after
  the reset window.

## Idempotency

`Idempotency-Key` (header) or `idempotency_key` (in-body, batch lanes)
dedupes `POST /harness/runs`, `/harness/jobs(·/batch)`,
`/harness/complete`, and `/harness/complete/batch`. Semantics, identical
on every deduped route:

- Same key + same body → the cached response replays with
  `replayed: true` — **before** the drain/capacity gates, and without
  re-resolving the backend (a retried `complete` never burns a second
  upstream generation).
- Same key + different body → `409`. Keys over 256 chars → `400`.
  Errors are never cached.
- Stores are FIFO-bounded (`FX1_API_IDEM_MAX`, 1024) and typed per
  response model — a key can never replay the wrong-typed body.
- `/harness/complete/stream` is deliberately undeduped: an SSE stream
  can't be replayed.
- `HarnessClient.run(...)` auto-mints a uuid4 key; `complete`/
  `complete_many`/`submit_run`/`submit_batch` take
  `idempotency_key=` explicitly. With `retry_writes=True` the client
  also retries write verbs — safe because the server dedupes.

## Version negotiation

The wire contract is `fx1.serve.contract.API_VERSION` (a single source
shared by server and client). Every response stamps it as
`X-Fx1-Api-Version`; `GET /harness/version` returns it with the package
version.

```python
client.check_compat()  # raises HarnessCompatError on mismatch
client.check_compat(strict=False)  # returns {"compatible": bool, ...}
client.last_api_version  # stamped header from the last response
```

`fx1 harness compat --remote URL` prints the report and exits 1 on
mismatch — deploy pipelines gate on it before routing traffic.

`fx1 harness capabilities --remote URL` (`client.capabilities()` /
`HarnessApiClient.capabilities()`) returns the server's declared feature
set (`features`: idempotency, SSE, webhooks, batch, jobs, drain,
streaming), its effective limits (`limits`: batch caps, store bounds,
`rate_limit_rps`, `sse_keepalive_s`, body/job-result byte caps), which
backends are configured (`backends`, booleans only), and the registered
command roles (`roles`). Clients self-configure from this instead of
hardcoding server internals.

## Async jobs & webhooks

`POST /harness/jobs` admits under the drain + `max_inflight` gates and
executes on a bounded pool — the slot is held for the job's whole life,
so saturation is an honest 503, never unbounded buffering. Stored
stdout/stderr cap at 1 MiB each (`*_truncated` flags). Options:

- **`callback_url`** — POSTs the full job record on every terminal
  transition (succeeded/failed from the worker, cancelled from DELETE).
  Transient faults (network errors, 5xx) retry up to 3 times with
  capped backoff; a 4xx is a definitive rejection and never retried.
  The job record exposes `callback_status` (`delivered`/`failed`),
  `callback_attempts` (deliveries tried), and `callback_error`.
- The 202 response carries `Location: /harness/jobs/{job_id}` so the
  status endpoint is discoverable without composing the path client-side.
- **`callback_secret`** — HMAC-SHA256 signs the payload:
  `X-Fx1-Webhook-Timestamp` + `X-Fx1-Webhook-Signature` over
  `<ts>.<body>`; receivers verify with
  `fx1.serve.webhooks.verify_webhook` (constant-time, 300s freshness).
  The secret is a `PrivateAttr` — never serialized.
- **Lifespan** — on shutdown the gate drains, queued jobs flip to
  `cancelled` (firing their webhooks), the executor releases pending
  futures; running jobs finish bounded by their command timeout.

Poll with `GET /harness/jobs/{id}`, or stream
`/harness/jobs/{id}/events` (`HarnessClient.stream_job`,
`wait_run_stream`, `fx1 harness watch`).

## Eval submissions

`POST /harness/evals` runs the seeded eval banks (capability,
calibration, tooluse, retrieval, ts_reasoning, ext_bench,
options_reasoning — `GET /harness/capabilities` lists them) against any
backend chain as an async job: same drain + `max_inflight` admission,
same bounded executor, same terminal-receipt contract — but the payload
is the eval suite, not a lab command. The model under test is the
resolved chain (`backend` + `fallbacks`, `byok`/`checkpoint_dir` bind
per-link exactly like completions); every suite call is metered under
`eval:{suite}:{backend}` (judged suites meter the grader separately as
`eval:{suite}:judge:{backend}`) so eval spend is visible, never
conflated with user traffic.

- **Decode pin** — evals run under `{"temperature": 0.0}`; the record's
  `sampling` field states it so the sealed receipt carries the decode
  config.
- **Validation** — unknown suite → 422; `judge_backend`/`judge_byok` on
  a non-judge suite → 422; `judge_byok` bound to anything but
  `judge_backend="byok"` → 422.
- **Record** — `report` is the suite's serialized report (pydantic or
  dataclass — a runner returning anything else fails closed as
  `failed`), `attempts` is the chain trace, `seed` is the bank seed.
- **CLI/SDK twins** — `fx1 harness eval <suite>` runs in-process through
  `Fx1Harness.run_eval` (or `--remote` submits + waits);
  `Fx1Harness.evals`/`eval_record`/`eval_receipt` mirror the wire reads;
  the TS client exposes `submitEval`/`eval`/`evals`/`evalReceipt`/
  `cancelEval`/`waitEval`.
- **Webhooks** — evals carry the same `callback_url`/`callback_secret`
  contract as jobs: the terminal record is POSTed on every terminal
  transition (succeeded/failed from the worker, cancelled from DELETE or
  lifespan drain), `callback_secret` HMAC-signs it
  (`X-Fx1-Webhook-Signature`, verified with
  `fx1.serve.webhooks.verify_webhook`), and delivery state lands on the
  record (`callback_status`/`callback_attempts`/`callback_error`).

## Ops knobs

CLI flags on `fx1 harness serve`, falling back to env, fail-closed on
out-of-range values:

| Flag | Env | Default | Meaning |
|---|---|---|---|
| `--max-inflight` | `FX1_API_MAX_INFLIGHT` | 16 | concurrent heavy requests; 503 + `Retry-After` when saturated |
| `--job-max` | `FX1_API_JOB_MAX` | 1024 | job-store capacity (LRU evict drops key backrefs) |
| `--idem-max` | `FX1_API_IDEM_MAX` | 1024 | idempotency-store capacity |
| `--sse-keepalive-s` | `FX1_API_SSE_KEEPALIVE_S` | 15 | `: keepalive` comment cadence; 0 disables |
| `--rate-limit-rps` | `FX1_API_RATE_LIMIT_RPS` | 0 (off) | per-client token bucket → 429 + `Retry-After`; every response also carries `X-RateLimit-Limit`/`Remaining`/`Reset` while the limiter is on. Public paths (`/health`) are exempt — LB probes never consume the client budget |
| `--gzip-min-bytes` | `FX1_API_GZIP_MIN_BYTES` | 1024 | gzip only when the client advertises it; 0 disables |
| `--cors-origins` | `FX1_API_CORS_ORIGINS` | (off) | comma-separated browser origins for CORS; each must be a scheme+host URL, `*` and non-http(s) refused; preflights bypass the API-key gate (they carry no credentials), every preflight reflects the `expose` list of stamped headers |
| `--breaker-threshold` | `FX1_API_BREAKER_THRESHOLD` | 5 | consecutive call faults that open a backend's circuit; 0 disables. While open, calls fast-fail `503 backend_unavailable` + `Retry-After` without burning an inflight slot; a single half-open probe is admitted after cooldown and closes the circuit on success. Resolution faults that surface as 503 count; client errors (404/422), capability gaps (501), and honesty-gate refusals never do |
| `--receipts-dir` | `FX1_API_RECEIPTS_DIR` | `receipts` | sealed-receipt store backing `GET /receipts*` — `503 receipts_unavailable` when absent |
| `--breaker-cooldown-s` | `FX1_API_BREAKER_COOLDOWN_S` | 30 | seconds an open circuit fast-fails before admitting a probe |

`POST /harness/drain` is the one-way graceful-exit latch: work routes
refuse `503 draining` while health/metrics/version/verify stay live so
an orchestrator watches `inflight` bleed to zero — or passes
`?wait_s=` for a blocking drain verdict (`drained` reports which
outcome happened).

## Error envelope

Non-2xx responses carry `{"detail": str, "code": str}` — the code is the
status-derived name (`not_found`, `validation`, `conflict`,
`too_large`, `too_many_requests`, `unauthorized`, `forbidden` …) or a
route-specific one (`backend_failure`, `backend_unavailable`,
`draining`, `honesty_gate`, `over_capacity`).
`HarnessClient` maps them back to the SDK's exception classes, so
`except Fx1HonestyError` works identically in-process and over the wire.
