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
| OpenAI-compatible | `GET /v1/models`, `POST /v1/chat/completions`, `/v1/responses`, `/v1/embeddings`, `/v1/moderations`, `/v1/files`, `/v1/batches` | drop-in for OpenAI SDKs / existing toolchains — set `base_url` to the harness |
| Anthropic-compatible | `POST /v1/messages`, `/v1/messages/count_tokens`, `/v1/messages/batches`, anthropic `GET /v1/models{,/{id}}` under `anthropic-version` | drop-in for Anthropic SDKs — set `base_url` to the harness; `x-api-key` authenticates unchanged |
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
`--temperature --top-p --max-tokens --seed`. Provider hints ride the
same surfaces: `reasoning_effort`, `service_tier`, `verbosity`
(`low`/`medium`/`high`), `prompt_cache_key` (≤128 chars), and
`prompt_cache_retention` (`in-memory`/`24h`) — enums fail closed
(422 on the wire, `ValueError` in-process) and the declared values
land verbatim in `body_fields()` for BYOK links that support them.

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
timestamp, usage, error class, `key_id` (the caller's key fingerprint —
`"env"` for the bootstrap credential, a managed-key id, null on
loopback-dev), and sha256 hashes of the request
messages and the pre-citation output — evidence handles, never
content. The id returns on `CompleteResponse.completion_id`,
per-item on batch results, on the stream's `final` frame, and as the
`X-Fx1-Completion-Id` response header (idempotency replays echo the
original id). Probes never log. In-process, `Fx1Harness.completions()`
/ `.completion(id)` return the same records.

**Usage aggregation:** `GET /harness/usage` rolls the completion ring
into a `UsageReport` — totals (requests/ok/errors/`usage_reported`,
prompt/completion/total token sums, mean latency) plus `by_backend`,
`by_model`, and `by_key` splits — `by_key` buckets calls under the
managed-key fingerprint that made them (the env bootstrap credential
lands under `"env"`, loopback-dev calls under `"(none)"`), with
`?backend=`/`?model=`/`?key_id=`/`?since=`/`?until=` filters. The
ring is bounded: `records_seen` counts only live records,
`records_dropped` + `ring_cap` disclose evictions, and provider-
specific counters (e.g. `cached_tokens`) land in `other_usage` rather
than dropping silently. `since > until` fails closed 400. The
in-process twin `Fx1Harness.usage()` aggregates the same way.

**Response-side seal:** responses that carry `X-Fx1-Completion-Id` also
carry `X-Fx1-Receipt-Sha256` — the `receipt_sha256` of the sealed
`fx1_completion_record.v1` document, so a client pins the evidence
without a second fetch (replays echo the original seal while the record
is in the log; `/harness/complete/stream` carries the digest in the
`final` frame instead).

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
| `POST /harness/score` | run text through the deterministic reward contract → `{object:"list", data:[{object:"score", index, total, components, violations}]}` — a string scores one input, a list scores each (cap 128); honesty violations cap `total` at `-10` and empty text scores `0`. Advisory like the gate pre-flight: never touches a backend, stays up during drain — also `Fx1Harness.score` / `HarnessClient.score` / `HarnessApiClient.score` / `fx1 harness score` |
| `GET /harness/completions` | newest-first window on the per-call completion log (`?limit≤256`, `?backend=`); `Fx1Harness.completions` / `HarnessClient.completions` / `fx1 harness completions` |
| `GET /harness/completions/{id}` | one logged call by `completion_id` → record or `404 not_found`; `Fx1Harness.completion` / `HarnessClient.completion` / `fx1 harness completion` |
| `GET /harness/completions/{id}/receipt` | the logged call sealed as a `fx1_completion_record.v1` document → verify via `POST /receipts/verify`; `Fx1Harness.completion_receipt` / `HarnessClient.completion_receipt` / `fx1 harness completion --receipt` |
| `GET /harness/usage` | usage accounting over the completion ring — totals + `by_backend`/`by_model`/`by_key` splits, `?backend=`/`?model=`/`?key_id=`/`?since=`/`?until=` filters; `records_dropped`+`ring_cap` disclose truncation; `Fx1Harness.usage()` / `HarnessClient.usage` / `fx1 harness usage [--key-id]` |
| `POST /harness/keys` | mint a managed API key → `201` mint record; the raw `key` (`fx1k_…`) is shown **only** in this response — the store keeps sha256 only. `{name?, admin?, rpm?, ttl_s?}`: `admin` keys may manage keys; `rpm` bounds the key to a fixed 60 s request window (over-limit → `429 rate_limited` + `Retry-After`) and every response to a rpm-declared key carries its standing budget as `X-RateLimit-Limit-Requests` / `X-RateLimit-Remaining-Requests` / `X-RateLimit-Reset-Requests` (OpenAI's header names — SDKs and dashboards read them unmodified; keys without `rpm`, the env credential, and loopback emit none — no false scarcity); `ttl_s` bakes an `expires_at` — a dead credential fails closed like a revoked one. `scopes` bounds the key to `read` (safe methods), `write` (data-plane mutations), and/or `admin` (key management + `/harness/drain`) — an out-of-scope call is refused `403 insufficient_scope` in the path's own error grammar; unset keeps `[read, write]` (plus `admin` for admin keys) and `admin:true` unions the admin scope onto an explicit list. `max_requests`/`max_tokens` declare hard budgets — an exhausted key answers `429 quota_exceeded` with **no** `Retry-After` (a budget never clears inside a call, so SDKs/TS client do not retry it; it maps to `HarnessTransportError`, not the auth-error class). `max_requests` counts authenticated calls; `max_tokens` is charged post-response off provider-reported usage (a budget gates the *next* call — the crossing call completes; the env credential and in-process SDK are unmetered). Requires the bootstrap credential or loopback. `Fx1Harness.key_create` / `HarnessClient.key_create` / `fx1 harness key-create [--admin] [--rpm] [--ttl-s] [--scope read|write|admin …] [--max-requests N] [--max-tokens N]` |
| `GET /harness/keys` | every key's fingerprint id + metadata (`prefix`, `admin`, `scopes`, `rpm`, `expires_at`, `enabled`, `uses`, `last_used_at`, `max_requests`, `max_tokens`, `tokens_used`) — never secrets or hashes. `Fx1Harness.keys` / `HarnessClient.keys` / `fx1 harness keys` |
| `GET /harness/keys/{id}` | one key's record → `404 key_not_found`. `Fx1Harness.key_get` / `HarnessClient.key_get` / `fx1 harness key-get` |
| `DELETE /harness/keys/{id}` | tombstone a key (`enabled:false` + `revoked_at`) — auth with it fails closed immediately; the record survives for audit. `404 key_not_found`, `409 key_revoked`. `Fx1Harness.key_revoke` / `HarnessClient.key_revoke` / `fx1 harness key-revoke` |
| `GET /harness/keys/{id}/usage` | one key's usage card — live counters (`uses`, `tokens_used`, `last_used_at`), declared budgets with derived headroom (`requests_remaining` / `tokens_remaining`, `null` when unbounded), the rpm window's live `window_remaining` / `window_reset_s` (`null` when no rpm), and `served` — the completion-ring spend split (`calls`/`prompt_tokens`/`completion_tokens`/`total_tokens` + `by_backend`). `log_cap` / `log_dropped` bound `served` honestly: the ring is bounded, so a `log_dropped > 0` card is a lower bound. Admin scope. `404 key_not_found`. `Fx1Harness.key_usage` / `HarnessClient.key_usage` / `fx1 harness key-usage` |
| `GET /harness/self` | the calling credential's own card — `credential: managed` (with the full usage card under `key`) for minted keys, `env` for the bootstrap credential, `none` for loopback dev mode; `metered` reports whether budgets/quota bind this credential (managed `true`, env/loopback `false`), `scopes` the effective set. Only needs `read` scope — a key watches its own budgets without admin. `Fx1Harness.self_usage` / `HarnessClient.self_usage` / `fx1 harness self` |
| `GET /harness/commands` | registered commands, optional `?role=` filter — `Fx1Harness.commands` / `HarnessClient.commands` / `fx1 harness commands [--role]` |
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
| `GET /harness/evals/{base}/diff/{cand}` | promotion-gate diff over two terminal records — per-task pass/fail transitions, gate move, `by_kind` deltas, exact sign-test `significance` on the discordant pairs, `verdict` (`comparable` needs same suite+seed and no bank-stamp mismatch; `404` unknown id, `409 eval_not_terminal`); `Fx1Harness.eval_diff` / `HarnessClient.diff_evals` / `client.diffEvals` / `fx1 harness eval-diff` |
| `POST /harness/drain` | latch draining; `?wait_s=` blocks until inflight empties |
| `GET /v1/models` | OpenAI `list` envelope: `fx1` + the backend names |
| `GET /v1/models/{id}` | `models.retrieve` — unknown id is `404 model_not_found`. Under an `anthropic-version` request header the same routes answer Anthropic's grammar instead: `models.list` → `{data:[{type:"model",id,display_name,created_at}], first_id, last_id, has_more}` with `limit≤1000` + `after_id`/`before_id` positional cursors (back-pagination returns the window tail); `models.retrieve` → one `{type:"model"}` card or `404 not_found_error`. `Fx1Harness.anthropic_models`/`anthropic_model` / `HarnessClient.anthropic_models`/`anthropic_model` / `client.anthropicModels`/`anthropicModel` / `fx1 harness models --anthropic` / `fx1 harness model --anthropic` |
| `DELETE /v1/models/{id}` | `models.delete` — unregister an `ft:` name (`{id, object:"model", deleted:true}`); built-in link ids refuse `400`, unregistered names `404`, the tombstone journals so restarts never resurrect it |
| `POST /v1/chat/completions` | OpenAI-compatible gated completion (JSON or SSE `stream:true`) |
| `POST /v1/completions` | Legacy OpenAI `completions.create` (pre-chat `text_completion` surface) over the same gated pipeline — `prompt` string or array (each element is its own gated turn), `n` ≤ 8 flat `choices`, `echo` prepends the prompt, `stop` ≤ 4; `stream:true` emits `text_completion` chunk frames ending `[DONE]` (`Last-Event-ID` resumes). `suffix`/`best_of`/`logprobs` fail closed 422 — never silently dropped. Not stored (`store` tolerated, ignored); `Idempotency-Key` dedupes like the chat surface. `Fx1Harness.openai_completion`/`openai_completion_stream` / `HarnessClient.create_completion`/`create_completion_stream` / `client.completionsCreate`/`completionsCreateStream` / `fx1 harness text-completion` |
| `POST /v1/messages` | Anthropic Messages surface over the same gated pipeline — `messages` are `user`/`assistant` turns (string or `text`/`tool_use`/`tool_result` blocks), `system` a string or text blocks, `tools` the `{name,description,input_schema}` shape, `tool_choice` `{type: auto|any|tool|none[, name]}`; `max_tokens` required, `stop_sequences` ≤4. The answer is Anthropic's `message` object (`id: msg_*`, `content` text/tool_use blocks, `stop_reason` mapped `end_turn`/`max_tokens`/`tool_use`/`refusal`, `usage.{input,output}_tokens`); `stream:true` emits the Anthropic SSE grammar (`message_start`/`ping`/`content_block_*`/`message_delta`/`message_stop`, frames carry `id:`+`event:`; `Last-Event-ID` resumes a keyed stream). `Idempotency-Key` dedupes like the OpenAI surface. Anthropic-only knobs the pipeline cannot honor (`top_k`, `thinking`, `cache_control`, image/document blocks) fail closed — never silently dropped; every refusal on this surface carries `{type:"error",error:{type,message}}`. `store` is forced off (no retrieval twin). `Fx1Harness.anthropic_message`/`anthropic_message_stream` / `HarnessClient.create_message`/`create_message_stream` / `client.messagesCreate`/`messagesCreateStream` / `fx1 harness message` |
| `POST /v1/responses` | OpenAI Responses surface — `input` string/items, `instructions`, `reasoning`, `text.format`; SSE `stream:true` emits the `response.*` event grammar |
| `POST /v1/embeddings` | OpenAI `embeddings.create` — verbatim provider forward, 501 when the link has no embeddings channel |
| `POST /v1/moderations` | OpenAI `moderations.create` shape over the honesty gate → per-input `{flagged, categories, category_scores, category_applied_input_types}` + content-derived `modr-<sha256>` id; categories are the gate's three checks (`forbidden_headline_metric`, `live_or_synthetic_claim`, `unlabeled_synthetic`) with deterministic 0/1 scores. Advisory: never touches a backend, stays up during drain — also `Fx1Harness.moderate` / `HarnessClient.moderate` |
| `POST /v1/files` | multipart upload of a JSONL (`purpose=batch` or `fine-tune`) |
| `GET /v1/files` / `GET /v1/files/{id}` | list / retrieve uploaded + output files |
| `GET /v1/files/{id}/content` | raw bytes — input JSONL in, batch result JSONL out |
| `DELETE /v1/files/{id}` | evict a stored file |
| `POST /v1/uploads` | open a chunked-upload intent (`purpose`, `filename`, declared `bytes`, `mime_type`) → `upload_*` pending record — `Fx1Harness.upload_create` / `HarnessClient.upload_create` / `client.uploadCreate` / `fx1 harness upload` |
| `POST /v1/uploads/{id}/parts` | multipart `data` field appends a part → `part_*` record; cumulative bytes may never exceed the declared total; parts on a terminal record are `409 upload_terminal` |
| `POST /v1/uploads/{id}/complete` | assemble `part_ids` in the caller's order into a `file-*` record (optional `md5` checksum is verified pre-mint — a failed check mints no file and leaves the intent pending); `assembled == declared` enforced |
| `POST /v1/uploads/{id}/cancel` | terminal cancel — replays 200 when already cancelled |
| `POST /v1/batches` | submit an input file as one batch (`endpoint` = `/v1/chat/completions`, `/v1/responses`, or `/v1/embeddings`) — async over the jobs channel |
| `GET /v1/batches` / `GET /v1/batches/{id}` | list (`?limit≤100`, `?after=`) / poll status + `request_counts` |
| `POST /v1/batches/{id}/cancel` | cooperative cancel — partial output still lands in `output_file_id` |
| `POST /v1/messages/batches` | Anthropic Message Batches — `requests[]` ride inline (`{custom_id, params}` each a full `/v1/messages` body, unique `custom_id` ≤256 chars, `stream` in a params refuses the whole submit), `Idempotency-Key` dedupes; the `message_batch` envelope (`msgbatch_*`, `processing_status: in_progress|canceling|ended`, `request_counts` all-`processing` until end, `expires_at` = created+24h, `results_url` null until ended) — `HarnessClient.create_message_batch` / `client.messageBatchesCreate` / `fx1 harness message-batch` (in-process twin `Fx1Harness.anthropic_batch` runs synchronously) |
| `GET /v1/messages/batches` / `GET /v1/messages/batches/{id}` | list (`?limit≤100`, `?after_id`/`?before_id` cursors) / poll status — a batch past `expires_at` ends on read with `expired` rows for the unfinished tail |
| `POST /v1/messages/batches/{id}/cancel` | cooperative cancel → `canceling`; in-flight items complete, the tail lands `canceled` rows; cancel on `ended` 400s — `HarnessClient.cancel_message_batch` / `client.messageBatchesCancel` / `fx1 harness message-batch-cancel` |
| `DELETE /v1/messages/batches/{id}` | tombstone an ended batch (`{id, type: "message_batch_deleted"}`) — refuses mid-flight with 400 — `HarnessClient.delete_message_batch` / `client.messageBatchesDelete` / `fx1 harness message-batch-delete` |
| `GET /v1/messages/batches/{id}/results` | `application/jsonl` — one `{custom_id, result}` row per request (`result.type` = `succeeded|errored|canceled|expired`; `succeeded` carries the full `message`, `errored` carries the inner `{type, message}` error object); 400 until ended — `HarnessClient.message_batch_results` / `client.messageBatchResults` / `fx1 harness message-batch-results` (waits via `client.wait_message_batch` / `fx1 harness message-batch-wait`) |
| `POST /v1/messages/count_tokens` | Anthropic token counting — the request is the `/v1/messages` shape minus `max_tokens`/`stream` (system folds to a leading system message); answers `{input_tokens: N}` from the provider's own tokenize route (vLLM/SGLang `/tokenize`, Moonshot `estimate-token-count`) — a backend without that channel fails closed `501`, `tools`/`tool_choice` refuse `400` (a tokenize route sees only the message channel, so counting them would undercount). Never an estimate. `Fx1Harness.anthropic_count_tokens` / `HarnessClient.count_message_tokens` / `client.countMessageTokens` / `fx1 harness message-tokens` |
| `POST /v1/fine_tuning/jobs` | submit a gated fine-tuning job on a `purpose=fine-tune` corpus — synchronous validation, `Idempotency-Key` dedup; `Fx1Harness.create_finetune_job` (in-process, synchronous) / `HarnessClient.create_finetune_job` / `client.createFineTuneJob` / `fx1 harness ft-create` |
| `GET /v1/fine_tuning/jobs` / `GET /v1/fine_tuning/jobs/{id}` | list (`?limit≤100`, `?after=`) / poll one job record |
| `GET /v1/fine_tuning/jobs/{id}/events` | the job's event feed, oldest first (`?limit`, `?after=`) |
| `GET /v1/fine_tuning/jobs/{id}/checkpoints` | the model artifacts the job registered, oldest first (`?limit`, `?after=`); empty for a job that produced none, a deleted `ft:` name drops off |
| `POST /v1/fine_tuning/jobs/{id}/cancel` | cooperative cancel — queued at once, running at the next stage boundary; terminal `409 job_terminal` |
| `POST /v1/fine_tuning/jobs/{id}/pause` / `.../resume` | cooperative pause — queued parks pre-start, running parks at the next stage boundary; `paused` is non-terminal; `HarnessClient.pause_finetune_job`/`resume_finetune_job` / `fx1 harness ft-pause`/`ft-resume` |
| `GET /v1/chat/completions` | list stored `chat.completion` envelopes, oldest first (`?limit≤100`, `?after`/`?before`/`?order`, `?model=`, `?metadata[k]=v` subset filter) — OpenAI's `chat.completions.list`; `Fx1Harness.openai_chat_list` / `HarnessClient.list_chat_completions` / `client.listChatCompletions` / `fx1 harness chat-list` |
| `GET /v1/chat/completions/{id}` / `DELETE` | retrieval: fetch / drop a stored `chat.completion` envelope |
| `POST /v1/chat/completions/{id}` | update a stored completion — `metadata` replaces wholesale (≤16 pairs, keys ≤64 chars, values ≤512), choices/usage sealed; `Fx1Harness.openai_chat_update` / `HarnessClient.update_chat_completion` / `client.updateChatCompletion` / `fx1 harness chat-update` |
| `GET /v1/chat/completions/{id}/messages` | the request messages a stored completion ran on (`?limit`, `?after`, `?before`, `?order`) — OpenAI's `messages.list` |
| `GET /v1/responses/{id}` / `DELETE` | retrieval: fetch / drop a stored `response` object |
| `POST /v1/responses/{id}/cancel` | cancel a queued/in-progress `background:true` response (`status` → `cancelled`; 409 once terminal) — `Fx1Harness.openai_response_cancel` / `HarnessClient.cancel_response` / `client.cancelResponse` / `fx1 harness response-cancel` |
| `GET /v1/responses/{id}/input_items` | the `input` items a stored response ran on (`?limit`, `?after`, `?before`, `?order`) — OpenAI's `input_items.list` |
| `POST /v1/conversations` | mint a `conv_*` container (`items` seeds, `metadata` string pairs) — `Fx1Harness.openai_conversation_create` / `HarnessClient.conversation_create` / `client.conversationCreate` / `fx1 harness conv-create` |
| `GET` / `POST` / `DELETE` `/v1/conversations/{id}` | fetch the conv object / replace its `metadata` wholesale / drop the container and its items (member responses stay retrievable on their own ids) |
| `GET /v1/conversations/{id}/items` | the conv's accumulated items, paged by item id (`?limit`, `?after`, `?before`, `?order`) |
| `POST /v1/conversations/{id}/items` | append item dicts — returns the minted items as a `{object:"list"}` page (no `item_ids` alias — items mint per append) |
| `DELETE /v1/conversations/{id}/items/{item_id}` | drop one item; returns the conv object |
| `POST /v1/vector_stores` | mint a `vs_*` retrieval store (`name`, `file_ids` seed, `metadata`, `expires_after` anchor policy) — `Fx1Harness.vector_store_create` / `HarnessClient.vector_store_create` / `client.vectorStoreCreate` / `fx1 harness vs-create` |
| `GET` / `POST` / `DELETE` `/v1/vector_stores/{id}` | fetch / rename+remetadata+`expires_after` re-anchor / delete the store (delete detaches member files; the `file-*` records survive) |
| `GET /v1/vector_stores` | newest-first page (`?limit≤100`, `?after`, `?before`, `?order`) — `fx1 harness vs-list` |
| `POST /v1/vector_stores/{id}/files` | attach a `file-*` record (`attributes` string pairs ≤16, `chunking_strategy.static` overrides) → `vector_store.file`; a double-attach is `409 file_already_attached` — `fx1 harness vs-file-add` |
| `GET /v1/vector_stores/{id}/files` | member page (`?limit`, `?after`, `?before`, `?order`, `?filter` in `in_progress|completed|cancelled|failed`) — bad filters fail closed `400 invalid_filters` |
| `GET` / `DELETE` `/v1/vector_stores/{id}/files/{file_id}` | fetch / detach one member (`vector_store.file.deleted`) |
| `GET /v1/vector_stores/{id}/files/{file_id}/content` | the stored decoded text as a `vector_store.file_content.page` of per-chunk `{type:"text",text}` parts — `fx1 harness vs-file-content` |
| `POST /v1/vector_stores/{id}/search` | ranked hits without a response turn → `vector_store.search_results.page` (`query` string or list-joined, `max_num_results≤50`, `filters`, `ranking_options.score_threshold`; `rewrite_query`/non-`auto` rankers refused) — `Fx1Harness.vector_store_search` / `HarnessClient.vector_store_search` / `client.vectorStoreSearch` / `fx1 harness vs-search` |
| `POST /v1/vector_stores/{id}/file_batches` | attach up to 500 `file-*` ids in one call → `vector_store.files_batch` (`file_ids` 1..500, shared `attributes`/`chunking_strategy`; members attach synchronously — per-file refusals count `failed` with `last_error`, never abort) — `Fx1Harness.vector_store_file_batch_create` / `HarnessClient.vector_store_file_batch_create` / `client.vectorStoreFileBatchCreate` / `fx1 harness vs-batch-create` |
| `GET /v1/vector_stores/{id}/file_batches/{batch_id}` | the `vsfb_*` object — standing `status` + `file_counts` (`{in_progress,completed,cancelled,failed,total}`) |
| `POST /v1/vector_stores/{id}/file_batches/{batch_id}/cancel` | batches are terminal at create, so this is always `409 file_batch_terminal` — honest, never a fake in-flight window |
| `GET /v1/vector_stores/{id}/file_batches/{batch_id}/files` | the frozen per-file verdicts in request order (`?limit`, `?after`, `?before`, `?order`, `?filter` status word) |
| `POST /v1/evals` | create an `eval` spec container (`name`, `data_source_config.item_schema` = suite knobs — credentials never on the spec) → `201`; `Fx1Harness.eval_spec_create` / `HarnessClient.eval_spec_create` / `client.evalSpecCreate` / `fx1 harness eval-spec-create` |
| `GET /v1/evals` | newest-first spec page (`?limit≤100`, `?after=`); `Fx1Harness.eval_specs` / `HarnessClient.eval_specs` / `client.evalSpecs` / `fx1 harness eval-spec-list` |
| `GET` / `POST` / `DELETE` `/v1/evals/{id}` | fetch / rename+remetadata / tombstone a spec — delete journals and orphans the `/v1` run subresources (records stay on `/harness/evals/{id}`) |
| `POST /v1/evals/{id}/runs` | run the spec — `model` resolves to a backend (`ft:` names via the registry), `data_source.source` may override suite knobs per run, BYOK on the body; `201` + `Location` (bare id) + per-spec `Idempotency-Key` scope; `Fx1Harness.eval_run_create` / `HarnessClient.eval_run_create` / `client.evalRunCreate` / `fx1 harness eval-run` |
| `GET /v1/evals/{id}/runs` / `.../{run_id}` | list the spec's runs / poll one (`result_counts` once `completed`, `per_testing_criteria_results: []` honest-empty) |
| `POST /v1/evals/{id}/runs/{run_id}/cancel` | cooperative cancel of a queued run (`409` terminal); `HarnessClient.eval_run_cancel` / `client.evalRunCancel` / `fx1 harness eval-run-cancel` |
| `DELETE /v1/evals/{id}/runs/{run_id}` | drop a terminal run's `eval.run` object (`{id, object:"eval.run", deleted:true}`, `409` non-terminal); the `/harness/evals` record survives |
| `GET /v1/evals/{id}/runs/{run_id}/output_items` | per-task verdict rows verbatim (`?limit≤100`); `Fx1Harness.eval_run_items` / `HarnessClient.eval_run_output_items` / `client.evalRunOutputItems` / `fx1 harness eval-run-items` |
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
  `timeout_s`, and `receipt_hashes` / `X-Fx1-Receipt-Hashes` (CSV of
  sha256 digests — header citations run the same mounted-store check,
  the `fx1` extension's `receipt_hashes` wins, a malformed digest is a
  fail-closed 400).
- **Per-request timeout:** the `fx1` extension's `timeout_s` (body) >
  `X-Fx1-Timeout` header (seconds). The header is the wire twin for
  clients that can't edit the JSON payload — same deadline reaching the
  backend resolver on chat, responses, and embeddings. A malformed,
  non-finite, or out-of-`(0, 3600]` header is a fail-closed `400
  invalid_request`, never a silent default. Batch submitters inherit it:
  `X-Fx1-*` headers replay per line.
- **Streaming:** `stream: true` returns SSE `chat.completion.chunk`
  frames — a `role` delta, ~64-char content deltas on whitespace
  boundaries, a `finish_reason: "stop"` frame, an optional
  `choices: []` + `usage` chunk (`stream_options.include_usage`), then
  `data: [DONE]`. The gate runs before the first delta — no ungated
  bytes ever ship; a refusal is an OpenAI-shaped 502, not a truncated
  stream.
- **Decode contract:** `stop` (string or ≤4 sequences, ≤512 chars each)
  truncates the completion at the earliest match — enforced harness-side
  after the gate, so stub/local backends honor it too, while providers
  that support `stop` also get it verbatim; `n` (1–8) fans out into n
  independent gated calls — each `choices[i]` is a separate
  honesty-gate pass with its own completion-log record, usage is the
  sum of actual spend, and streams emit per-index frame groups;
  `presence_penalty`/`frequency_penalty` (±2) and `logit_bias`
  (token-id keys, ±100) are range-checked and forwarded verbatim;
  `reasoning_effort`, `service_tier`, `prompt_cache_key`,
  `prompt_cache_retention`, `verbosity`, and `user` pass through as
  provider hints (enums fail closed 422), and `user`/`metadata`
  (≤16 pairs)
  also stamp the call's audit-ledger record;
  `max_completion_tokens` is the OpenAI alias for `max_tokens` — a
  disagreeing pair is a 422, never a silent pick.
- **Tool calls:** `tools` (≤128 `{type: "function"}` specs),
  `tool_choice` (`none`/`auto`/`required` or a named-function dict),
  `parallel_tool_calls`, assistant `tool_calls` history, and
  `role: "tool"` outputs are first-class — they forward verbatim to a
  tool-capable link (any backend implementing `complete_with_tools`:
  `hosted_k3`, `byok`, `local_fx1`), the answer's `tool_calls` ride
  `choices[i].message` with `finish_reason: "tool_calls"`, and the
  streamed form emits a `delta.tool_calls` frame. The completion
  record's `output_sha256` binds text + the verbatim call list.
  A link without the channel answers 501 (`not_implemented`) — never
  a silently dropped spec. The honesty gate reads the assistant
  *text* only: `tool_calls[].function.arguments` are machine-bound
  JSON, not claims. Legacy `functions`/`function_call` stay refused —
  `tools` is the only function-calling grammar.
- **Logprobs channel:** `logprobs: true` + `top_logprobs` (0–20,
  requires `logprobs`) ride the same structured channel — they forward
  verbatim to a capable link and the provider's `choices[].logprobs`
  payload lands verbatim on the choice (null under provider silence).
  The streamed form emits one aggregated `delta.logprobs` frame per
  choice before the finish frame — provider token boundaries don't
  align with the harness's whitespace re-chunking, so the array ships
  whole rather than faking alignment. The completion record's
  `output_sha256` binds text + calls + the score payload when present.
  The native `/harness/complete` route takes the same fields;
  `/harness/complete/stream` answers 501 (use `stream: true` here).
- **Fail-closed surface:** `response_format` types
  outside `text`/`json_object`/`json_schema`,
  `modalities`, `audio`, `prediction`,
  `web_search_options`, `suffix`, `echo`, `best_of`, and
  `None`/non-text-part content are all rejected — nothing is silently
  dropped. `store` is honored, not refused: it governs the retrieval
  index (below). The native `/harness/complete` route takes the same
  `tools`/`logprobs` fields; `/harness/complete/stream` and
  `/harness/complete/batch` are text surfaces — tool context there is
  a refusal (501 / 422), not a dropped field.
- **Structured output:** `response_format` `json_object` and
  `json_schema` are honored by post-validation — the harness can't
  constrain-decode an arbitrary provider, so the gate's second pass
  validates the returned text instead (parsed JSON object for
  `json_object`; `jsonschema` validation against the declared schema —
  which is itself checked at request time — for `json_schema`). A
  non-conforming output is a provider-side 502
  (`format_violation`), never shipped and never pinned into an
  idempotency record; the verdict is identical on the SDK's
  `openai_chat`/`openai_chat_stream` (an `OpenAICompatError`).
- **Error envelope:** under `/v1`, every error — validation, auth
  (401/403), rate limit (429), body cap (413), over-capacity (503),
  honesty refusal (502) — returns OpenAI's
  `{error: {message, type, param, code}}` shape with OpenAI's type names
  (`invalid_request_error`, `authentication_error`,
  `rate_limit_error`, `server_error`, `service_unavailable`).
- **Model ops:** `GET /v1/models` lists `fx1` + the backend ids;
  `GET /v1/models/{id}` is `models.retrieve` — unknown ids fail closed
  404 (`model_not_found`), never a fabricated card.
- **Retry-safe:** `Idempotency-Key` dedupes retries — the same key +
  body replays the stored response byte-identically (JSON envelope or
  the SSE chunk sequence, `created` pinned) with
  `X-Fx1-Idempotent-Replay: true` and the original
  `X-Fx1-Completion-Id`; a key reused under a different body fails
  closed 409; keys are bounded (≤256 chars, over → 400) and only
  successful completions are pinned — a gate refusal re-executes on
  retry instead of replaying a cached error.
- **Auth:** when `FX1_API_KEY` is set, `/v1` also accepts the OpenAI
  `Authorization: Bearer` header in place of `X-API-Key`. Managed
  `fx1k_…` keys authenticate through both headers on `/v1` — stock
  OpenAI SDKs work with either credential unmodified.

Client-side: `HarnessClient.chat_completion` /
`chat_completion_stream` / `list_models` / `retrieve_model` in Python;
`HarnessApiClient.chatCompletion` / `chatCompletionStream` /
`listModels` / `retrieveModel` in TS — the chat calls accept
`idempotency_key` /
`idempotencyKey` and mark the call retryable for the built-in retry
policy. Any OpenAI SDK works directly — point it at the harness
`base_url` and use `model: "fx1"`. The claim is measured, not
asserted: `receipts/fx1_oai_sdk_audit.json`
(`fx1.serve.oai_sdk_audit.oai_sdk_audit_bench`) drives the stock
`openai` SDK — typed parsing, `async for` auto-pagination, SSE
streams, typed error classes (`BadRequestError`/`ConflictError`/
`NotFoundError`/`UnprocessableEntityError`) — through every `/v1`
resource group: models, chat (incl. `store`+`list`), responses (incl.
`background`+cancel+input_items), files+batches+fine-tuning,
embeddings, moderations, uploads, evals, conversations, vector
stores.

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

`POST /v1/responses` is the OpenAI Responses surface over the same
translation layer — same gated completion, same backend precedence,
same OpenAI error taxonomy:

- **Input:** a bare `input` string, shorthand message items
  (`{role, content: "…"}`), or full items with `input_text` /
  `output_text` parts; `instructions` prepends a system turn and
  `developer` roles map to system. Parts join by concatenation
  (per the spec), empty input is 422/400, `function_call` and
  `function_call_output` items carry the agent's tool history
  (they fold onto the shared chat channel — assistant
  `tool_calls` + `role:"tool"` messages), and item types outside
  that set (`computer_call`, `reasoning`, …) fail closed 400 —
  the harness never fabricates tool output.
- **Envelope:** the `response` object — `{id: "resp_…", status:
  "completed", output: [{type:"message", content: [{type:
  "output_text", …}]}], usage: {input_tokens, output_tokens,
  total_tokens} or null}` — plus request echoes (`temperature`,
  `top_p`, `max_output_tokens`, `metadata`, `instructions`,
  `service_tier`, `reasoning`, `text`, `prompt_cache_key`,
  `prompt_cache_retention`). A tool-call turn appends
  `{type: "function_call", call_id, name, arguments,
  status: "completed"}` items to `output` (a calls-only turn ships
  no message item).
- **Tools channel:** `tools` takes the flattened Responses spec
  (`{type: "function", name, description, parameters, strict}`),
  `tool_choice` is `none`/`auto`/`required` or
  `{type: "function", name}`, `parallel_tool_calls` sets the
  parallel flag — all three translate onto the same shared tool
  channel as `/v1/chat/completions` (specs nest under
  `function`, a dict choice folds to `{type, function:{name}}`),
  verbatim to tool-capable links; a link without the channel
  answers 501. Validation mirrors chat: >128 tools refuse 422,
  `tool_choice`/`parallel_tool_calls` without tools refuse 422,
  malformed `function_call`/`function_call_output` items refuse
  400.
- **Tool-call cap:** `max_tool_calls` bounds the function calls one
  response may carry — over the cap the emitted `output` truncates
  at the bound and the response lands `status: "incomplete"` with
  `incomplete_details: {"reason": "max_tool_calls"}` (OpenAI's own
  truncation semantics — never a silent drop). A calls-only turn
  capped at zero ships `output: []` (no phantom empty message); the
  stream's terminal frame is `response.incomplete`; stored objects,
  batch lines, and conversation appends keep the truncation. `ge=0`
  validated (negative is 422).
- **Logprobs channel:** `include: ["message.output_text.logprobs"]`
  is the only honored `include` member — it asks the provider for
  per-token scores, and `top_logprobs` (0–20) requires it. The
  provider's array lands on the message item's `output_text` part as
  `logprobs`, on both the JSON object and the stream's terminal
  `content_part.done` / `output_item.done` payloads and the embedded
  `response.completed` object. Other `include` members and a bare
  `logprobs` field (that's the chat surface's name) refuse 422.
- **Decode contract:** `max_output_tokens` maps to `max_tokens`;
  `reasoning.effort`, `service_tier`, `user`, `safety_identifier`,
  `metadata`, `prompt_cache_key`, and `prompt_cache_retention`
  forward like their chat counterparts (enums fail closed 422);
  `text.verbosity` (`low`/`medium`/`high` — anything else is 422)
  rides inside `text` and echoes verbatim. `text.format`
  is the same post-validated structured-output channel as
  `response_format` (`text` / `json_object` / `json_schema`, a
  violation is the same 502 `format_violation`).
- **Streaming:** `stream: true` emits the `response.*` event
  grammar (`response.created` → `response.in_progress` →
  `output_item.added` → `content_part.added` → `output_text.delta`
  ×N → `done`s → `response.completed`) with `event:` + `id:` +
  `data:` per frame — `id` is the frame index, no `[DONE]` sentinel
  (the completed event is terminal). Tool calls emit their own
  `function_call` item events at their own `output_index` —
  `output_item.added` → `function_call_arguments.delta` ×N →
  `function_call_arguments.done` → `output_item.done` — and the
  completed frame embeds the same response object the JSON path
  returns. `Last-Event-ID` resume works identically to the chat
  stream: the keyed response replays byte-identically, frames ≤
  the cursor dropped.
- **Fail-closed surface:** `truncation` and
  `include` members outside
  `message.output_text.logprobs`, plus every other unsupported
  field, refuse 422 at validation; nothing is silently dropped.
  `store` and `background` are honored, not refused.
- **Background calls:** `background: true` returns immediately
  with a `status="queued"` response object; the model call runs on
  the harness's job executor (same `inflight` capacity budget as
  synchronous work — submissions fail closed `503 draining` while
  the harness drains). Poll `GET /v1/responses/{id}` until
  `status` lands terminal (`completed` / `failed` / `cancelled` /
  `incomplete`); `POST /v1/responses/{id}/cancel` flips a live
  one to `cancelled` (409 `cancel_terminal` once terminal).
  `background` requires `store` (400 `background_requires_store`
  otherwise) and can't nest inside a batch line (the batch is
  already the async surface). `previous_response_id` chains
  validate at submit AND at run time — a parent deleted
  mid-flight still fails the work honestly. `stream:true` takes
  precedence over `background` — a stream is already the async
  surface, so the combination runs the normal stream.
- **Stateful chains:** `previous_response_id` chains the turn onto
  a stored `response` — the model runs on the parent's stored
  input items + its output + this request's `input`, and the
  child's `GET /v1/responses/{id}/input_items` returns the whole
  history. Chains nest to arbitrary depth. An unknown, deleted,
  or `store=false` parent fails closed
  `400 previous_response_not_found` before the model runs.
- **Named containers:** `conversation` (`conv_*` id or `{"id":
  "conv_*"}`) anchors the turn to a `/v1/conversations` container
  — its accumulated items are the context, and each completed
  turn appends its input + output items back. A conv is its own
  store: turns append even under `store: false`, and a deleted
  conv fails `400 conversation_not_found`. `conversation` and
  `previous_response_id` are mutually exclusive (422) and conv
  requests can't nest in a batch line — a shared container would
  race across lines.
- **Retry-safe:** `Idempotency-Key` shares the `/v1/chat/completions`
  dedup space — same key + body replays the stored envelope (or the
  pinned stream) byte-identically; a key reused under a different
  body is 409.

Client-side: `HarnessClient.responses_create` /
`responses_create_stream` / `cancel_response` in Python
(`Fx1Harness.openai_response` / `openai_response_stream` /
`openai_response_cancel` in-process — same `(envelope|events, cid)`
returns); `HarnessApiClient.responsesCreate` /
`responsesCreateStream` / `cancelResponse` in TS; `fx1 harness
respond [--background]` / `response-get` / `response-cancel` on the
CLI.

### Embeddings (`/v1/embeddings`)

`POST /v1/embeddings` is the OpenAI embeddings surface over the same
link chain — same backend precedence, same `fx1.*` extension block and
`X-Fx1-*` headers, same error taxonomy — for the retrieval/eval lanes
that need vectors:

- **Input:** a string, `list[str]` (N inputs), a token array
  `list[int]`, or `list[list[int]]` (N token-array inputs) — the same
  shapes the OpenAI surface accepts. Empty/blank strings, empty lists,
  mixed-type lists, and >2048 items refuse 422; `encoding_format` is
  `float`/`base64`, `dimensions` ≥ 1, `user` ≤ 512 chars.
- **Forward:** `model`, `input`, `encoding_format`, `dimensions`,
  `user` reach the provider **verbatim** — embedding models name
  themselves, so the request `model` goes on the wire, not the link's
  chat pin. The provider's `data[]`, `model`, and `usage` echo back
  untouched (`usage: null` under provider silence).
- **Channel:** capability is the `embeddings` method on the backend —
  hosted_k3 and BYOK have it, `local_fx1` deliberately does not (fx-1
  is a decoder; point BYOK at an embedding engine). A link without the
  channel answers **501 `not_supported`** — never fabricated vectors.
  Provider faults are the same 502/503 split as chat
  (`BackendNotConfiguredError` → 503 and the chain advances;
  `RuntimeError` → 502 on the last link).
- **Evidence:** vectors aren't claims — no honesty gate — but the call
  lands in the completion log and metrics exactly like a completion:
  `prompt_sha256` binds `{model, input}`, `output_sha256` binds the
  verbatim `data[]`, `X-Fx1-Completion-Id` links the record, and the
  entry is fetchable at `/harness/completions/{id}`.
- **Batch:** `endpoint: "/v1/embeddings"` is a first-class batch
  endpoint — same JSONL in/out channel as chat/responses.

Client-side: `HarnessClient.embeddings_create` in Python
(`Fx1Harness.openai_embeddings` in-process — same `(envelope, cid)`
return); `HarnessApiClient.embeddingsCreate` in TS.

### Fine-tuning (`/v1/fine_tuning/jobs`)

The OpenAI fine-tuning surface over the gated training pipeline —
upload a chat-format corpus, submit a job, poll events, collect
artifact files.

- **Corpus:** `POST /v1/files` with `purpose=fine-tune` accepts
  chat JSONL (`{"messages": [...]}` per line); validation is
  synchronous at submit — a malformed corpus or a file with the
  wrong purpose is a `400 invalid_training_file`, never a queued
  job. `model` is restricted to the trainable set (`fx1`,
  `local_fx1`) — `byok`/`hosted_k3` is a `400
  model_not_trainable`.
- **Lifecycle:** `validating_files` → `queued` → `running` →
  `succeeded|failed|cancelled`. The job holds one inflight slot on
  the jobs executor; `429`/`503` carry `Retry-After`. Events land
  on `GET .../events` (validated → started → runner emissions →
  terminal).
- **Artifacts:** each `FTJobOutcome.artifacts` entry is
  re-registered as a `purpose=fine-tune-result` file and listed in
  `result_files` — fetch bytes via `GET /v1/files/{id}/content`.
- **Runner contract:** `FTJobRunner(spec, *, emit, should_cancel, pause_gate)`
  — `pause_gate` is optional on injected runners (the worker
  introspects); `pause_gate()` blocks while the job is paused and
  returns True when a cancel landed while parked — call it between
  stages and return early on True to unwind to `cancelled`.
  — the default runner (`default_ft_runner`) executes the gated
  pipeline in-process (quality gate → baseline eval → train →
  candidate eval) and fails honestly (`status=failed`,
  `error.code=job_failed`) when the trainer can't run on this
  host. `trained_tokens` stays `null` — no tokenizer exists, and
  the harness never fabricates counts.
- **Model registry:** a `succeeded` job whose outcome carries a
  `checkpoint` registers its `ft:{model}:{suffix}:{job}` name into
  the model inventory — `GET /v1/models` lists it and
  `GET /v1/models/{id}` retrieves its card. Completions, responses,
  and embeddings naming the `ft:` model resolve to the `local_fx1`
  lane pinned at the producing job's checkpoint; an explicit backend
  pin (the `fx1` extension's backend field, `X-Fx1-Backend`, or BYOK
  headers) still overrides, and an
  `ft:` name with no registered job is a `404 model_not_found` —
  never a silent default link. Evicting the job record drops the
  card (registration is provenance-bound, not permanent). The SDK
  twin shares the same store, so `openai_models()`/`openai_chat`
  behave identically in-process.
- **Cancel:** `POST .../cancel` — queued jobs cancel at once;
  running jobs stop cooperatively when the runner's
  `should_cancel()` reports the flag (between stages).
- **Pause/resume:** `POST .../pause` marks the job `paused`
  (non-terminal): a queued job's worker parks at a pre-start gate —
  no `job started` event until resumed; a running job's worker parks
  inside `pause_gate()` at the next stage boundary — the hook is
  opt-in on the runner contract (`FTJobRunner(..., pause_gate)`),
  so a gate-free runner completes normally through a running-pause
  and only queued pauses still hold. `POST .../resume` restores the
  captured status (`queued` or `running`) and releases the gate.
  Pausing a paused job replays its record — idempotent; pause/resume
  on a terminal job is `409 job_terminal`, resume on a non-paused
  job is `409 job_not_paused`. A paused job still honors cancel
  (terminal write lands at once, the parked worker exits without a
  duplicate event) and drain (a restart replays `paused` → `failed`
  like every non-terminal state).
- **Retry-safe:** `Idempotency-Key` dedups submission against the
  body fingerprint — a replay returns the same job record, a key
  reused under a different body is `409 idempotency_conflict`.
- **SDK twin:** `Fx1Harness.create_finetune_job(training_jsonl=...)`
  runs the same runner contract in-process and synchronously — it
  returns the terminal record directly (no queue), and stores the
  record in the same `FTJobStore` for `finetune_job`/events reads.
  Because the create is synchronous there is no pause window — the
  SDK carries no `pause_finetune_job`; the verbs are wire-only
  (`HarnessClient`/`client.pauseFineTuneJob`/`ft-pause`).

### Batches + files (`/v1/batches`, `/v1/files`)

The OpenAI async-batch surface over the same gated pipeline — upload
a request JSONL once, submit it as one tracked batch, collect an
output JSONL of per-line results.

- **Files:** `POST /v1/files` takes `multipart/form-data` with a
  `purpose` field (`"batch"` or `"fine-tune"` — fail-closed) and a `.jsonl`
  `file` part; the response is the OpenAI `file` object. Files live
  in a bounded store (`FX1_API_FILE_MAX` entries, default 128;
  `FX1_API_FILE_BYTES` per file, default 8 MiB — LRU eviction like
  every store on this surface). `GET /v1/files` lists newest-first,
  `GET /v1/files/{id}` retrieves the card, `GET
  /v1/files/{id}/content` returns the raw bytes, `DELETE` evicts.
- **Batches:** `POST /v1/batches` takes `{input_file_id, endpoint,
  completion_window, metadata}` — `endpoint` is one of
  `/v1/chat/completions`, `/v1/responses`, or `/v1/embeddings`,
  `completion_window` is
  `"24h"` (the only declared window; `expires_at` is set +24h). Line
  shape is validated at submit — a batch never starts on a corrupt
  file: bad JSON, missing/oversized `custom_id`, non-`POST` method,
  or a `url` that doesn't match `endpoint` is a submit-time 400 with
  the line number. Per-line cap `FX1_API_BATCH_LINES` (default 1024).
- **Execution:** the batch holds ONE inflight slot on the jobs
  executor (`FX1_API_JOB_MAX` bounds the store, `FX1_API_BATCH_MAX`
  the batch index). Each line runs through the endpoint's own
  request model + the same extracted completion core the live route
  uses — parity is literal, not a second pipeline. The submitter's
  `X-Fx1-*` routing headers (`X-Fx1-Backend`, `X-Fx1-Byok-*`, …)
  apply to every line — a BYOK batch stays on the caller's endpoint
  and never reads ambient env. Status moves `validating` →
  `in_progress` → `finalizing` → `completed`; `GET /v1/batches/{id}`
  carries `request_counts.{total,completed,failed}` live.
- **Output:** on terminal the output lines land in a new file
  (`output_file_id`) — `GET /v1/files/{id}/content` returns one
  OpenAI batch-result line per input: `{id, custom_id,
  response:{status_code, request_id, body}, error}`. A line whose
  request the live route would refuse lands as a `status_code`
  line with the OpenAI error body — the batch itself still
  completes; only a worker crash fails the batch (`errors.data`
  carries the fault, `error_file_id` the partial output).
  `stream: true` inside a line is a per-line 400 — batch results
  are never streams.
- **Cancel:** `POST /v1/batches/{id}/cancel` is cooperative — the
  worker checks between lines, lands `cancelled`, and writes
  whatever output lines exist. Terminal batches 409.
- **Retry-safe:** `Idempotency-Key` shares the `/v1` dedup space —
  a resubmitted create replays the submit envelope; a key reused
  under a different body is 409.

Client-side: `HarnessClient.upload_file` / `files` / `file` /
`file_content` / `delete_file` / `create_batch` / `batch` /
`batches` / `cancel_batch` / `wait_batch` in Python;
`HarnessApiClient.uploadFile` / `files` / `file` / `fileContent` /
`deleteFile` / `createBatch` / `batch` / `batches` / `cancelBatch`
/ `waitBatch` in TS. In-process, `Fx1Harness.openai_batch(lines,
endpoint=…)` runs the same lines through `openai_chat` /
`openai_response` synchronously and returns `(batch,
output_lines)` — no upload/poll machinery needed weights-direct.
The CLI drives the whole lifecycle over `--remote`: `fx1 harness
files` / `file-upload` / `file-content` / `file-delete` for the
file store, `batch-submit` (upload + submit + poll to terminal;
`--no-wait`, `--metadata`, `--idem-key`, `--callback-url`,
`--callback-secret`) / `batches` / `batch-status` /
`batch-cancel` / `batch-output` (fetch `output_file_id` bytes to
`--out` or stdout) for the batch lifecycle, and `ft-create`
/`ft-jobs`/`ft-status`/`ft-events`/`ft-cancel` for fine-tuning —
`ft-create` also accepts the webhook flags on both the remote and
in-process SDK paths. `fx1 harness batch-run` is the weights-direct
twin: no server — a local JSONL runs synchronously through the same
per-endpoint request models and gate, `--backend`/`--checkpoint-dir`/
`--byok-*`/`--fallback` map onto the wire's `X-Fx1-*` headers, and
`--out` writes the OpenAI batch-result lines. `fx1 harness models` /
`model <id>` expose the `/v1/models` inventory both ways — remote over
the wire, or in-process where the `ft:` registry lists your own
fine-tunes. `fx1 harness respond` (`/v1/responses` — JSON items arg,
`--instructions`/`--format`/`--tools`/`--tool-choice`/
`--max-tool-calls`/`--previous-response-id`/`--conversation`/
`--background`/`--verbosity`/`--prompt-cache-key`/
`--prompt-cache-retention`; `--stream` prints
the Responses event stream's delta frames — token text and tool-call
arguments — on either leg instead of the one-shot JSON object), `embed`
(`/v1/embeddings` — repeatable input, `--encoding`/`--dimensions`), and
`moderate` (`/v1/moderations` — the honesty gate as an OpenAI verdict,
no backend needed) each run both legs: `--remote` over the wire or
in-process through the SDK twin. `chat-get`/`chat-update`/`chat-delete`/
`response-get`/`response-delete` cover the stored-object
`GET`/`POST`/`DELETE` routes (missing ids exit 2 — never a fabricated
envelope; `chat-update --metadata` is a JSON object of string pairs), `fx1 harness score <text...>` scores through the
reward contract with no model spend, `fx1 harness commands`
lists the registry (`--role` filters; a bogus role exits 2 like
the wire's 422), and `fx1 harness verify <dir>` posts the whole
directory through POST /receipts/verify/batch in one call.

### Retrieval (`store` + `GET`/`DELETE`)

The `store` flag is honored on both `/v1` surfaces: `store: false`
(the OpenAI default is `true`) keeps the call's envelope out of the
retrieval index; every gated call still lands in the completion log
and its sealed receipt — the flag governs *retrieval*, never
evidence. The index is a bounded LRU (`FX1_API_STORE_MAX`, default
256) holding whole envelopes — `chat.completion` for the chat
surface, `response` for Responses.

- `GET /v1/chat/completions/{chatcmpl-…}` returns the stored
  envelope verbatim; `GET /v1/responses/{resp_…}` the stored
  response object. A miss (evicted, deleted, or sent with
  `store:false`) is an OpenAI-shaped 404 `not_found`; an id from
  the wrong surface is the same 404, not a cross-read.
- `DELETE` drops the envelope and returns
  `{id, object: "<type>.deleted", deleted: true}`; deleting a miss
  is 404.
- Stored envelopes index on **completion**: sync calls, streams
  (`store:false` on a stream keeps the assembled envelope out),
  n-fan-out (one envelope per request), batch lines, and
  idempotent replays all land identically — a replayed call
  re-pins its envelope at the head of the LRU.
- The index is a fetch cache for callers, not the audit trail —
  the completion log (hash-only) and sealed receipts still carry
  every call regardless of `store`.

`/v1/conversations` is the named-container twin of the chain
surface: `POST` mints a `conv_*` object (optional seed `items` +
`metadata`), `GET`/`POST`/`DELETE` read, re-metadata, and drop
it, and `/items` lists, appends, and deletes the accumulated
item stream a `conversation`-anchored response draws its
context from.

Client-side: `HarnessClient.retrieve_chat_completion` /
`delete_chat_completion` / `retrieve_response` / `delete_response`
in Python (`KeyError` on 404), `Fx1Harness.openai_chat_get` /
`openai_chat_delete` / `openai_response_get` /
`openai_response_delete` in-process, `retrieveChatCompletion` /
`deleteChatCompletion` / `retrieveResponse` / `deleteResponse` in
TS.

### Vector stores + `file_search` (`/v1/vector_stores`)

`/v1/vector_stores` is the server-side RAG surface: stores are
journaled under `--state-dir` (`vector_stores.jsonl`, replayed on
restart) and bounded (`FX1_API_STORE_MAX` bounds the store count —
LRU-evicting the oldest at the cap; files per store, text bytes,
chunks, and `vs_*` ids are fixed constants — oversized attaches
fail closed). Attached `file-*` records are chunked (word windows
with overlap) and indexed with a hashed bag-of-words + per-store
idf — cosine ranking, no embedding service required.

The `file_search` tool on `POST /v1/responses` searches them
in-band:

- `tools: [{"type": "file_search", "vector_store_ids": ["vs_…"]}]`
  runs the user's latest turn as the query over the listed stores
  (≤8 ids, `max_num_results ≤ 50`,
  `ranking_options.score_threshold` ∈ [0,1] — violations are
  fail-closed `422`/`400` before the model runs).
- The call lands in `output` as a `file_search_call` item
  *before* the assistant `message`; `include:
  ["file_search_call.results"]` gates whether `results` carries
  the ranked hits (`{file_id, filename, score, text}`) — absent
  the flag, `results` is `null`.
- The top hits are prepended to the model's context as one
  `developer`-item (`[file_search results] …`, capped at a fixed
  injection budget) so the gated pipeline sees the retrieval.
- Streaming emits `response.output_item.added` →
  `response.file_search_call.in_progress` → `.searching` →
  `.completed` → `response.output_item.done` frames ahead of the
  message deltas, and the `file_search_call` survives on the
  stored/replayed envelope. Refed input items fold back into a
  single `[prior file_search results]` system message.
- `tool_choice: {"type": "file_search"}` forces the retrieval
  call; any other `tool_choice`/`parallel_tool_calls` without a
  `function` tool on the request is dropped rather than
  mistranslated.

`POST /v1/vector_stores/{id}/search` queries one store directly —
the same ranked hits the tool turn would inject, returned as a
`vector_store.search_results.page` (`{file_id, filename, score,
attributes, content:[{type:text}]}` entries, `has_more:false`) —
for callers that want retrieval without spending a response turn.
`query` accepts a string or a list (joined on spaces);
`max_num_results` (≤50), `filters` (the OpenAI comparison/
condition schema — `{type: "eq", key, value}` leaves and
`and`/`or` trees ≤4 deep), and
`ranking_options.score_threshold` all behave exactly as on the
tool spec. `rewrite_query` and any ranker other than `"auto"`
are fail-closed `422`s — no silent query mutation.

`POST /v1/vector_stores/{id}/file_batches` is the bulk-attach
surface: `file_ids` (1..500, OpenAI's cap) attach one at a time
through the same `attach` path as `…/files`, so a missing file,
a double-attach, an oversized blob, or a full store counts
`failed` with that refusal as the row's `last_error` — the
batch never aborts on a bad member and never half-attaches.
The batch object (`object: "vector_store.files_batch"`,
`vsfb_*` id) reports `file_counts` and is terminal at return:
`completed` when ≥1 member attached, `failed` when none did.
`POST …/file_batches/{id}/cancel` therefore always answers
`409 file_batch_terminal` — there is no fake in-flight window.
`GET …/file_batches/{id}/files` pages the frozen per-file
verdicts (`filter` accepts an OpenAI status word); the rows
are the batch's record — a later `DELETE` of a member file
doesn't rewrite history. Batches journal under `--state-dir`
like the stores themselves and disappear with their store.

Stores support OpenAI's standing-expiry policy:
`expires_after: {"anchor": "last_active_at", "days": 1..365}`
on create/update (any other anchor or bound fails closed
`400 invalid_expires_after`). `last_active_at` bumps on every
attach, batch create, and search, and `expires_at` re-anchors
from it; once `now >= expires_at` the store reports
`status: "expired"` and refuses *writes* — file attaches,
batch creates, and search return `410 vector_store_expired` —
while reads (`GET` store/files/content, `DELETE`) still
resolve. An `expires_after` update re-anchors from the
recorded `last_active_at`, which can revive an expired store
honestly (no undelete semantics — `status` recomputes).
Activity bumps journal as `vs_touch` lines so replay
preserves the expiry window.

Identical contract in-process: `Fx1Harness.openai_file_create`
(content bytes → `file-*`) + `vector_store_*` twin methods drive
the same store, and `openai_response` emits the same
`file_search_call` grammar. `HarnessClient.vector_store_*` +
`fx1 harness` `vs-*` cover the wire leg; TS exposes
`vectorStoreCreate`/`…List`/`…Files`/`…FileContent`. Unknown
stores fail closed `vector_store_not_found` (404 on the wire,
`VectorStoreError`/`OpenAICompatError` in-process).

## Auth & safety

- `fx1 harness serve` binds **loopback-only** unless `FX1_API_KEY` is
  set; with a key, every route except `/health` requires
  `X-API-Key` (constant-time compare). An empty key equals unset — never
  a bypass.
- **Managed API keys** (`/harness/keys`) ride beside the env key: mint
  `fx1k_…` workload keys that authenticate on every gated route like
  `X-API-Key` (and `Authorization: Bearer` on `/v1`). A non-empty key
  store turns remote auth on even with no `FX1_API_KEY` — provisioning
  on loopback is the opt-in. Only the bootstrap credential (env key)
  or loopback-dev may mint/list/revoke — `403 admin_required`
  otherwise — and `admin:true` mints a key that can manage keys
  itself, so a no-env-key deployment keeps a control plane. Revocation
  is a tombstone (`enabled:false`, fail-closed); records persist under
  `--state-dir` (journaled to `keys.jsonl`, replayed on restart —
  `uses`/`last_used_at`/`tokens_used` are live counters, deliberately
  not journaled).
  Declared policy travels with the record: `rpm` bounds the key to a
  fixed 60 s request window — the over-limit refusal is `429
  rate_limited` with an honest `Retry-After`, and a refused request
  never counts as a use — `ttl_s` stamps an `expires_at` past
  which the key authenticates as dead (same 401 shape as revoked — no
  oracle for which keys exist), and `max_requests`/`max_tokens`
  declare hard budgets — the exhausted-key refusal is `429
  quota_exceeded` **without** `Retry-After` (a budget never clears
  inside a call, so the SDKs/TS client do not retry it; it maps to a
  terminal transport error, not the auth class). `max_tokens` charges
  post-response off provider-reported usage — the crossing call
  completes; the budget gates the next. Every completion record
  attributes its caller's `key_id`
  fingerprint, so `GET /harness/usage?key_id=` reads per-key spend
  without ever exposing secrets.
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
- Drop-in SDK headers: `Openai-Processing-Ms` (integer wall-clock ms)
  and `openai-version` ride every response on the `/v1` grammar —
  `openai-version` is our own wire contract (`API_VERSION`, currently
  `"1"`), not a dated OpenAI deployment spec, since the surface is a
  contract superset rather than a snapshot of one upstream version.
  The Anthropic dialect (`/v1/messages*`, or any `/v1/*` request under
  an `anthropic-version` header) answers with Anthropic's names for
  the same surfaces: `request-id` (the same id `X-Request-ID` stamps,
  echoed from an inbound `X-Request-ID` or minted, on success, error,
  and SSE-open alike) and `x-should-retry` — `true` on the transient
  statuses the stock SDK retries (408/429/500/502/503/504/529),
  `false` on the ones its defaults would get wrong here (409
  idempotency conflict, 501 unimplemented), omitted everywhere else.
  A managed key minted with `rpm` reports its standing window on the
  Anthropic surface too — `anthropic-ratelimit-requests-limit` /
  `-remaining` / `-reset` (an RFC 3339 instant, Anthropic's
  convention) beside the `X-RateLimit-*-Requests` family; env /
  loopback / unwindowed keys emit neither (no false scarcity), and
  there is no token-window family because no token window is metered.
  We deliberately do **not** emit `openai-organization`, `cf-*`, or
  other org/edge provenance headers — no organization layer or CDN
  fronts this process, so minting them would fabricate provenance.
  `HarnessClient.last_response_headers`,
  `Fx1Harness.last_response_headers` (stamped per gated call), and the
  TS client's `lastResponseHeaders` expose the last response's
  lowercased header map on their respective legs — `{}`/`null` before
  the first call or after a transport fault.
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
client.last_response_headers  # the last response's full header map
```

`fx1 harness compat --remote URL` prints the report and exits 1 on
mismatch — deploy pipelines gate on it before routing traffic. Both
`compat` and `version --remote` fold the just-answered call's
`x-request-id` into their JSON (`request_id`) when the peer stamps
one — the trace id a bug report should quote.

`fx1 harness capabilities --remote URL` (`client.capabilities()` /
`HarnessApiClient.capabilities()`) returns the server's declared feature
set (`features`: idempotency, SSE, webhooks, batch, jobs, drain,
streaming), its effective limits (`limits`: batch caps, store bounds,
`rate_limit_rps`, `sse_keepalive_s`, body/job-result byte caps), which
backends are configured (`backends`, booleans only), and the registered
command roles (`roles`). Clients self-configure from this instead of
hardcoding server internals.

`fx1 harness selftest` is the deploy gate: zero-config golden-path smoke
of the whole contract. With no flags it boots a stub OpenAI engine and
the production app on loopback and walks auth, commands, a BYOK
completion, SSE reassembly, idempotent replay, the async job lifecycle,
sealed-receipt verification, drain, and in-process parity with
`Fx1Harness` — 17 checks, exit 0 only when all pass. `--state-dir DIR`
adds a real process restart proving job-record recovery. `--remote URL`
flips to read-only probes against a live deployment (no model spend):
health, version negotiation, commands, advisory surfaces, and the auth
gate when `--api-key` is given.

`fx1 harness bench` is the perf gate: it times `--n` gated `complete`
calls at `--concurrency` workers (after `--warmup` unmeasured requests)
and prints the latency card (p50/p90/p95/p99/max/mean), throughput,
token rates, and an error histogram by exception class. The prompt is
digested (`prompt_sha256`), never embedded. Both legs work: default is
the in-process SDK, `--remote URL` benches a live deployment;
`--backend`/`--byok-*`/`--seed`/`--max-tokens` forward per request.
`--receipt` prints the sealed `fx1_bench_result.v1` doc (verify with
`dipcatcher verify-receipt` / `POST /receipts/verify`). Exits 0 only
when every measured request succeeded, 1 on any error — a deploy gate
beside `selftest`.

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
- **Durability** — with `--state-dir` (`FX1_API_STATE_DIR`) every state
  transition and cancel/evict across the async surface appends to a
  hash-chained JSONL journal (fsync'd per append): `jobs.jsonl`,
  `evals.jsonl`, `batches.jsonl`, `abatches.jsonl`, `ft_jobs.jsonl`,
  `files.jsonl` + `files/<id>.bin` blob files, and
  `idem_{runs,complete,complete_batch,openai}.jsonl`. On boot each chain is verified line-by-line — a torn
  tail or edited line truncates at the first bad record — and the
  stores are rebuilt: terminal records return as-was, anything still
  `queued`/`running`/`validating`/`in_progress`/`finalizing`/
  `cancelling` at the crash recovers as `failed` with a
  restart-explaining `error` (payloads are not journaled, so nothing is
  silently re-run) — an Anthropic `message_batch` caught mid-flight ends
  with its unfinished items as `errored` rows carrying the restart note —
  and `Idempotency-Key` mappings survive — the idem
  stores journal the recorded response itself, so a retried submission
  replays the recorded answer (`replayed: true`) after a restart
  instead of re-running. Upload payloads live in content blobs, not
  the journal; a record whose blob is missing drops with a
  `recover_warnings` entry, and deletes/evictions tombstone the blob.
  The ft journal also restores each job's event feed and the `ft:`
  model registry — a model card never outlives its producing job
  (eviction drops the card). `callback_secret` never reaches disk, so
  a recovered record with a `callback_url` keeps it for audit but
  cannot deliver post-restart. Boot compacts each journal to live
  records. Unset = the same in-memory stores as before. The in-process
  SDK binds the same journals: `Fx1Harness(state_dir=...)` (or the
  `FX1_SDK_STATE_DIR` env var) journals evals and fine-tune jobs with
  identical restart semantics — a mid-eval crash recovers as `failed`,
  terminal records return as-was.

The same contract applies on the OpenAI-compatible async surfaces:
`POST /v1/fine_tuning/jobs` and `POST /v1/batches` accept
`callback_url`/`callback_secret` and POST the terminal record (job or
batch object) once — same HMAC headers, same 3-attempt/4xx-definitive
delivery, same `callback_status`/`callback_attempts`/`callback_error`
fields on the record. A 4xx is a definitive rejection and never retried;
transient faults retry up to 3 times with capped backoff. In-process,
`Fx1Harness.create_finetune_job`, `Fx1Harness.openai_batch`, and
`Fx1Harness.anthropic_batch` take the same kwargs and deliver over real
HTTP before returning. `POST /v1/messages/batches` accepts the same pair —
its terminal webhook posts the `message_batch` envelope and the verdict
fields ride on it under the same names.

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
| `--store-max` | `FX1_API_STORE_MAX` | 256 | /v1 retrieval-index capacity (LRU evict) |
| `--sse-keepalive-s` | `FX1_API_SSE_KEEPALIVE_S` | 15 | `: keepalive` comment cadence; 0 disables |
| `--rate-limit-rps` | `FX1_API_RATE_LIMIT_RPS` | 0 (off) | per-client token bucket → 429 + `Retry-After`; every response also carries `X-RateLimit-Limit`/`Remaining`/`Reset` while the limiter is on. Public paths (`/health`) are exempt — LB probes never consume the client budget |
| `--gzip-min-bytes` | `FX1_API_GZIP_MIN_BYTES` | 1024 | gzip only when the client advertises it; 0 disables |
| `--cors-origins` | `FX1_API_CORS_ORIGINS` | (off) | comma-separated browser origins for CORS; each must be a scheme+host URL, `*` and non-http(s) refused; preflights bypass the API-key gate (they carry no credentials), every preflight reflects the `expose` list of stamped headers |
| `--breaker-threshold` | `FX1_API_BREAKER_THRESHOLD` | 5 | consecutive call faults that open a backend's circuit; 0 disables. While open, calls fast-fail `503 backend_unavailable` + `Retry-After` without burning an inflight slot; a single half-open probe is admitted after cooldown and closes the circuit on success. Resolution faults that surface as 503 count; client errors (404/422), capability gaps (501), and honesty-gate refusals never do |
| `--receipts-dir` | `FX1_API_RECEIPTS_DIR` | `receipts` | sealed-receipt store backing `GET /receipts*` — `503 receipts_unavailable` when absent |
| `--state-dir` | `FX1_API_STATE_DIR` | (off) | durable dir for the state journals (jobs/evals/batches/ft-jobs/files/idempotency) — crash/restart recovers records + keys; unset = in-memory |
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
