# fx-1 harness — BYOK runbook

BYOK ("bring your own key") runs every gated surface — completions,
streams, evals, batches, the OpenAI/Anthropic drop-ins — against *your*
OpenAI-compatible endpoint while keeping the harness's honesty gates,
metering, idempotency, and sealed receipts. This page is the request
shapes on each leg, the credential-handling contract, and the provider
quirks table. Wire semantics live in `docs/FX1_HARNESS_API.md`; the ops
runbook in `docs/FX1_DEPLOY.md`; the four client legs in
`docs/FX1_CLIENTS.md`.

## What BYOK binds to

`backend="byok"` resolves to `OpenAICompatBackend`: a chat-completions
client for **any** OpenAI-compatible `/chat/completions` endpoint (vLLM,
SGLang, OpenRouter, Azure OpenAI, Ollama's compat shim, …). The backend
needs three values:

| Value | Env (server-side default) | Per-request |
|---|---|---|
| Base URL | `FX1_BYOK_BASE_URL` — joined to `<base>/chat/completions` | `byok.base_url` |
| API key | `FX1_BYOK_API_KEY` | `byok.api_key` |
| Model | `FX1_BYOK_MODEL` | `byok.model` |

Resolution order per request: explicit `byok` in the body (or
`X-Fx1-Byok-*` headers on `/v1/*`) wins; absent that, the env triple
supplies the link. A partially configured link fails closed at resolve —
`BackendNotConfiguredError` → `503 backend_unavailable` — never a guessed
endpoint.

## Request shape on each leg

**`/harness/*` body field** — `byok` is a validated `ByokOverride` on
`CompleteRequest`, `EvalSubmitRequest`, and the batch/eval/route models:

```json
POST /harness/complete
{
  "backend": "byok",
  "messages": [{"role": "user", "content": "…"}],
  "byok": {"base_url": "https://provider.example/v1",
           "api_key": "sk-…",
           "model": "gpt-fake"}
}
```

Rules the wire enforces (all 422s, verified in
`_resolve_request_backend`):

- `byok` on a non-`byok` backend → `422 "a byok override applies only to
  backend='byok'"`.
- `byok` present while the server ran with `FX1_API_BYOK_OVERRIDE=0` →
  `422 byok_override_disabled`. Deployers use this to pin workloads to
  server-configured lanes only.
- `base_url` must be a real http(s) URL (validated on the model).

**Python SDK / remote client** — `byok` is the same dict on `complete`,
`stream_complete`, `complete_many`, `run_eval` / `submit_eval`:

```python
fx.complete(messages, backend="byok",
            byok={"base_url": URL, "api_key": KEY, "model": MODEL})
client.submit_eval("tooluse", backend="byok", seed=0,
                   byok={"base_url": URL, "api_key": KEY, "model": MODEL},
                   judge_backend="byok", judge_byok={"base_url": J_URL, ...})
```

**CLI** — the three flags pair per-command:

```bash
fx1 harness complete "…" --backend byok \
  --byok-base-url $URL --byok-api-key $KEY --byok-model $MODEL
fx1 harness eval tooluse --backend byok --remote $H --api-key $K \
  --byok-base-url $URL --byok-api-key $KEY --byok-model $MODEL
fx1 harness bench --remote $H --n 32 --byok-base-url $URL --byok-model $MODEL  # + --byok-api-key
```

**`/v1/*` drop-in surfaces** — OpenAI SDKs can't send a `byok` field, so
two equivalent channels exist:

```bash
# (a) X-Fx1-* headers — backend defaults to byok when the headers are present
curl -X POST $H/v1/chat/completions \
  -H "X-API-Key: $K" -H "Content-Type: application/json" \
  -H "X-Fx1-Byok-Base-Url: https://provider.example/v1" \
  -H "X-Fx1-Byok-Api-Key: sk-…" \
  -H "X-Fx1-Byok-Model: gpt-fake" \
  -d '{"model": "gpt-fake", "messages": [{"role":"user","content":"…"}]}'

# (b) the fx1-extension object — body-level equivalent
curl -X POST $H/v1/chat/completions -H "X-API-Key: $K" \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-fake",
       "fx1": {"backend": "byok",
               "byok": {"base_url": "…", "api_key": "…", "model": "…"}},
       "messages": [{"role":"user","content":"…"}]}'
```

Header rules: `X-Fx1-Byok-Base-Url` **requires** `X-Fx1-Byok-Api-Key`
(missing key → 400); `X-Fx1-Byok-Model` defaults to the request `model`
when that isn't a backend name — so a stock `model: "gpt-4o"` body plus
the three headers needs no other edit. Backend selection order on `/v1/*`:
fx1.backend > `X-Fx1-Backend` > a `model` naming a backend
(`hosted_k3`/`local_fx1`/`byok`) > byok-headers-present → `byok` >
`hosted_k3`.

**TypeScript** — same shapes, camelCased where the client wraps:

```ts
await api.complete({
  backend: "byok",
  messages: [{ role: "user", content: "…" }],
  byok: { base_url: U, api_key: K, model: M },
});
await api.chatCompletion(
  { model: M, messages: [...] },
  { "X-Fx1-Byok-Base-Url": U, "X-Fx1-Byok-Api-Key": K, "X-Fx1-Byok-Model": M },
);
```

## Credential handling — the honesty contract

- **Fail-closed.** Any missing leg of the triple raises
  `BackendNotConfiguredError` (in-process) / `503 backend_unavailable`
  (wire) naming the missing env var — never a fabricated endpoint.
- **Never logged, never echoed.** `api_key` is used for the upstream call
  only; it does not appear in error text, completion records, usage
  logs, or the idempotency store (the dedupe fingerprint is hashed — the
  stored record carries no secret bytes).
- **Never journaled.** Under `--state-dir` the key store persists sha256
  fingerprints, and `callback_secret` is deliberately absent from job
  journals — secrets don't touch disk.
- **Per-request > env.** A `byok` body/header set on one request affects
  only that resolve; the server's `FX1_BYOK_*` stays the default for
  callers that don't override.
- **`X-API-Key` vs `X-Fx1-Byok-Api-Key`.** The first authenticates *you to
  the harness*; the second authenticates *the harness to your provider*.
  They never substitute for each other — a request can need both.

## Provider quirks — what works vs what honestly 501s

`OpenAICompatBackend` declares `complete`, `complete_with_tools`,
`stream`, `embeddings`, `count_tokens`. Whether your provider supports
each decides the verdict — the harness never fakes a missing channel:

| Surface | Route | BYOK behavior |
|---|---|---|
| Chat completion | `POST /harness/complete`, `/v1/chat/completions`, `/v1/messages`, `/v1/responses` | Always — the core link. Provider 4xx/5xx surfaces verbatim (`backend_failure`) |
| Streaming | `POST /harness/complete/stream`, `/v1/chat/completions` + `stream:true` | SSE token deltas from the provider, then the gated final — provider must support `stream:true` on chat completions |
| Tool calling | `tools`/`tool_choice`/`parallel_tool_calls`/`logprobs` fields | Passed through verbatim; a provider that doesn't know them answers its own 4xx — which surfaces as `backend_failure`/`422`, never a silent drop |
| Embeddings | `POST /v1/embeddings` | Goes to `<base>/embeddings` sibling; the request's `model` reaches the wire verbatim (the BYOK chat pin is not an embedding model). Provider without the route → its 4xx surfaces |
| Token counting | `POST /v1/messages/count_tokens`, `fx1 harness check-text`/SDK `check_text` | Uses the provider's `/tokenize` sibling route (vLLM/SGLang shape). **Providers without `/tokenize` → `TokenCountUnavailableError` → 501** — never a guessed count |
| Evals | `POST /harness/evals` (+`judge_backend`) | Full parity — BYOK works as subject and as judge (`judge_byok` for a second endpoint) |
| Batches | `POST /v1/batches`, `POST /harness/complete/batch` | Server-side queue; each line resolves the same BYOK link |
| Files / vector stores / fine-tuning | `/v1/files`, `/v1/vector_stores`, `/v1/fine_tuning/*` | Harness-side stores — independent of the model lane; BYOK orthogonal |
| `seed`, `temperature`, `top_p`, `max_tokens`, `stop`, penalties, `logit_bias`, `reasoning_effort`, `service_tier`, `prompt_cache_*`, `verbosity` | — | Declared fields pass through to the provider verbatim; a provider that doesn't know one answers honestly. Unset `temperature` defaults to 0 (the eval pin) |

Transport honesty details:

- `timeout_s` per request caps the upstream call (`(0, 3600]` on the
  wire); each backend's default is 120 s.
- `fallbacks=["hosted_k3"|"local_fx1"|"byok"]` (max 2) retry the request
  on a *later* link **only** for availability faults — a provider 4xx,
  gate refusal, or mid-stream fault never silently re-routes.
- The backend breaker (`FX1_API_BREAKER_THRESHOLD`) counts BYOK upstream
  faults too: N consecutive failures fast-fails the link 503 +
  `Retry-After` until the cooldown probe.
- Metering reads the provider's `usage` block verbatim into the
  completion record (`prompt_tokens`/`completion_tokens`); a provider
  that omits `usage` reports honestly — meters don't fabricate.
- BYOK streaming usage: the trailer's `usage` lands on the record when
  the provider emits one.

## `x-fx1-*` headers × BYOK — how they interact

On `/v1/*` surfaces these compose (all browser-exposable):

| Header | Effect with BYOK |
|---|---|
| `X-Fx1-Backend` | Explicit link selector; overrides the byok-headers → `byok` default |
| `X-Fx1-Byok-{Base-Url,Api-Key,Model}` | The BYOK credential channel — see above for precedence/required-pair rules |
| `X-Fx1-Fallbacks` | Comma-separated alternate links — each resolves the same way (a `byok` fallback still reads `FX1_BYOK_*` env when no override binds it) |
| `X-Fx1-Checkpoint-Dir` | `local_fx1` only; rejected (422) on other links |
| `X-Fx1-Timeout` | Per-request backend deadline in seconds; `(0, 3600]`, malformed → 400 |
| `X-Fx1-Receipt-Hashes` | Comma-separated sha256 citations into the completion's evidence footer |
| `Idempotency-Key` | Makes any keyed write replay byte-identically (`X-Fx1-Idempotent-Replay: true`); a key reused with a different body → 409 |
| `Last-Event-ID` | Resumes a keyed SSE replay — drops frames at/below the index; needs the original `Idempotency-Key` (409 `resume_miss` otherwise) |

The response side: `X-Fx1-Completion-Id` links the served record to its
sealed receipt (`X-Fx1-Receipt-Sha256` when the store is mounted);
`X-Fx1-Api-Version` stamps the wire contract on every response —
`HarnessClient.check_compat()` / `HarnessApiClient.checkCompat()` read it.
