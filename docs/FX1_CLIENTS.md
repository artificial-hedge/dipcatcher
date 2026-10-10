# fx-1 harness — the four client legs, same task each

One task — submit an eval, stream a response, run a batch, rotate a key —
shown on every consumption surface: raw HTTP, the in-process SDK, the
Python remote client + CLI, and the TypeScript client. Every snippet is
derived from the real method signatures; the runnable end-to-end versions
live in `examples/fx1_quickstart_sdk.py` and `examples/fx1_quickstart_http.py`.
The wire semantics (error taxonomy, idempotency, SSE frames) are in
`docs/FX1_HARNESS_API.md`; the ops runbook in `docs/FX1_DEPLOY.md`.

## The legs

| Leg | Entry point | State |
|---|---|---|
| HTTP API | `POST /harness/...` on `fx1 harness serve` | `curl` + any HTTP stack |
| In-process SDK | `from fx1.sdk import Fx1Harness` | no socket; the app embedded |
| Remote client | `from fx1.serve.client import HarnessClient` — same result types as the SDK; `fx1 harness <cmd> --remote $URL` is the same client under a flag | Python callers on a deployed harness |
| TypeScript | `import { HarnessApiClient } from "./client.ts"` (`clients/typescript/fx1`) | generated types off the pinned OpenAPI spec |

Auth on every leg is `X-API-Key: <key>` (the bootstrap `FX1_API_KEY`, or a
managed `fx1k_…` key). On `/v1/*` routes `Authorization: Bearer` works
too, so stock OpenAI SDKs drop in unchanged.

Below: `$URL` is the base (`http://127.0.0.1:8011` by default), `$KEY` a
credential, and `suite="tooluse"` + `backend="byok"` the evals the audit
suite itself uses (12 seeded tasks, no judge needed). SDK snippets assume
`fx = Fx1Harness()`; client snippets `client = HarnessClient($URL, api_key=$KEY)`;
TS snippets `const api = new HarnessApiClient({ baseUrl: URL, apiKey: KEY })`.

## Task 1 — submit an eval and wait for the terminal record

**curl**

```bash
EVAL_ID=$(curl -sf -X POST $URL/harness/evals \
  -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -H "Idempotency-Key: eval-$(date +%s)" \
  -d '{"suite": "tooluse", "backend": "byok", "seed": 0}' | jq -r .eval_id)
# -> 202 {"eval_id","status":"queued","replayed":false} + Location: /harness/evals/<id>
while :; do
  S=$(curl -sf $URL/harness/evals/$EVAL_ID -H "X-API-Key: $KEY" | jq -r .status)
  [ "$S" = "succeeded" ] || [ "$S" = "failed" ] || [ "$S" = "cancelled" ] && break
  sleep 0.5
done
curl -sf $URL/harness/evals/$EVAL_ID/receipt -H "X-API-Key: $KEY"   # sealed fx1_eval_record.v1
```

**Fx1Harness (in-process)**

```python
rec = fx.run_eval("tooluse", backend="byok", seed=0)  # runs to terminal in-process
rec.status  # 'succeeded' | 'failed' | 'cancelled'
doc = fx.eval_receipt(rec.eval_id)  # sealed fx1_eval_record.v1
```

**HarnessClient + CLI**

```python
sub = client.submit_eval(
    "tooluse", backend="byok", seed=0, idempotency_key="eval-1"
)  # -> {"eval_id","status","replayed"}
rec = client.wait_eval(sub["eval_id"], timeout_s=120)  # terminal record, or HarnessJobError
doc = client.eval_receipt(sub["eval_id"])  # sealed doc (409 while non-terminal)
```

```bash
fx1 harness eval tooluse --backend byok --remote $URL --api-key $KEY          # submit + wait
fx1 harness eval tooluse --backend byok --remote $URL --receipt               # + sealed doc
fx1 harness eval tooluse --remote $URL --no-wait                              # submit, return the id
fx1 harness eval-status <eval_id> --remote $URL; fx1 harness eval-wait <eval_id> --remote $URL
```

**TypeScript**

```ts
const sub = await api.submitEval(
  { suite: "tooluse", backend: "byok", seed: 0 },
  "eval-1",                                          // optional Idempotency-Key
);
const rec = await api.waitEval(sub.eval_id, { timeoutS: 120 });  // terminal record returned, not thrown
const doc = await api.evalReceipt(sub.eval_id);                  // sealed fx1_eval_record.v1
```

## Task 2 — stream a gated response

The stream surfaces deliver **buffered, gated** deltas: the joined text
passes the honesty gate before chunks reach you, so a refusal is a mapped
error, never a truncated stream.

**curl** (SSE: `event: token` frames, then `event: final` carrying the
full `CompleteResponse`; keepalive `: ...` comments every
`--sse-keepalive-s`)

```bash
curl -N -X POST $URL/harness/complete/stream \
  -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{"backend":"byok","messages":[{"role":"user","content":"summarize this receipt"}]}'
```

**Fx1Harness**

```python
chunks = fx.stream_complete(
    [{"role": "user", "content": "summarize this receipt"}],
    backend="byok",
)
print("".join(chunks))
```

**HarnessClient + CLI**

```python
chunks = client.stream_complete(
    [{"role": "user", "content": "summarize this receipt"}],
    backend="byok",
)  # -> list[str]
```

```bash
fx1 harness complete "summarize this receipt" --backend byok --stream --remote $URL --api-key $KEY
```

**TypeScript** — the per-event callback gets each SSE frame
(`token`, `final`):

```ts
const final = await api.streamComplete(
  { backend: "byok", messages: [{ role: "user", content: "summarize this receipt" }] },
  (ev) => {
    if (ev.event === "token") process.stdout.write(JSON.parse(ev.data).token);
  },
);                                                   // -> CompleteResponse | null
```

## Task 3 — run a batch of completions

**curl** — one request, the server parallelizes over a shared backend;
per-item verdicts in `results[]`:

```bash
curl -sf -X POST $URL/harness/complete/batch \
  -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -H "Idempotency-Key: batch-1" \
  -d '{"backend":"byok","max_workers":4,"batch":[
        [{"role":"user","content":"alpha"}],
        [{"role":"user","content":"beta"}]]}' | jq '.results[] | {ok, content, completion_id}'
```

**Fx1Harness**

```python
results = fx.complete_many(
    [[{"role": "user", "content": "alpha"}], [{"role": "user", "content": "beta"}]],
    backend="byok",
    max_workers=4,
)  # -> list[CompletionResult]; lowest-index failure raises
```

**HarnessClient + CLI**

```python
results = client.complete_many(
    [[{"role": "user", "content": "alpha"}], [{"role": "user", "content": "beta"}]],
    backend="byok",
    max_workers=4,
    idempotency_key="batch-1",
)
```

```bash
printf '["alpha","beta"]' > prompts.json
fx1 harness batch prompts.json --backend byok --workers 4 --remote $URL --api-key $KEY
fx1 harness batch prompts.json --out results.json --remote $URL --api-key $KEY
```

**TypeScript**

```ts
const out = await api.completeBatch(
  {
    backend: "byok",
    max_workers: 4,
    batch: [
      [{ role: "user", content: "alpha" }],
      [{ role: "user", content: "beta" }],
    ],
  },
  "batch-1",                                            // optional Idempotency-Key
);                                                      // -> CompleteBatchResponse (per-item ok/error)
```

## Task 4 — rotate a managed key

Rotation is atomic on every leg: the successor mints under the
predecessor's declared policy (name/scopes/admin/rpm/budgets) and, by
default, the old secret tombstones in the same transaction — the
response's key payload is the only place the new raw secret appears.
Pass keep-old (`revoke_old`/`revokeOld` false) for a verify-first
cutover where both secrets authenticate until the old key is revoked or
expires. Without `ttl_s`/`ttlS` the successor inherits the
predecessor's absolute expiry — rotation never extends a credential's
lifetime.

**curl**

```bash
NEW=$(curl -sf -X POST $URL/harness/keys/<old-id>/rotate -H "X-API-Key: $KEY" \
  -H "Content-Type: application/json" -d '{"revoke_old":true}' | jq -r .key.key)
# ^ 'key' under 'key' is the only place the raw fx1k_… ever appears;
#   'id' is the fingerprint used below.
# Verify-first alternative — run this instead of the default rotation above:
# curl -sf -X POST $URL/harness/keys/<old-id>/rotate -H "X-API-Key: $KEY" \
#   -H "Content-Type: application/json" -d '{"revoke_old":false}'       # overlap
```

**Fx1Harness** (the in-process store is the same `ApiKeyStore`)

```python
rot = fx.key_rotate("<old-id>")  # {"object":"key_rotation", "key":{...,"key": raw}}
new_key, new_id = rot["key"]["key"], rot["key"]["id"]
# Alternative before default rotation: rot = fx.key_rotate("<old-id>", revoke_old=False)
fx.key_update("<id>", rpm=120, clear=["max_tokens"])  # PATCH twin: in-place policy
```

**HarnessClient + CLI**

```python
rot = client.key_rotate("<old-id>")  # rotate -> {"key": {..., "key": raw}}
client.key_update("<id>", rpm=120, clear=["max_tokens"])  # -> updated record
# unknown id -> KeyError; revoked -> 409-shaped error; auth with an old
# revoked secret -> HarnessAuthError on the next call
```

```bash
fx1 harness key-rotate <old-id> --remote $URL --api-key $KEY             # atomic (default tombstone)
# Alternative before default rotation:
# fx1 harness key-rotate <old-id> --keep-old --ttl-s 86400 --remote $URL --api-key $KEY
fx1 harness key-patch <id> --rpm 120 --clear max_tokens --remote $URL --api-key $KEY
fx1 harness keys --remote $URL --api-key $KEY          # list incl. tombstones
fx1 harness key-usage <id> --remote $URL --api-key $KEY  # budget headroom (admin)
fx1 harness self --remote $URL --api-key $KEY            # caller's own card (read)
```

**TypeScript**

```ts
const rot = await api.keyRotate("<old-id>");        // { key: { key: raw, id, ... }, revoked_previous: true }
const newKey = rot.key.key;                         // raw fx1k_… — once only
// Alternative before default rotation:
// await api.keyRotate("<old-id>", { revokeOld: false, ttlS: 86400 });
await api.keyUpdate("<id>", { rpm: 120, clear: ["max_tokens"] });  // PATCH twin
const card = await api.keyUsage(rot.key.id);        // budget headroom (admin scope)
const mine = await api.selfUsage();                 // the calling key's own card (read)
```

A manual mint → switch callers → revoke sequence still works when the
successor should carry a *different* policy than the predecessor's
(rotate inherits it verbatim): `keyCreate`/`key_create`, verify, then
`keyRevoke`/`key_revoke`. Revocation leaves a tombstone for audit, and
subsequent requests using the old credential fail closed immediately.

## Reading a failure the same way on every leg

The wire's `code` survives every translation:

```python
# HarnessClient / SDK raise mapped classes; curl/TS read the envelope.
try:
    client.complete(messages, backend="byok")
except HarnessAuthError:  # 401/403 — invalid key, insufficient scope, admin required
    ...
except (
    HarnessTransportError
) as exc:  # other non-2xx + transport faults; exc.code carries the wire code
    ...
```

```ts
try {
  await api.complete({ backend: "byok", messages });
} catch (e) {
  if (e instanceof HarnessApiError) console.error(e.status, e.detail);  // {detail, code}
}
```

Retry rules worth copying into your own client: `429 rate_limited` and
`503 over_capacity`/`draining`/`backend_unavailable` come with
`Retry-After` — honor it. `429 quota_exceeded` deliberately carries no
`Retry-After` — the budget is terminal; don't retry. Both Python and TS
clients implement exactly this (idempotent calls retried by default under
`max_retries`/`maxRetries`; writes only ever retry when they carry an
`Idempotency-Key` — `retry_writes`/`retryWrites` widens keyed writes,
never unkeyed ones).
