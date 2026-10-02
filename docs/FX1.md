# fx-1

**fx-1** (always lowercase, package `src/fx1`) is an in-tree sub-project. It
contains the corpus builder, the eval bank, and training plumbing, plus a
training plan. No trained checkpoint is in this repository. Local generation
is unimplemented. The declared base constant is `moonshotai/Kimi-K3` (2.8T
total / 104B active MoE, Kimi K3 License).

**dipcatcher** is the harness: the data engine, evaluation bench, and
verification layer. `src/fx1/harness.py` is the typed bridge; `fx1 harness list`
shows the registered lab surfaces.

## Consuming the harness

The same contract — registry, backends, honesty gate, receipt verifier —
is exposed four ways, all implemented over one code path:

| Surface | Entry point | Use when |
|---|---|---|
| HTTP API | `fx1 harness serve` → `src/fx1/serve/api.py` | fx-1 (or any service) calls over the network |
| Typed SDK | `from fx1.sdk import Fx1Harness` | in-process Python — no socket |
| CLI | `fx1 harness {list,run,complete,batch,verify,health}` | shell/CI |
| Direct | `Harness().run(...)` / `get_backend(name)` | library composition |

Completion routes beyond `POST /harness/complete`:
`POST /harness/complete/batch` fans up to 64 conversations over one
shared backend (per-item `ok`/`error_class` verdicts; a gate refusal
fails the slot, not the request — SDK twin `complete_many`), and
`POST /harness/complete/stream` emits SSE `token` events + a `final`
envelope + `[DONE]` — deltas are buffered and the joined text passes
the honesty gate before any frame leaves, so gate refusals are plain
JSON 502s, never truncated streams (SDK twin `stream_complete` returns
the gated chunk list).

Backends (the model side of `complete`/eval lanes):

- `hosted_k3` — the K3 endpoint (`MOONSHOT_API_KEY`), temperature 0.
- `local_fx1` — weights-direct: attaches to a running engine
  (`FX1_LOCAL_SERVE_URL`) or spawns one (`FX1_LOCAL_SERVE_CMD`, with
  `$checkpoint_dir`/`$python` template vars); a card'd checkpoint dir
  (`--checkpoint-dir` or `FX1_CHECKPOINT_DIR`) is required and
  signature-gated before any spawn.
- `byok` — bring-your-own-key to any OpenAI-compatible endpoint:
  `FX1_BYOK_BASE_URL` + `FX1_BYOK_API_KEY` + `FX1_BYOK_MODEL` (kwargs
  beat env). Construction fails closed on missing credentials; the URL
  must be http(s) with a netloc.

Every surface returns structured errors mirroring HTTP status classes
(`404` unknown command/backend, `422` contract violation, `503`
unconfigured backend, `502` honesty-gate refusal), and `complete` always
closes the backend — spawned engines never leak. The honesty gate runs
before output bytes reach the caller; cited receipts are appended to
completions as a provenance footer.

`fx1 harness serve` binds loopback-only unless `FX1_API_KEY` is set, in
which case every route requires `X-API-Key` (constant-time compare);
`/health` leaks presence booleans only — never env values.

The CLI fronts either surface: every `fx1 harness` subcommand takes
`--remote URL` (drives the API through `HarnessClient` — the same wire
client fx-1 uses) plus `--api-key`/`--timeout`, falling back to
`FX1_API_KEY`; without it they run the in-process SDK. `fx1 harness
batch prompts.jsonl` reads a JSON array or JSONL of strings/`{"prompt":
...}` records and writes the gated completions as JSON (stdout or
`--out`). Faults are one clean stderr line plus exit 2 — never a
traceback. Parity of all three surfaces (SDK / API / remote client,
byte-identical payloads and error classes) is sealed by
`receipts/fx1_parity_audit.json`; the real-socket lifecycle is sealed by
`receipts/fx1_e2e_audit.json`.

## What the plumbing enforces

The table is the corpus and eval contract. It is a plan for a future training
run, and a record of what the package checks today.

| Property | General LLM | fx-1 plan |
|---|---|---|
| Training data | Web-scale text | Only dipcatcher artifacts whose receipts pass `verify-research` — notebooks, scorecards, bench outputs, tournament ledgers, gate decisions |
| Negative examples | None | Gate rejections and live-claim artifacts, teaching refusal and honest reporting |
| Provenance | None | Every training example carries the SHA-256 of its source receipt |
| Honesty contract | Prompt scaffolding | Tested natively: `fx1.honesty` + eval harness block forbidden Sharpe/P&L headlines, live-performance claims, unlabeled synthetic evidence |
| Evidence classes | Blurred | Explicit: research / backtest / simulated paper / SYNTHETIC |

## Package layout (`src/fx1/`)

- `data/receipts.py` — receipt loading and eligibility (`research_only=true`,
  `live_pnl_claim=false`); ineligible artifacts become negative examples.
- `data/corpus.py` — SFT corpus builder → JSONL, one `SFTExample` per line,
  each with `receipt_sha256` provenance. Run:
  `python -c "from fx1.data import build_corpus; build_corpus('receipts', 'data/fx1/corpus.jsonl')"`
- `eval/suite.py` — deterministic eval harness (honesty / domain / general
  task kinds). Runs **before** any training; the honesty gate blocks shipping.
- `eval/capability.py` — aggregate capability battery over the sealed
  SYNTHETIC banks: `ts_reasoning`, `calibration_eval` (ECE + Spiegelhalter Z,
  gate at 0.02), `tooluse_eval`, `retrieval_eval`. One `ModelFn` in, one
  `CapabilityEvalReport` out; honesty sub-gates are hard. CLI:
  `fx1 capability-eval --backend hosted_k3|local_fx1 --seed N`.
- `train/config.py` — validated run config: ladder stage (`proxy` →
  `final_k3` → `distill`), LoRA hyperparameters, mandatory cost disclosure,
  K3 constraints (multi-node, preserved `reasoning_content`).
- `train/run.py` — `build_training_manifest`: enforces eval-before-train and
  corpus provenance, then writes an immutable run manifest
  (`live_pnl_claim=false`, `research_only=true`) for the cluster launcher.
- `prompts/fx1_system.md` — the system prompt / behavioral contract written
  into the corpus builder.

## License

Kimi K3 License: internal/research use is unrestricted. A Model-as-a-Service
business above $20M revenue requires a separate Moonshot agreement; above
100M MAU or $20M monthly revenue requires UI attribution. Current tier:
**internal_research**. Re-check at every fx-1 release.

## Hard boundaries (inherited from the lab)

The package leaves lab data, gates, receipts, and promotion logic to the
harness. It refuses live-performance claims and keeps synthetic results
labeled as synthetic. These rules are enforced in code (`tests/fx1/`) and in
the corpus builder.
