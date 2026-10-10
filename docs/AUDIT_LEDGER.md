# Audit Ledger

Per-directory evidence for `quality/audit_coverage.json`. "Audited" means every
module was read and its risk-bearing paths verified against the honesty
contract: strict causality (no feature reads data after the decision origin),
fail-closed validation (degenerate input raises or returns a labeled NaN),
honest labeling (SYNTHETIC forced, forbidden metrics gated), and durable writes
(atomic, sealed where they are evidence).

| Directory | Verified | Findings → fix |
|---|---|---|
| `api/` | `app.py` line-by-line: loopback-only auth default + constant-time compare, TOCTOU-safe receipt verification (hash before+after), path-containment on configs/artifacts, request size limits, security headers, `_stamp_research_honesty` force-overwrite on every response | — |
| `audit/` | Merkle-chained ledger, flock'd cross-process appends, checkpoint integrity; Windows O_BINARY/LF-only fix | #286 stress tests, #293 Windows corrupt-ledger |
| `backtest/` | loop.py divergence bookkeeping, fast_replay byte-identity vs reference (P4.2), sleeves incl. residual_mr | #215 mark-price leak on non-executing names, #338 divergence-resume weighting |
| `backtest/event_sim/` | All 7 modules line-by-line (child lane, #343): NAV/gating on pre-exec marks, resting-order expiry on missing/NaN tape, borrow charge unclamped, turnover per fill bar, duplicate-key bar rejection | #343 — 11 defects fixed (mark-leak parity with #215, starved orders, free-margin borrow clamp) |
| `calendars/` | Session/window arithmetic; trading-day math verified against exchange-calendar semantics | — |
| `cli/` | All 15 modules: argv-list subprocess only, no shell, config-error UX | #205 doctor UX |
| `compute/` | Per-task seed derivation (murmur-finalize mix) so process_map result order = item order; serial fallback on pool failure; task exceptions propagate | — |
| `config/` | safe_load, inherit path-containment, cycle detection | — |
| `data/` | PIT sources, atomic writes, universe tie parity, manifest seals | #176 P6.3 fixes, #261 manifest seal, #275 pooled_stream atomic |
| `diffbacktest/` | `delay>=1` structural gate; `end = t - delay` verified causal in every builder; adversarial radius documented as upper bound | — |
| `execution/` | Money-path fills, costs | #173 P6.6, #174 P6.1 (exec-NAV leak, leverage-cap trap) |
| `features/` | lot_spread eps dead-zone parity with zero_share | #337 |
| `formal/` | Z3-checked closed forms (NAV/cost/split/dividend); sim-only lifecycle spec — nothing submits orders; real-vs-IEEE gap documented | — |
| `fusion/` | Cross-fitted ridge stacking is OOF-only (leakage-safe); finite guards on stacker inputs/params | — |
| `hedge_lab/` | Every module line-by-line: slate lanes, book, race, hunt, tape, zoo | #320 same-bar look-ahead (delay>=1), #323 benchmark date-alignment + POSIX RAM |
| `hmm/` | Stochasticity contract, log-space variants | #307 EM underflow + contract enforcement |
| `labels/` | Triple-barrier observability, full-path gating | #212 (pipeline join), #325 last-bar events |
| `leakage/` | Scanner rule-by-rule | #339 (child lane: 14+ evasion classes) |
| `lightspeed/` | Vol measurement → sizing causality | #328 unmeasurable vol de-risks flat |
| `market_sim/` | Agent/book/simulator/ecology, AS quote guards, metaorder counters | #284 (child audit), #335 edge tests |
| `mc_engine/` | Philox counter streams, chunk-pure engine, checkpoint fingerprinting, TDigest/Welford/P² merges, POT/GPD | #268 KATs, #295/#319 coverage |
| `metrics/` | Proper-score suite, VaR backtests, inference (HAC/DM/MCS) | #206 clustered-variance, #304 propriety, #312 DQ, #315 KLM |
| `microstructure/` | Quote/impact estimators | #175 P6.5 |
| `models/` | All 152 modules estimator-by-estimator (child lane, `AUDIT_MODELS.md` on #345): estimator contracts, NaN/fail-closed edges, spec drift | #345 — 25 contract fixes |
| `monitoring/` | PSI drift emits `insufficient_data` not false all-clear; kill_switch: unknown state blocks, invalid transitions raise, auto-flatten never on exception | — |
| `native/` | env-gated dispatch; docs honestly separate bit-exact kernels (cumsum, wealth, SHA-256) from tolerance kernels (EMA/RSI/book-OLS) | — |
| `northset/` | All 9 modules: estimators (Kyle/Roll/Parkinson/GK/RS/YZ/CS/AR/CKS/BNS formula-verified), CKS OFI, sweep battery (executable next-open timing, cross-sectional demeaning, PIT vol regimes, matched eligible controls, predeclared primary test) | #317 flat-bar junk + VPIN remainder |
| `observe/` | OTLP export error swallowing, log redaction | #336 |
| `paper/` | loop, ledger, sim_live resume cursors, divergence merge, atomic receipts | #258 seals, #303 fail-closed cursors, #338 |
| `parity/` | `__main__` dual-clock smoke, fail-closed report | — |
| `pit/` | Vault enforcement | #289 (child: 12 enforcement gaps) |
| `pipeline/` | All 21 modules: purged walk-forward (observed label-end fallback), asof-bounded everything, unfitted-clone GARCH/RGARCH refits, spec-bytes + full-history digest caches, named-estimator family/spec/object pinning, PIT history builders (strict `<` origins, null `available_time` fail-closed, duplicate-key rejection on the full frame), labeled homoskedastic fallback emitted per-row | #213 (calibration unrealized-label fallback) |
| `portfolio/` | Optimizer (infeasible diagnostics), allocators (PSD cov), factor betas (trailing ridge), conformal (chrono split), attribution (prev-bar weights) | #332 non-PSD refusal, fingerprint framing |
| `pretrade/` | Hot risk-gate surface | #287 (child audit doc) |
| `proof/` | HMAC env-only, merkle canonical dumps | — |
| `research/` | 62 modules audited via `AUDIT_RESEARCH.md` on #341 (receipt writers/verifiers first, then signal→return causality, forced SYNTHETIC stamps, degenerate-input handling); +51 modules built this campaign, each shipped with sealed receipts + contract tests: 19-lane sequential-inference suite (e-processes, MCS, monitors — #380-#410), `receipt_lattice` cross-claim determinism (#414), `corpus_epoch` hash-chained membership root + committed head pins (quality/epoch_heads.json — rollback fails closed), `crown_jewels` byte-pin over gate-defining files (coverage set is code-defined — the pin cannot silently shrink), `admission` gate | #341 — `oos_rank_scores` no-fold fallback lacked boundary purge (real look-ahead, fixed); `sota_receipt.json` unsealed/non-atomic (fixed); `benches/common.py` purge gap fixed on #342 (documented below); #394 `changepoint_localize` max-over-candidates level inflation (Bonferroni-fixed); #380 e-process clipped-bet supermartingale violation (sign-bet fix) |
| `proofcore/` | Hash-chain export ordering | #279 wall-clock → genesis walk |
| `quant_models/` | BS/Greeks/GEX/HRP/TSMOM/risk-parity/MC; causal TSMOM, honest receipts | — |
| `reality/` | Bailey–LdP PSR/DSR/MinTRL, CSCV/PBO ω-logit, SPA, BH-FDR — all verified against papers | — |
| `registry/` | MLflow store: artifact identity binding, research_only claim tags on every run | — |
| `reporting/` | Tearsheet return-chain integrity | #326 interior NaN breaks chain |
| `risk/` | Overlay causality, gate stack, ruin latching | #308 ruin latch, #310 fail-closed tail gates |
| `robustness/` | Closed forms, swarm invariants | #264 KATs |
| `schemas/` | Fail-closed pydantic validators (finite, tz-aware, book ordering, crossed-book rejection) | — |
| `simtest/` | Deterministic exchange timeline, IEEE-hex event log, real ddmin shrinking, replay re-derivation, conservation rebuild | — |
| `stress/` | replay/reverse/bootstrap/garch_copula/jumps | — |
| `utils/` | Hashing, atomic io | #161 selector injection, #332 fingerprint framing |
| `validation/` | Purging/embargo/CPCV/walk-forward/FDR line-by-line vs papers | #211 (P6.2 audit + 22 KATs) |

## Deferred findings (audited, fix pending)

| Item | Where |
|---|---|
| `benches/common.py` split helpers (`_holdout`/`_triple_split`/`_gaussian_interval_split`) — chronological row cuts with no boundary label purge | found via #341's `docs/AUDIT_RESEARCH.md`; **fixed on #342** — splits now return boolean row masks cut at unique-date boundaries via `purge_mask` on both edges (train↔cal and cal↔test), callers fail closed on emptied sides, OnlineCRC warms only on kept rows |

## fx1 (the gated product — `quality/audit_coverage_fx1.json`)

Parallel manifest, same schema + ratchet tests: 115/116 modules `audited`
(`__init__.py` waived as package surface). Directory pins: eval 28, forecast
14, bench 5, serve 14, data 20, train 13; root modules audited individually.

| Area | Verified | Findings → fix |
|---|---|---|
| `honesty.py` + `data/` | Child audit — regex evasions, corpus screening, contamination floor, vacuous honesty gate | #321 |
| `eval/` | Deterministic seeded banks, contract-validated prompts, unparseable answers counted (never propagated), degenerate forecasts fail closed NaN, `passed` requires finite ECE AND finite \|Z\| | — |
| `serve/` | HMAC-env signing (never hardcoded), fail-closed compare_digest verify, structural TEE + zkML manifests with honest crypto delegation; consumption lanes: BYOK OpenAI-compat backend (fail-closed creds, temp-0 wire), harness HTTP API (auth/loopback-only, body cap, batch + SSE stream routes gated before bytes leave), `sdk.py` in-process twin, `parity_audit` — SDK↔API↔HarnessClient byte-identical content/envelope/errors/verifier over one injected backend + in-flight cap; `e2e_audit` — full lifecycle over real loopback sockets (stub OpenAI engine + uvicorn + production middleware), auth/gate/size-cap faults survive the wire; `contract_audit` — OpenAPI surface pinned to a committed golden (routes/methods/params/responses/body-required), additive+breaking drift fails the seal; `fault_audit` — adversarial-condition battery (torn/reordered key journals, mint/revoke/quota/token races, TTL+quota+rpm boundaries, wire abuse + CRLF/NUL header + BYOK smuggling, same-key idempotency races, drain latch, readonly/missing state-dir). The historical SYNTHETIC receipt `receipts/fx1_fault_audit.json` records 49 probes and 7 divergences; current expectations are verified by live regression checks and never by rewriting that receipt | #839 |
| `bench/` + `forecast/` | `bench/dip.py` verified: causal dip detection (fires below running peak only), recovery windows bounded at data end → `None` never imputed, out-of-range probabilities raise, `assert_bench_output_honest` mirrors the forbidden-token contract inside the bench itself; `forecast/evaluate.py`: `assert_no_label_overlap` per fold, walk-forward fails closed on empty folds, finite-masked metrics; `runner.py` decision-time machinery (`_at_decision`, late-release flag) | — |
| `train/` + root modules | `train/pipeline.py`: hard stage gates, mandatory pre-training honesty gate (`eval_base` blocks TRAIN when `honesty_gate_passed` is false), corpus dedup/decontamination vs eval prompts, frozen split manifest; `train/receipts.py`: receipt pins SHA-256 of config/corpus/split/eval_base + git rev + dirty flag + env fingerprint with forced `live_pnl_claim=False`/`research_only=True`; `default_trainer` fails closed (no local GPU trainer — injectable); `reward.py`: executable honesty-contract reward (forbidden-headline/live-claim/synthetic-unlabeled → low score); `hypotheses.py`: only gate-resolved traces admissible; `mrm.py`: dossier sections cite artifact hashes, missing artifact fails closed | — |
| wave-3 depth pass — all 66 files | Full re-verification of every module against fail-closed/no-silent-pass/no-forbidden-metric contracts; evidence in `docs/AUDIT_FX1_WAVE3.md` | 26 defects fixed, incl. `eval_prompt_surface` bare-global `NameError` + user-only role filter, windowed near-dup dedup (`[-500:]` evasion), boundary-ambiguous split hashing, vacuous frozen-split/contamination-audit passes, truthy non-bool `research_only`/`live_pnl_claim` evasion, unverified `verify_ok` traces, free-form ledger hashes, 1500-char notebook truncation, non-date `as_of` PIT keys, honesty-gate reporting-verb phrasing, trace-score substring evasion, dip recovery off-by-one, silent out-of-universe forecast drops, normalized-key bench evasion, `eval_base` hash taken on faith (now re-verified vs receipt), eval gate checking only the top flag, non-atomic evidence writes across mrm/bench/train, `--config` containment bypass, null→`"None"` completion coercion |

## Rules

- A directory may only move `partial → audited` when every module has a row of
  evidence here or in a linked audit doc (`AUDIT_P6*.md`, child audit tables).
- `test_audited_dirs_pin_file_count` fails the build if an audited dir gains or
  loses a module without a manifest update.
- `AUDITED_FLOOR` ratchets only upward.

### Observability audit maintenance (PR #2805)

`observe_audit` checks selected health/readiness, metrics, Prometheus,
drain, request-id, error-envelope, scope and client contracts using SYNTHETIC
stub backends and in-process ASGI clients. Routing-level 404/405 exceptions
now use the existing structured error handler without losing exception headers.
The resolved audit rejects empty results, checks HELP < TYPE < first sample,
isolates app creation from ambient backend/state settings, and observes a real
condition wait before releasing the held drain workload. A resource stack closes
HTTP clients and joins job-executor workers on both successful and failed audits,
before temporary persistent state is removed.

The incoming `fx1_observe_audit.json` was not accepted as evidence for this
resolved source: it names an earlier parent revision and the original helper
assertions were incomplete. No historical receipt was rewritten. A passing
local battery is a bounded correctness check, not exhaustive input coverage,
network-transport validation, external-process restart evidence or market evidence.


### Upload lifecycle maintenance (PR #2808)

The new `uploads_audit` checks 93 selected upload lifecycle, checksum, part-bound,
expiry, tenancy, recovery, downstream-plumbing and client-error contracts using
SYNTHETIC stubs. Its existing empty-result refusal is retained and now explicitly
names `no_probes`. A resource stack closes clients and joins their job workers
before removing private state, including on probe failures; replay probes close
one application before constructing the next. Ambient FX1/MOONSHOT settings are
restored. Run this diagnostic in a dedicated process because environment overrides
are process-wide. Worker exceptions propagate instead of being lost in threads.

Additional deterministic regressions exposed a publication race: two completions
could mint two files while only one upload completed. HTTP and SDK now use one
store-owned completion operation, holding the upload lock across assembly, checksum
validation, publication and the durable terminal transition. Before-write upload
journal failures preserve retryable metadata and previously durable part bytes;
a newly published file is removed only when the terminal journal is demonstrably
unchanged. Add-part, cancel, expiry and create/eviction also journal before making
the corresponding state visible or dropping durable parts.

These checks do not establish atomicity across the separate upload and file
journals, recovery from arbitrary partial filesystem writes, rollback of unrelated
file-capacity evictions, external-process crash recovery, network behavior, or
fine-tuning corpus/training quality. The error-envelope handler was already fixed
in #2805. The incoming all-pass `fx1_uploads_audit.json` names the earlier parent
`a0cb7c9` and is excluded; historical receipts remain unchanged. The serve census
moves from 39 to 40 and remains `partial`.


### Model registry maintenance (PR #2812)

The new `models_audit` checks 157 selected registry contracts — list/retrieve
envelopes in both dialects, cursor paging in both directions, `ft:` mint →
resolve → tombstone lifecycle, checkpoint gating, delete idempotency,
job↔registry sync, `model` parameter validation, state-dir restart
durability, concurrent mints, and enveloped refusals for malformed,
traversal and oversized ids — over SYNTHETIC stub backends, a gate backend
for mid-flight reads, in-process ASGI clients and a spy resolver. The audit
reuses the conversation lane's resource context and gate backend rather than
re-shipping that scaffolding.

Five registry defects were found and fixed: `ft:` model cards stamped
`created` at application boot instead of the registration instant (cards now
carry the registry stamp in both dialects); an unknown non-`ft:` model fell
through to the `hosted_k3` default link (completion surfaces now refuse with
`model_not_found` 404 — embeddings keep free-form upstream names); the
response `model` reported the checkpoint backend's version rather than the
served `ft:` alias (a `served_model` attribution is plumbed end to end); an
empty `model` string passed validation (`min_length=1` → 422); and a
header-carried `X-Fx1-Checkpoint-Dir` on a non-`local_fx1` link escaped the
request validator as a 500 (now enveloped 422). The committed TypeScript
OpenAPI golden was already stale on the parent and is regenerated.

The incoming all-pass `fx1_models_audit.json` names the earlier merge base
and is excluded; historical receipts remain unchanged. The serve census
moves from 43 to 44 and remains `partial`.


### Idem audit maintenance (PR #2816)

The new `idem_audit` checks 98 selected idempotency contracts using SYNTHETIC
stubs: `Idempotency-Key` coverage on the selected idempotency-enabled mutating
routes (chat, responses,
messages, completions, batches, message-batches, fine-tuning jobs, runs,
evals/runs, jobs, uploads create/parts/complete/cancel, files, and key
mint/rotate/patch/revoke), byte-identical replay with no re-execution, the
deterministic 409 `idempotency_conflict` in both error grammars, per-route and
per-credential namespacing, key hygiene, LRU eviction without tombstone
resurrection, replay-after-delete serving the stored snapshot, SSE replay, a
`--state-dir` restart restoring the journal, and same-key concurrency (one
execution, losers replay or 409).

The battery found four live defects, fixed on the same PR: key-lifecycle and
upload routes ignored the header entirely (a retried mint fabricated a second
credential); the sync `_IdemStore` routes plus `_submit_eval` and fine-tuning
ran lookup→execute→put without a claim lock, so parallel same-key submits could
double-execute (a claim dep now holds `store.async_claim_lock` across the
handler span); claims were credential-blind until `_idem_scope` bound them to
the request's authenticated key id plus a per-(verb, target) namespace; and the
409 code drifted by route until it was unified on `idempotency_conflict`. Key
replay records stay in-memory only — the stored mint answer carries the raw
secret, which never persists.

These checks do not establish cross-store atomicity, distributed-lock
durability across multi-process deployments, TTL expiry probes (only LRU
eviction is exercised), or client-SDK retry behavior. The incoming all-pass
`fx1_idem_audit.json` is excluded; historical receipts remain unchanged. The
serve census moves from 44 to 45 and remains `partial`.
### BYOK surface maintenance (PR #2817)

`byok_audit` grows from a 251-line backend-only battery into a whole-surface
credential-isolation audit over the whole BYOK serve surface — the
`fx1.byok` body block and `X-Fx1-Byok-*` headers on chat, responses,
messages, legacy completions, embeddings, `/harness/*`, batches and the
SDK twin. Every probe runs offline against a threaded localhost stub
that records the real wire per hit: path, headers and body.

The battery found and this change fixed five defect classes. A shared
`byok_base_url_problem` rule now refuses userinfo, query, params,
fragment and bad ports on the body, header, env and SDK paths without
echoing the pasted URL. Malformed `X-Fx1-Byok-*` headers surface 422
instead of a bare 500. `fx1.byok` off a `byok` link — including an
`ft:` model that previously dropped the credential silently — refuses
422 at link resolution, and residual request-model failures envelope
as 422s at construction. Validation detail redacts `input`, `ctx` and
`url` on credential locs (`byok`, `judge_byok`, `api_key`,
`*_key`/`_secret`/`_token`/`_password`) while keeping the pinned echo
for ordinary fields. Transport faults — `TimeoutError`, `OSError`,
`http.client.HTTPException` — and non-JSON upstream bodies no longer
escape the error envelope as bare 500s at any OpenAI-compatible call
site (BYOK, `local_fx1` shared helpers, the hosted link); all map to
enveloped 502s. Credentialed upstream calls refuse redirects before a
second origin is contacted, and transport errors do not echo the configured
URL or path into caller-visible details.

Pinned semantics, exercised not assumed: the `fx1.byok` body block does
not self-select the `byok` link (it needs `fx1.backend="byok"`,
`model="byok"` or the header triple); response `model` is the requested
`byok.model`; batch header credentials are `PrivateAttr` — in-memory
only, never journaled; a read-scoped key and an over-budget key refuse
before any upstream contact; `store:false` persists nothing while still
streaming, and `store:true` persists the record, never the credential;
idempotent replay never re-hits the upstream. A plain-`http` upstream
remains accepted by design (local vLLM/Ollama shims).

The sealed `fx1_byok_audit` names this pre-merge parent and is excluded
from the diff per receipt convention; historical receipts remain
unchanged. Probes are SYNTHETIC — labelled stubs, no market evidence.
The serve census stays at 45 (the module was already counted) and
remains `partial`.
### Cross-key tenancy (PR #2818)

The new `tenancy_audit` checks 141 tenancy contracts — the ownership
matrix over every stateful family (responses, conversations, files,
uploads, batches, evals/specs, ft jobs, the `ft:` registry, vector
stores, stored completions, harness jobs), secrets-leak channels,
authentication ambiguity, per-key metering, idempotency namespacing, quota
independence, BYOK isolation, revocation boundaries, drain, and
restart durability — over SYNTHETIC stub/gate backends and isolated
state dirs. The battery reuses the conversation lane's resource
context, gate backend, and client plumbing.

The measured contract is scope-based shared workspace, not per-principal
tenancy: every cross-key matrix cell holds — any key holding the verb's
scope reads, mutates, and deletes any peer's resource; isolation is
claimed (and proven) only for metering, admin surfaces, header
ambiguity refusal, key material, budgets, and tombstones. `rate_limit_rps` is
a per-client-host valve; eval runs expose no mid-flight cancel (409 for
minter and peer alike); the max_tokens budget admits the call that
crosses the cap and refuses the next.

One defect was found and fixed: `_REQUEST_KEY_ID` did not propagate
into `jobs_executor` workers, so `background=true` responses ran
unattributed (`key_id: null` in the completions ledger) and unmetered
(`tokens_used: 0`); the principal is now captured at submit and
restored inside `_bg_run`, mirroring the batch worker's
`batch._key_id` handling. The two cross-credential idempotency probes
also pass on the integrated tree because #2816 scopes replay claims by
the authenticated key id and stores only digested client keys.

The incoming `fx1_tenancy_audit.json` names the earlier merge base and
is excluded; historical receipts remain unchanged. The serve census
moves from 45 to 46 and remains `partial`.

### Header surface maintenance (PR #2819)

The new `header_audit` checks selected header contracts across the
`X-Fx1-*`, authentication, idempotency, request-id, Anthropic,
rate-limit, CORS, response-evidence and request-framing surfaces. The
battery uses SYNTHETIC stub backends, a resolver spy, in-process and raw
ASGI calls, and loopback socket fixtures; it is correctness evidence,
not a claim about every reverse proxy or HTTP server.

Malformed BYOK header models now land in the caller's error grammar,
managed-key refusals retain already-consumed global rate-limit headers,
malformed `Content-Length` uses the route's error envelope, and token
count transport failures map to `502 backend_failure`. Security-sensitive
headers are singletons: duplicate `X-Fx1-*`, authorization, idempotency,
request-id and framing headers fail closed before Starlette's first-value
and translator last-value views can disagree. Mixed `Content-Length` /
`Transfer-Encoding`, invalid or contradictory lengths, and actual bodies
over the cap are rejected without trusting the declared length. Partial
BYOK credential headers refuse instead of silently selecting another
provider, and idempotency control characters refuse rather than becoming
opaque store keys.

PR #2819 was authored before the merged #2817 BYOK hardening. Integration
retains #2817's shared URL rule, credential redaction, redirect refusal and
sanitized transport errors; the older transport and validator variants are
not reapplied. Historical receipts remain unchanged. The generated audit
receipt is `SYNTHETIC`, `research_only`, and makes no live-PnL claim. The
serve census moves from 46 to 47 and remains `partial`.

### Jobs audit maintenance (PR #2822)

The new `jobs_audit` battery pins the async job orchestration surface
end to end: `POST /harness/jobs` (+ `/harness/jobs/batch`), record GETs,
list/filter/paging, `DELETE` cancel, SSE events, `/harness/jobs/{id}/receipt`
and the `/harness/runs` synchronous twin. It exercises the
`queued→running→terminal` lifecycle, the submission validator's refusal
envelopes, the executor contract, `Idempotency-Key` dedup under
sequential and parallel submits, the drain/`max_inflight` gate ordering
against auth and scope, HMAC-signed terminal webhooks (fire-once,
retry cap, 4xx definitiveness, no resurrection), `--state-dir`
journaled restart durability, per-item batch isolation, per-key
metering, and the `HarnessClient` wire legs over the in-process
transport.

Building the lane surfaced three real defects, fixed on the same PR:

* `_JobStore` had no atomic start claim — the worker's lock-free
  `status == "cancelled"` check-then-set let a cancel landing in the
  gap resurrect to `running→succeeded` (and journal the lie). The
  store now owns `start(job_id)` under its lock; `_exec` claims the
  transition through it. The store also gained `delete(job_id)`
  tombstoning for refused submissions.
* `job_store.put` ran *after* `jobs_executor.submit` — the worker's
  first `mark` could journal a transition ahead of the record itself
  (a submitted job briefly 404'd; a refused `submit` left a ghost
  journal entry). The record now registers before hand-off and a
  refused submit journals a `{"deleted": job_id}` tombstone.
* `/harness/runs` let a runner fault escape as a bare plain-text 500
  while the jobs twin captured the same fault honestly into
  `job.error` — the route now folds unhandled executor faults into the
  `{detail, code}` envelope.

The battery also pins measured (pre-existing, non-defect) semantics:
a nonzero `exit_code` completes `succeeded` with `result.ok=False`
(`status` tracks execution, `ok` the command's verdict); `?wait_s` on
the job GET is inert (SSE and drain's `wait_s` are the wait channels);
idempotency replays mark `replayed` in the body with no replay header;
unsigned webhooks send no signature or timestamp headers;
`callback_*` bookkeeping lands after the terminal flip; the
`fx1_job_record.v1` receipt projects the record with digested
stdout/stderr; `uses` bills every authorized call including
gate-refused ones, and the rpm refusal code is `rate_limited`.

The generated audit receipt is `SYNTHETIC`, `research_only`, and makes
no live-PnL claim. The serve census moves from 47 to 48 and remains
`partial`.


### Ops-surface audit maintenance (PR #2868)

The new `ops_audit` battery (141 probes) pins the operations surface a
load balancer, orchestrator, and operator dashboard actually poll —
`/health`, `/ready`, `/harness/version`, `/harness/capabilities`,
`/harness/backends`, `/harness/backends/{name}/probe`,
`/harness/gate/check`, `/harness/score`, `/metrics`,
`/harness/drain`, `/docs`, `/redoc`, `/openapi.json`, and `/` — using
SYNTHETIC stub backends and an in-process ASGI client. Measured
contracts: liveness is the single public path and reports
presence-of-credentials flags off the live env; readiness is keyed,
backend-blind, reflects the real inflight gauge, and 503s `draining`
once the latch is set; version publishes `fx1.__version__` /
`API_VERSION` from one source of truth consistent with health,
capabilities, and the spec's info block; the metrics scrape observes
the pre-scrape state, partitions `by_status`, keeps `uptime_s` on the
monotonic clock, fabricates no zero backend series, and negotiates the
Prometheus exposition only on `?format=prom` or a text/plain Accept;
backend status reports the breaker's real circuit state and caches
each deep-probe verdict; the probe returns `ok:false` +
`error_class` verdicts instead of HTTP faults, stays slot-gated under
drain, and bypasses the breaker without feeding it; the advisory
preflights answer verdicts, resolve no backend, and survive drain;
the spec is deterministic, declares the middleware's stamped headers
(rate-limit headers only when the limiter exists), mirrors the route
table's methods exactly except the undocumented `/v1/{path:path}`
catch-all, and is credential-gated like the rest of the surface; `/`
is a clean enveloped 404; every ops refusal class (401/403/404/405/
422/429/503) lands in the `{detail, code}` envelope with no bare 5xx;
`X-Request-ID` echoes well-formed ids and mints on absent/malformed;
and `openai-version` plus the Anthropic dialect headers stay scoped to
`/v1`.

Two defects were found and fixed in `api.py`:

* `_is_anthropic_surface` honored `anthropic-version` on *any* path,
  so ops answers carried Anthropic-dialect headers (`request-id`,
  `x-should-retry`, and — for rpm-windowed managed keys —
  `anthropic-ratelimit-requests-*`), contradicting the documented
  "/v1/messages tree plus /v1/* under the header" contract. The
  surface check is now scoped to `is_openai_path`; probes pin both the
  ops absence and the /v1 presence.
* `POST /harness/backends/{name}/probe` returned its resolve-stage
  verdicts (`backend_unavailable`) through an early `_verdict` that
  skipped `record_complete`, so the documented `probe:<name>` verdict
  series silently missed resolver failures — a monitoring scrape
  watching only the metric could never see them. The 503 verdict
  branch now records its own `probe:<name>` error.

The battery also pins measured (pre-existing, non-defect) semantics:
`local_fx1` probes 422 without `checkpoint_dir` (and the kwarg is
refused on non-local backends), unknown backend names 422 on the
path-literal, `?format=xml` on `/metrics` 422s, and `wait_s` bounds on
drain are 0..600.

The generated audit receipt is `SYNTHETIC`, `research_only`, and makes
no live-PnL claim. The serve census moves from 53 to 54 and remains
`partial`.


### SDK concurrency audit maintenance 

The new `sdk_concurrency_audit` battery pins the shared-state seams of
the in-process `Fx1Harness` SDK under threaded contention: the
`_CompletionLog` record store (bounded capacity, `dropped` accounting),
the `_last_response_headers` publication, the `_bg_cancel` background
response registry, and the `_files` upload map — plus a mixed storm of
completers, uploaders, background submitters and readers running
together.

The battery pins 69 measured contracts: parallel writes never lose a
record or tear a `CompletionRecord`; the bounded log evicts oldest and
counts `dropped` honestly; readers see coherent snapshots mid-flood;
each `last_response_headers` read returns a fresh dict isolated from
caller mutation; cid/rid pairing stays consecutive under serial and
flood scheduling (proven by tagged-uuid minting); background responses
always reach a terminal state with unique rids and a drained registry;
cancel lands once regardless of how many threads race it, and persists
when the response later completes; the files map caps at 256 with
oldest-evicted under a parallel burst.

Deterministic scheduling via per-thread line tracing parks a writer
inside `_record_call` at the exact line that used to mutate the
published header dict, exposing the store-then-mutate tear window —
a real defect fixed on this PR: `sdk.py` now builds the headers dict
completely (including `x-fx1-completion-id`) and publishes it in a
single store, so no reader can observe a header set missing its
completion id.

Not verified (documented in `coverage.not_verified`): cross-process
contention (the seams are in-process by design), real wire-level
ordering (transports are spy backends), and journal-replay recovery
(the log's `--state-dir` persistence lane).

The generated audit receipt is `SYNTHETIC`, `research_only`, and makes
no live-PnL claim. The serve census moves from 58 to 59 and remains
`partial`.


### Client retry audit maintenance 

The new `client_retry_audit` battery pins the retry mechanics inside
`HarnessClient._request` past the error-map surface `client_audit`
already covered: exact attempt counts (max_retries=N means N retries
past the initial call, only on retryable calls — unkeyed writes never
retry, keyed writes and `retry_writes` do, every attempt hits the same
path), the sleep schedule as arithmetic (the doubling backoff feeds
transport-fault retries while a declared Retry-After — slept verbatim
under the cap — feeds refusal retries, and the backoff still doubles
across a status sleep for the next fault), Retry-After parsing edges
(seconds-only `float()` with a `max(0, …)` floor: whitespace, signs,
fractions and scientific forms parse; HTTP-date/junk/empty fall to
not-retryable; `inf` breaks over budget; `nan` floors to 0.0 and
retries immediately), the lifecycle of the shared slots
(`_last_response_headers` clears only on transport-class escapes —
a mapped 503/404 keeps its headers while a mapped 429/500 loses
them — and `_last_api_version` survives faults untouched), the
circuit breaker's error-class split (transport-class escapes trip,
mapped refusals don't — an honestly-refusing 503 stays reachable),
threshold-1 opening, half-open probe close/fault-reopen, strict-<
`open_until` boundary, fail-fast with no sleep and no transport call,
and consecutive-fault counting across a mid-window success; plus
`timeout_s` and the Idempotency-Key (and API key) reaching every
attempt verbatim, and parallel callers getting independent retry
schedules over the shared circuit counter.

Maintenance review found four production defects the first pass
pinned as semantics; all repaired and re-pinned:

- `Retry-After` non-finite (`nan`/`inf`) or negative values were floored
  to `0.0` — an untrusted header could collapse backoff into an
  immediate retry or wedge the sleeper. `_retry_after_s` now treats
  them as malformed: the status is not retryable and maps normally.
- Transport-fault sleeps ignored `max_retry_wait_s`, doubling without
  bound. They now share the cap (`min(backoff, max_retry_wait_s)`);
  sleeps stay deterministic by design — callers wanting spread inject
  a jittering `sleep=` hook.
- The half-open window admitted every caller: a `_cb_lock`-guarded
  single-probe flag now lets exactly one racer dial while the rest
  fail fast; the probe releases on any completed response, re-opens
  on a transport fault, closes via `_cb_reset` on success.
- `retry_writes=True` retried unkeyed POSTs — an ambiguous fault could
  replay a landed write. Writes now retry only when keyed
  (`Idempotency-Key`); keyed calls already mark idempotent, so the
  flag only ever widens keyed traffic.

The generated audit receipt is `SYNTHETIC`, `research_only`, and makes
no live-PnL claim. The serve census moves from 59 to 60 and remains
`partial`.

### Webhook delivery audit maintenance 

The new `webhook_delivery_audit` battery runs the outbound webhook
dispatcher (`sign_webhook`/`verify_webhook`/`deliver_signed`) over a
real loopback `http.server` recorder with an event-clock harness —
`time.sleep` patched to a log so backoff arithmetic is measured, never
slept — plus direct probes of the SSRF seams
(`_resolved_addresses`, `_is_public_unicast`, the pinned connections).

It pins the signature contract end to end: `sha256=`-prefixed hex
over `<ts>.<raw body>` byte-exact (reserialized JSON fails), the
`sha256=` prefix required, ASCII-only timestamp and signature, an
inclusive `abs(ref - ts) <= tolerance` freshness window in both
directions, negative tolerance disabling freshness, non-finite
tolerance/timestamps refused, and a verify path that never raises on
garbage. The callback-URL grammar is pinned scheme/host-validating
with every literal-private form refused — loopback, RFC-1918, CGNAT,
link-local, multicast, and IPv6 ULA/link-local — and
`FX1_WEBHOOK_ALLOW_PRIVATE_NETWORKS` is checked live per call
(`1`/`true`/`yes`), scoped per leg so the opt-in never leaks.

The dispatcher semantics are pinned as measured: per-attempt fresh
timestamps and signatures; `backoff_s << (attempt - 1)` sleeps;
`4xx` definitive with no retry; `5xx` exhausting `max_attempts` then
returning `(False, err, attempts)`; a connect/HTTP fault walking the
validated address list while a `5xx` response short-circuits it into
the outer retry; per-attempt `timeout_s` arming a slow endpoint into a
retryable fault on loopback; unsigned delivery sending no signature or timestamp
headers; `Content-Type`/`Content-Length` and the full path+query on
the wire; `max_attempts <= 1` single-shot; invalid URLs refused with
`(False, err, 0)` and zero network attempts; and a private literal
target refused with zero attempts even when the resolver could
resolve it.

Zero production defects — every held semantic was already correct;
the battery's only corrections were its own measurement seams
(server delay moved off `time.sleep` so a patched backoff clock
cannot speed the server, hits-keyed timestamp capture so `time.time`
stays safe for the server's Date header).

Maintenance-review repairs (2026-10-06): the battery now pins the
exact 111-probe key set (`_EXPECTED_PROBES` — a dropped or renamed
probe fails `audit()` loudly, and `bench()` refuses to seal an
all-True subset); `_audit_context` saves, clears, and restores
`FX1_WEBHOOK_ALLOW_PRIVATE_NETWORKS` verbatim so a caller's ambient
opt-in can no longer flip the refusal legs; the SSRF literal battery
gained RFC-1918 172.16/12 edges, CGNAT 100.64/10, and IPv6 ULA
fc00::/7–fd00::/8 literals plus resolution-level pins
(`resolved_17216/cgnat/ula_refused`); the misnamed `mixed` answer
list (duplicate publics) was split into a real
`mixed_public_private_refused` fail-closed probe and a
`resolved_17232_public_passes` boundary sanity; and the receipt was
resealed at the integrated source commit with the committed-vs-fresh
and revision-binds-source tests the ledger now requires.

Not verified (documented in `coverage.not_verified`): real DNS
resolution (the recorder is a literal loopback IP), HTTPS delivery
(the pinned-HTTPS leg is probed at the connection object, not a live
TLS socket), and the `fx1_job_record.v1` webhook path in the jobs
lane's own battery.

The generated audit receipt is `SYNTHETIC`, `research_only`, and makes
no live-PnL claim. The serve census moves from 60 to 61 and remains
`partial`.

### Webhook delivery audit maintenance (PR #2905)

The `webhookdel_audit` battery pins the HMAC-signed webhook *dispatcher*
end to end (the `webhook_audit` lane covered the registration surface):
the verdict table measured on a live loopback sink — 2xx delivers on
attempt 1, every 4xx including 429-with-`Retry-After` is definitive in a
single attempt, 5xx and 3xx retry to the `WEBHOOK_MAX_ATTEMPTS=3`
ceiling, and redirects are never followed (the `Location` target never
receives a request). The exponential backoff schedule is measured on the
wire (0.5s then 1.0s gaps), each attempt opens a fresh connection,
re-mints the signature pair, and re-resolves DNS; connection-refused,
unresolvable DNS, read timeout, and TLS-mismatch faults each classify
loudly into `callback_error` and stay bounded; a dead resolved address
fails over to its sibling inside the same attempt.

At the app surface: submit returns before the delivery verdict can
exist, the terminal status is GETable while the verdict is mid-flight,
a queued-cancel DELETE returns only after the `cancelled` delivery lands
(delivery runs on the request thread), and a retrying delivery holds its
inflight slot — a second submit is refused `503 over_capacity` until the
verdict lands. A single-worker executor delivers FIFO; a pool dispatches
concurrently; cancel-path deliveries run on request threads even with
every pool worker asleep. Every terminal surface fires exactly once and
never re-fires. Under `--state-dir` the post-verdict mark journals
`callback_status`/`callback_attempts` atomically, the signing secret
never touches disk, a restart restores the verdict without re-firing,
a recovered queued job fails closed as `failed` with zero attempts
(honest abandon — nothing can re-sign), and idempotency mappings
survive. The drain latch refuses new work `503 draining` without
freezing admitted work, and lifespan shutdown flips queued jobs to
`cancelled` and fires the webhook exactly once. `callback_*` verdicts
surface honestly on every record GET, and every refusal arrives in the
path's own error grammar.

No defects found — the contract held on all 75 probes.

The generated audit receipt is `SYNTHETIC`, `research_only`, and makes
no live-PnL claim. The serve census moves from 61 to 62 and remains
`partial`.


### Compat audit (dialect-translation battery)

The new `compat_audit` pins 126 translation invariants over
`openai_compat` and `anthropic_compat` — the dialect boundary every
request and every answer crosses. Probes are pure-function calls: no
transport, no server, deterministic by construction.

Coverage: request-model accept/refuse pairs (bounds, `max_tokens` vs
`max_completion_tokens` disagreement, `tool_choice` requiring
`tools`); `openai_to_kwargs` / `response_to_kwargs` /
`embeddings_to_kwargs` link resolution in the documented order
(`fx1.backend` > header > model > byok-headers > `hosted_k3`) with
fail-closed 404s on unregistered models and refused half-formed BYOK
credentials; envelope builders (`chatcmpl-`/`cmpl-`/`resp_`/`msg_`
derivations, n-fanout, null-content tool-call turns, legacy echo and
usage sums); stream chunk grammar (role-first deltas, ~64-char
whitespace pieces, per-index grouped `n`, terminal usage frames);
Responses input folding (`function_call` → `tool_calls`,
`function_call_output` → tool turns, loud refusals on empty
`call_id`/unknown item types); cursor pagination that fails closed;
the `OpenAIEnvelopeStore` put/get/list/evict/delete contract with
re-put refresh and journaled ops; Anthropic field translation
(system→system, tool_use↔tool_calls, tool_result→tool, `store=False`
forced), the `finish_reason`→`stop_reason` map, bad tool args held
in-band as `_raw`, the typed SSE lifecycle, and the batch/file/model
request shapes with `results_url` gated on `ended`.

The generated audit receipt is `SYNTHETIC`, `research_only`, and makes
no live-PnL claim. The serve census moves from 62 to 63 and remains
`partial`.


### Backends audit (transport/policy battery)

The new `backends_audit` pins 198 contracts over `serve.backends` —
the module every other audit patched at its seams but none probed.
Probes are pure functions plus the documented `_openai_urlopen` seam:
no sockets, no real child processes (a faked `Popen` records the spawn
template), deterministic by construction.

Coverage: `SamplingParams.body_fields` (deterministic `temperature`
default, declared-fields-only, `stop` tuple→list, frozen dataclass);
harness-side stop truncation including mid-chunk cuts and the
empty-piece rule; URL normalization (`_chat_completions_url`,
`_openai_sibling_url`) and `_env_float` precedence/refusals; the BYOK
policy chain — `byok_base_url_problem` (userinfo/query/fragment/port/
host refusals; reasons never echo the URL), `_byok_address_allowed`
(private opt-in never admits link-local/multicast/unspecified),
`_byok_resolved_addresses` fail-closed on prohibited or mixed answer
sets with v4-mapped normalization; transport (`_RefuseRedirects`
unconditional refusal, `_openai_urlopen` normal + DNS-pinned legs,
non-2xx→HTTPError with response and connection closed, per-request
policy stamps, `FX1_BYOK_ALLOW_PRIVATE_NETWORKS` truthy variants);
`_extract_usage`/`_extract_token_count` rejecting bools, strings,
non-finite floats; `_UsageTracker` concurrent accumulation; the
fail-closed `tool_calls[]`/embeddings `data[]` shape validators; wire
helpers (`_openai_chat_complete`, `..._tools`, `..._stream`,
`_openai_tokenize_count`, `_openai_embeddings_complete`) — request
shapes, labeled `RuntimeError` envelopes for HTTP/URL/transport/
decode faults, `TokenCountUnavailableError` on refused tokenize
routes (never an estimate), stream frame grammar (`[DONE]`,
role frames, usage-only frames, `usage_out` capture,
`stream_options` never sent); the three backend classes (key/env
precedence, missing-config `BackendNotConfiguredError`, ship-gate +
signature enforcement, `_ensure_engine` spawn template expansion +
`FX1_CHECKPOINT_DIR`, `close()` semantics); `get_backend` dispatch;
and the capability protocols (`InferenceBackend` deliberately not
runtime-checkable — capability checks belong on the optional
channels only).

The generated audit receipt is `SYNTHETIC`, `research_only`, and makes
no live-PnL claim. The serve census moves from 63 to 64 and remains
`partial`.


### Finetune audit (job-store state machine)

The new `finetune_audit` pins 112 contracts over `serve.finetune` —
the `/v1/fine_tuning/jobs` surface. In-process only: tmpdir journals,
no trainers, no network.

Coverage: `validate_chat_jsonl` admission (utf-8, per-line JSON,
`{"messages": [...]}` shape, role whitelist, non-empty string
content, lineno in the error, all-blank files refused); the request
shapes (`model`/`training_file` bounds, `suffix` charset, `seed>=0`,
`method="supervised"` only, metadata cap, `extra=forbid` on every
envelope, `FTHyperparameters` ranges); the `FTJobStore` state machine —
newest-first listing with exclusive `after` cursors, `get`/`lookup_idem`
MRU refresh, bounded eviction dropping job+key+cards, the `ft:` model
registry (sorted listing, register-only-while-alive, unregister
tombstones, `checkpoint_for`, checkpoints oldest-first with derived
`ftckpt-` ids), the event feed (oldest-first, 256 cap pops oldest),
cancel verdicts (queued/paused flip terminal now, running flag-only),
pause verdicts + `paused_from` bookkeeping + idempotent re-pause,
resume restoring the captured status, `cancel_pending` drain
semantics (queued→cancelled, running→flagged), per-key async claim
locks, and journaled replay (terminal jobs return as-was, in-flight
recover as failed with a restart-explaining error, `ft:` keys
resolve, model cards rebuild minus evicted-producer refs,
`callback_secret` never touches disk).

The generated audit receipt is `SYNTHETIC`, `research_only`, and makes
no live-PnL claim. The serve census moves from 64 to 65 and remains
`partial`.


## Ops-receipt audit (sealed-export battery)

`ops_receipt` is the disclosure boundary every other audit leans on —
it turns completion-log rows, job-ledger rows, and bench results into
sealed documents (`receipt_sha256` over canonical JSON) verifiable by
`verify_receipt_payload` / `POST /receipts/verify`. `opsreceipt_audit`
pins 30+ contracts of the sealed-export shape in-process:

- *Envelope* — every doc carries `kind`/`schema`/`git_revision`,
  `data_label="OPS"`, `research_only`, `live_pnl_claim: False`; the seal
  is 64-hex canonical-JSON, re-verifies, is deterministic per record,
  leaves the input unmutated, and breaks on a one-byte tamper.
- *bench_receipt* — `fx1_bench_result.v1`; record verbatim.
- *completion_record_receipt* — `fx1_completion_record.v1`; verbatim
  passthrough, idempotent re-export.
- *run_result_receipt* — `fx1_run_result.v1`; only declared passthrough
  keys survive, `stdout`/`stderr` ship as `*_sha256` digests, `ok`
  derives from `exit_code == 0` only when unreported.
- *job_record_receipt* — `fx1_job_record.v1`; ledger keys pass through,
  `callback_url` becomes `callback_url_sha256`, `callback_secret` never
  appears anywhere in the export, terminal `result` embeds
  pre-digested.

The generated audit receipt is `SYNTHETIC`, `research_only`, and makes
no live-PnL claim. The serve census moves from 65 to 66 and remains
`partial`.


## Scoring audit (proper-score primitives)

The `score_*` operations are the harness's honesty-critical math — every
sealed research claim bottoms out in pinball, interval, Brier/log-loss,
or empirical-CRPS scoring, so a silent sign or denominator change turns
every downstream receipt into a lie. `scoring_audit` pins 60+ contracts
against hand-computed fixtures and an independent O(n²) CRPS reference:

- *Descriptor contract* — `skills.*` ids, `kind="skill"`, module-local
  schemas/handlers, host-facing `describe()` shape.
- *score_quantiles* — exact level-weighted pinball orientation,
  per-level means + unweighted cross-level mean, crossings rejected
  (ties allowed), strictly increasing levels, shape/finite/`extra=forbid`
  enforcement.
- *score_intervals* — additive width + `2/(1-c)` miss penalties,
  inclusive endpoint coverage, inverted/unequal/out-of-range refused.
- *score_binary_forecasts* — `(p-y)²` Brier on raw probabilities,
  nats log loss, impossible endpoints → `positive_infinity` + null
  unless a declared clip rescues (Brier still uses the raw p).
- *score_empirical_crps* — exact empirical-CDF CRPS: degenerate
  perfect forecasts score 0, singletons `|s-y|`, sorted-gap integration
  matches `E|X-y| − E|X−X'|/2` on sorted/unsorted/tied/unequal-ensemble
  fixtures.

The generated audit receipt is `SYNTHETIC`, `research_only`, and makes
no live-PnL claim. The operations census moves from `pending` to
`partial` with 44 modules pinned.


## Stats audit (feature-math battery)

The `features.*` operations are the descriptive transform layer —
every z-score, EWMA forecast, trend slope, and entropy figure a skill or
eval consumes is produced here under causal trailing-window semantics.
`stats_audit` pins ~110 contracts against hand-computed fixtures:

- *simple_returns / drawdown_path* — lagged fractional changes with
  null warmups; running high-water marks, nonpositive drawdowns,
  durations resetting on reattained peaks; nonpositive prices refused.
- *ewma_variance* — strictly pre-observation forecasts under the
  declared zero-mean recursion; `next_variance` continues the chain.
- *rolling_zscore / rolling_mad / rolling_rank* — population z-scores,
  median + raw MAD + unscaled robust scores (`warmup`/`zero_mad`/
  `numeric_overflow` statuses), average-rank ties, unique-max = 1.
- *rolling_autocorrelation* — separately centered Pearson lagged pairs,
  `window − lag ≥ minimum_pairs` admission, clamped to [-1, 1].
- *rolling_linear_trend* — OLS on the centered index: slope, midpoint
  intercept, `sqrt(RSS/(w−2))` scale, R² null on constant windows.
- *bipower_variation* — `RV = Σr²`, `BV = (π/2)·factor·Σ|rᵢ₋₁||rᵢ|`,
  `n/(n−1)` correction named; single return → `insufficient_pairs`.
- *permutation_entropy* — delayed ordinal patterns, stable/drop/reject
  tie policies, `H/log(m!)` normalization, full status ladder, bounded
  pattern reporting.
- *spectral_summary* — boxcar periodogram with interior doubling,
  DC-excluded peak/entropy, constant input zeroing positive power,
  Parseval invariant verified against direct time-domain energy.
- *time_weighted_mean* — left-continuous latest-observable-event
  integration; activation gated on event AND availability clocks; stale
  revisions hold zero duration; missing initial coverage fails closed;
  naive/unix clocks and duplicate event times refused.

The generated audit receipt is `SYNTHETIC`, `research_only`, and makes
no live-PnL claim. The operations census moves from `pending` to
`partial` with 44 modules pinned.


## Data-quality audit (`skills.audit_*` operation battery)

`fx1.operations.dataqual_audit` exercises all twelve `audit_*` data-quality
sentinels in-process: `audit_bar_integrity` (per-cell failure taxonomy,
inverted-range envelope suppression, bounded diagnostics),
`audit_duplicate_keys` (JSON scalar semantics — `1 == 1.0`, `True != 1`,
`"1" != 1` — under reject/equal/distinct null policies),
`audit_missingness` (absent/null/optional-empty counting, supplied-order
runs, unassessed empty states), `audit_monotonic_sequences` (never
reorders input; group-wide duplicate clocks incl. timezone-equivalent
instants; singleton unassessed), `audit_cross_field_contracts`
(type-aware equality, bool ≠ number, ordering restricted to numbers,
missing-before-null precedence), `audit_missingness_association` (2×2
contingency, Jaccard, null phi on constants), `audit_panel_gaps`
(per-security anchored elapsed-time grid), `audit_point_in_time`
(availability always required; completed-event/ingestion opt-in),
`audit_referential_integrity` (parent/child verdicts under numeric
policy), `audit_revision_conflicts` (conflicting fields vs exact
duplicates), `audit_schema_drift` (added/removed/type/optionality/
nullability), `audit_source_coverage` (latest-event selection,
exclusions, staleness clocks). ~95 literal-bool probes seal into a
`dataqual_audit.v1` receipt.


## I/O operations audit (`plugins.*` readers + `skills.*` bitemporal joins)

`fx1.operations.ioops_audit` exercises the file-backed operation surface
inside a real temp workspace plus the bitemporal selection machinery:
`read_csv`/`read_jsonl`/`read_toml` (exact-string cells, strict JSON with
duplicate-key and NaN/Infinity refusal, temporal classification,
fingerprinted sources, honest paging), `inspect_numpy_array`/
`inspect_parquet`/`inspect_zip` (header/footer/central-directory metadata
without materializing payloads; object arrays and trailing bytes
refused), `verify_file_hash` (byte-exact SHA-256 + optional size check),
workspace containment probes (escape, wrong suffix, missing file,
symlink, byte-cap all fail closed), `join_asof_observations`
(activation at `max(event, available)`, latest-vintage winner with
declared tie policy, same-clock payload conflicts refused at
validation), `select_asof_revisions` (latest observable vintage,
opt-in future-event exclusion), `select_universe_membership`
(latest-availability revision, effective intervals, expired exclusions
revealing inclusions, same-effective ambiguity), and
`summarize_ingestion_latency` (signed lags keep clock anomalies
visible). ~85 literal-bool probes seal into an `ioops_audit.v1`
receipt.


## Operations framework audit (`base.py` + `registry.py`)

`fx1.operations.opsframe_audit` pins the machinery every operation
stands on: descriptor id/kind/namespace agreement, module-local
input/output schemas, the `fx1.operation-result/v1` invoke envelope
with input/output SHA-256 and honesty flags, `extra="forbid"` +
`allow_inf_nan=False` on both model classes, the frozen abspath'd
`OperationContext`, `resolve_file` spelling checks vs the
`open_binary` descriptor-relative `O_NOFOLLOW` walk (internal symlinks
resolve but refuse at open; escape symlinks refuse at resolve),
`WorkspaceReader` bounded reads with EOF-gated `source_sha256`,
canonical JSON (sorted, compact, raw UTF-8, NaN refused, byte budget
enforced during encoding), `\\?\`/`\\?\UNC\` normalization, and the
reviewed registry: exactly 40 implementations, every on-disk
non-battery module registered, bounded discovery with kind/query
filters, and schema-gated dispatch for the three literal function
tools. ~80 literal-bool probes seal into an `opsframe_audit.v1`
receipt.


## Extensions audit (`extensions/` machinery + `capabilities.py` seed ledger)

`fx1.extensions.extensions_audit` pins the generated-card machinery:
naming rules (owner regex, dash→underscore, kind dispatch failing
closed), `ExtensionModule` post-init identity checks (declared module
path must equal `module_path(kind, owner)`, nonempty references),
bounded `records()` paging with foreign-record refusal, per-kind
subclass invariants (skill→command, plugin→source adapter,
feature→catalog metadata), the reviewed 45-entry feature catalog
(`synthetic_only` iff family `synthetic_oracle`, every name backed by
an on-disk module file), the capabilities seed ledger
(`owner_references` progressions per owner, `resolve_seed_id`
round-trip for every owner's first and last seed, total residue
layout: skills ≡0, plugins ≡1, features ≡2 mod 3), and the registry's
`list_extensions`/`get_extension`/`extension_manifest` across all 86
generated modules — each of which must export a `MODULE` matching its
registered (kind, owner, path) identity and verify its own seed
references. ~90 literal-bool probes seal into an
`extensions_audit.v1` receipt.


## Selftest audit (`selftest.py`)

`fx1.selftest_audit` pins the deploy-gate machinery beneath the golden
walk: `SelftestReport.ok` requires a nonempty all-true check set and
`as_dict` preserves name/ok/detail; `_run` never escapes (expect-match
vs truthiness, exceptions captured as `{Type}: {msg}` details);
`_wait_job` returns terminal statuses immediately and reports
`{"status": "timeout"}` past the deadline instead of hanging;
`_server_down` distinguishes a live listener from a closed port. The
battery runs remote mode twice — against an unreachable target (every
check recorded with an honest failure detail, never raised) and
against a real in-process app (health, version parity, commands,
score advisory, gate preflight, and the missing-key auth-gate check
all green) — plus the full local golden path, once bare and once with
a `state_dir` so the `restart_recovers_job` leg runs. The environment
contract is pinned end-to-end: every env var the selftest touches
(`FX1_API_KEY`, `FX1_BYOK_*`, `MOONSHOT_API_KEY`,
`FX1_BYOK_ALLOW_PRIVATE_NETWORKS`) and the `fx1.serve.api` logger
level are restored after the run — the audit's sentinel probes caught
two restore gaps (`FX1_BYOK_ALLOW_PRIVATE_NETWORKS` and
`MOONSHOT_API_KEY`) which the lane fixes. ~45 literal-bool probes seal
into a `selftest_audit.v1` receipt; `selftest.py` moves to `audited`.


## Capability seed ledger audit (`capabilities.py`)

`fx1.capabilities_audit` proves the compact seed table *is* the
declaration ledger: every shard under
`scripts/generated_capability_declarations/{features,skills,plugins}/`
is expanded line-by-line and its `_register(seed_id)` ids must
reproduce the owner's `(first, stride, count)` progression exactly
(hyphenated owners map to underscored shard/wrapper names); each
shard's header must import a resolvable `fx1.extensions.<kind>.<owner>`
`MODULE`. Resolution contract pinned: `owner_references` exact
arithmetic per owner, `ValueError`/`KeyError` error split,
`resolve_seed_id` bounds at `0..1_000_000`, the per-kind owner-field
echo (`feature`/`command`/`source`), and the full partition — the
three tables tile every in-range id exactly once with zero intra-kind
collisions and zero cross-kind shadowing, so resolve is a total
injective owner lookup. ~115 literal-bool probes seal into a
`capabilities_audit.v1` receipt; `capabilities.py` moves to `audited`,
completing the fx1 root-module surface.

## Cited completion + release signing audit (`chat.py`, `signing.py`)

`fx1.serve.chat_signing_audit` pins the two contracts the serving layer
leans on. `chat.py`: `sampling.stop` truncates before the gate (a
forbidden tail cut by the stop list passes; one left in fails),
`validate_fx1_output` raises `Fx1HonestyError` inside the wrapper so
the footer can never be reached ungated, and provenance footers carry
16-char receipt prefixes plus the `verify-research` hint. The tool
half: a backend without `complete_with_tools` fails closed naming the
backend class, the gate reads `content` only (forbidden text inside
`tool_calls[].function.arguments` is machine JSON, not a claim),
`logprobs` ride through verbatim (`None` under silence), and an
untouched completion returns the identical object while a gated one
preserves tool_calls/finish_reason/logprobs. `signing.py`: manifest
covers the complete regular-file inventory minus the two root
metadata files (nested `release.sig` under a subdir is a real
artifact), signing needs `FX1_SIGNING_KEY` (never hardcoded),
verification authenticates before hashing and fails closed on
tampered bytes, added/removed files, forged manifest or signature,
corrupt manifest JSON, a manifest declaring traversal paths (never
opened — key comparison first), missing metadata, missing key
(RuntimeError), symlink artifacts and FIFOs. ~55 literal-bool probes
seal into a `chat_signing_audit.v1` receipt.

## Managed key store audit (`keys.py`)

`fx1.serve.keys_audit` pins the multi-tenant auth store the wire
batteries lean on. Mint: `fx1k_`+40-hex secrets shown once,
sha256-only storage, `key_id`/`prefix` display fingerprints, wire
views strip `sha256` and every `_`-counter, scope resolution
(defaults, additive `admin`, canonical `SCOPES` order, empty/unknown
refused), per-field validation, and the store cap. Authenticate:
uniform `None` for wrong/unknown/disabled/expired (no oracle),
refusal ordering budgets → scope → window so a refused call never
counts as a use nor burns a window slot, `rate_limited` carrying
honest `retry_after` + `key_id`, rolling-window recovery, token
budgets charged after served responses. Revoke/update/rotate:
tombstones not deletes, double-revoke refuses `key_revoked`, patches
keep counters while `clear` reverts only `CLEARABLE_KEY_FIELDS`,
rotation inherits declared policy verbatim (absolute `expires_at`
carried unless a fresh `ttl_s` rebases it), the predecessor's
tombstone and successor journal in one line, and `keys_cap` is
checked before the predecessor is touched. Durability: replay
restores records + counters, live rate windows stay process-local,
torn journals quarantine recovered keys while keeping auth required,
and a later clean mint is not re-quarantined. ~75 literal-bool probes
seal into a `keys_audit.v1` receipt.

## Receipt index audit (`receipt_store.py`)

`fx1.serve.receiptstore_audit` pins the lazy sha256→file index behind
`GET /receipts*`: admission is strict (regular `*.json` files whose
document carries a full-match 64-hex `receipt_sha256`; malformed,
non-dict, wrong-length, uppercase, symlinked, directory, binary, and
recursion-depth inputs never index and never raise), duplicate
digests resolve to the lexicographically first filename, items list in
name order, and staleness is honest — additions, removals, renames,
same-name replacements, corrupt-then-repaired files, and a missing or
unreadable root all flip lookups correctly while unavailable roots
clear the cache instead of serving stale results. ~30 literal-bool
probes seal into a `receiptstore_audit.v1` receipt.


## API surface audit (`api.py` route wiring)

`fx1.serve.apisurface_audit` pins what requests meet before any handler
runs: the route inventory itself (required families present, ≥60
routes, no double-bound (method, path) pairs, the shared scope map —
`/harness/keys*` and `/harness/drain` are `admin`, safe methods `read`,
rest `write`), the public surface (`/health` exactly, in both dev and
key-armed mode), and the auth wiring — authentication precedes routing
(unknown paths 401 unauthenticated, 404 authenticated),
`Authorization: Bearer` honored only under `/v1`, `X-API-Key`
everywhere, dev mode trusting loopback only (a non-loopback client is
403, not silently trusted), and minted keys laddering
read < write < admin with 403 `insufficient_scope` in the path's own
grammar (OpenAI `error` object under `/v1`, `{"detail","code"}`
elsewhere). Error dialects are per-family (v1 catch-all 404
`Invalid URL`, harness 405 `method_not_allowed`), response headers
always carry `X-Request-ID`/`nosniff`/`X-Fx1-Api-Version`, the drain
latch is one-way while liveness answers, and CORS defaults closed with
wildcard refused and explicit-origin preflight unauthenticated. ~77
literal-bool probes seal into an `apisurface_audit.v1` receipt.


### Audit-contract meta battery 

`src/fx1/audits_audit.py` audits the audit suite itself: 922 probes over
the 72 discovered `*_audit.py` modules plus suite-level conventions. For
each battery it pins that the module imports, exports an audit/bench
callable pair (resolving the documented stem-prefix exceptions —
`dip_run_audit`, `eval_core_audit`, `forecast_*`, `train_receipt_audit`,
`run_audit`, `fx1_tail_audit` — via a unique-callable fallback), executes
inside a shared 180s budget, restores the process environment and leaves
no leaked non-daemon threads behind. Its returned bench pins the sealed
receipt contract: required keys, `SYNTHETIC`/`research_only`/
`live_pnl_claim=False` honesty fields, `*.v1` schema, non-empty
`claim.results` (probe-keyed dict or verdict-row list), boolean
`claim.ok`, and a 64-hex `receipt_sha256` that the verifier accepts.

Batteries claiming literal-bool `claim.results` get `bools_literal` and
`ok_consistent` probes (`ok == all(results)`); the 16 older batteries
whose results deliberately carry measured values are pinned in
`_MEASURED_RESULTS` — a module may only appear there while its results
actually are non-bool, so the allowlist flags itself stale instead of
growing silently. `_DEFERRED` carries `fx1.serve.api_audit` whose
offline-DNS/callback repair rides an open lane; the set is pinned to
exactly that one entry. Convention probes require every battery to be
imported by at least one `test_*.py` (dotted or `from <parent> import
<stem>` forms), require every fx1-root `*.py` to hold a census `modules`
entry, and recount each directory census `n_modules` recursively.

The first full run exposed eight contract-level defects in existing
batteries, all repaired in this PR rather than waived: five batteries
leaked `fx1-job` thread-pool workers (`create_app`'s
`ThreadPoolExecutor` is only shut down by the lifespan, which bare
`TestClient(app)` never enters) — `anthropic_sdk_audit`,
`oai_sdk_audit`, `stream_audit`, `webhook_audit` and
`eval_lifecycle_audit` now register created apps and drain
`app.state.jobs_executor` in teardown; `ds_audit` left its
`DS_AUDIT_SENTINEL` marker set; `calibration_audit` inherited
BLAS/numexpr `KMP_*`/`_RJEM_MALLOC_CONF` mutations and `tail_audit`
inherited mlflow `_MLFLOW_TELEMETRY_SESSION_ID`/`MLFLOW_TRACKING_URI`
mutations — both now snapshot and restore `os.environ`. These are
teardown defects in the batteries, not in `api.py`'s executor contract.

The receipt is `SYNTHETIC`, `research_only`, makes no live-PnL claim, and
a running battery in the caller's process necessarily shares ambient
state — budget/thread checks bound, not eliminate, that coupling.
