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
