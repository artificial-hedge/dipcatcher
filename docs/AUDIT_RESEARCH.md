# Research-lane correctness audit — `src/quant_fund/research/`

Scope: all 29 top-level modules of `src/quant_fund/research/` (~14k LOC),
audited against the repo contract — fail-closed causality and honesty.
Look-ahead, silent NaN propagation, fabrications, forged/mutable receipts,
and non-contract error handling are defects. Protocol order: (a) receipt
writers and verifiers, (b) signal→return causality (walk-forward cursors,
per-date cross-sections, expanding windows), (c) forced SYNTHETIC stamps and
`FORBIDDEN_RESEARCH_METRIC_KEYS` gating, (d) fail-closed empty/degenerate/
missing-input handling. Verdicts: `CLEAN`, `FIXED` (defect repaired, with a
regression test), `DEFERRED` (real finding, not repaired here — blast radius
exceeds a minimal contract-preserving fix).

Regression tests for the fixes live in
`tests/unit/research/test_research_audit.py`.

## Bugs fixed

1. `benches/ranking.py::oos_rank_scores` — **FIXED**. When the panel has too
   few dates for the configured validation scheme, `walk_forward` returns no
   folds and the function fell back to a plain chronological holdout —
   `times[:cut]` fit, `times[cut:]` scored — with no boundary purge or
   embargo. The last `horizon_bars` train decisions carry labels that reach
   inside the holdout, so the "out-of-sample" IC/decile statistics were
   contaminated by realized test-period returns — the exact leak the fold
   path purges via `purge_mask` + `embargo_bars`. The fallback now applies
   the identical semantics (`purge_mask` against `[times[cut], times[-1]]`,
   then `idx[t] + embargo_bars < cut`); when purging leaves no train dates it
   fails closed to all-NaN scores instead of fitting on overlapping labels.
   Regression: `test_oos_rank_scores_small_panel_fallback_still_purges`,
   `test_oos_rank_scores_fallback_fails_closed_when_purge_empties_train`.
2. `sota_protocol.py::run_sota_protocol` — **FIXED**. `sota_receipt.json`
   was written by a bare `dest.write_text(json.dumps(receipt, default=str))`:
   non-atomic (a crash mid-write leaves a partial receipt on disk), unsealed
   (no `receipt_sha256` — `verify_receipt_file` reported
   `receipt_sha256_missing_or_invalid`), and type-corrupting (`default=str`
   silently stringifies foreign objects, changing the sealed content). The
   receipt is now canonicalized, self-sealed with `receipt_sha256` under the
   `canonical_json_bytes` convention, and published via
   `NamedTemporaryFile` + fsync + `os.replace` (the `agent.py`/"latest file"
   atomic-publish pattern). `verify_receipt_file(sota_receipt.json)` now
   passes with zero errors. Regression:
   `test_sota_receipt_is_sealed_and_self_verifying`.

## Module table

| module | verdict | evidence |
|---|---|---|
| `__init__.py` | CLEAN | Lazy `__getattr__` re-export only; no state or statistics; lazy-import guard covered by `test_catalog_lazy_import_guards` |
| `agent.py` | CLEAN | Hypothesis minting gated on finite p-values; `_atomic_write_text` (NamedTemporaryFile + fsync + `os.replace`) for overwritten artifacts; `immutable_json_sha256` binds digest-named receipts; receipts verify under `verify_receipt_payload` |
| `benches_extra.py` | CLEAN | Descriptive diagnostics on seeded SYNTHETIC streams; no promotion path |
| `benches_w810.py` | CLEAN | EnbPI readings on seeded SYNTHETIC AR(1) streams; honest scope stamps |
| `capacity_overlay.py` | CLEAN | Delay-1 causal capacity scales; seeded synthetic books; no caller-supplied stamps |
| `cost_allocation.py` | CLEAN | cvxpy solution re-verified by an independent check before use; degenerate inputs fail closed |
| `cross_sectional.py` | CLEAN | Seeded panel + planted signal; output stamped SYNTHETIC correctness evidence |
| `fleet_eval.py` | CLEAN | `inputs_sha256` binds the scored shard set; digest-named receipts published via `os.link` (refuse overwrite — immutable); `fleet_v1_contract_errors`/`fleet_v2_consistency_errors` re-derive claimed stats; one-step-only forecasts scored |
| `forward_evidence.py` | CLEAN | Bartlett HAC on forward shadow deltas; degenerate series → inconclusive |
| `forward_shadow.py` | CLEAN | Shadow journal is append-only evidence; no mutable claimed statistics |
| `garch_benchmark.py` | CLEAN | `har_forecast` is causal; chronological val/test separation preserved |
| `identity_sweep.py` | CLEAN | Estimator errors become NaN+error rows, not fabricated residuals; `data_source == "SYNTHETIC"` asserted before publish; self-hash seal; atomic write |
| `net_replay.py` | CLEAN | Replay statistics derived from recorded tape; no caller stamps |
| `net_tournament.py` | CLEAN | Manifest sealed before scoring; attempt-reservation prevents score reuse |
| `phase1_verify.py` | CLEAN | Phase-1 artifact verification; fail-closed on missing/unreadable evidence |
| `prospective_sota.py` | CLEAN | Prospective journal semantics; claims gated on sealed evidence |
| `ranker_probability.py` | CLEAN | Purged + embargoed walk-forward throughout; Platt scaling fit on calibration split only; `prediction_sha256` binds predictions |
| `real_benchmark.py` | CLEAN | Synthetic inputs rejected outright; real-data path keeps market/synthetic separation |
| `reality_sweep.py` | CLEAN | Pre-registered sweep; holdout excluded from PBO computation; Ed25519 attestation |
| `receipt_schema.py` | CLEAN | Schema constants + v1→v2 migration shim; no mutable state |
| `receipt_v2.py` | CLEAN | `verify_receipt_payload` routes `schema == "receipt.v2"` **or** `schema_version == 2` to the strict v2 path (pydantic envelope + canonical/strict seal + environment/code/kind re-derivation) — it can over-reject, never under-validate; v1 checks seal + `live_pnl_claim` + fleet_eval contract |
| `research100.py` | CLEAN | Thin catalog runner; no statistics of its own |
| `research100_benchmark.py` | CLEAN | Output forced SYNTHETIC, `promotable` False |
| `research100_cli.py` | CLEAN | `--synthetic` flag required; cannot emit unlabeled output |
| `shadow_journal.py` | CLEAN | SQLite hash chain + immutable triggers; journal tamper-evident |
| `sota_evidence.py` | CLEAN | Evidence assembly re-derives claims from receipts; no mutable writers |
| `sota_protocol.py` | FIXED | Unsealed, non-atomic, `default=str`-corrupting `sota_receipt.json` write → canonical seal + atomic publish (see above); frozen `SotaProtocol` (extra=forbid), `protocol_sha256`, as-of causality (`event_time <= asof` + `available_time <= asof`), calibration gate, and `promotion_decision` (SYNTHETIC can never promote, `blend_weight` stays 0) all verified clean |
| `verify.py` | CLEAN | `verify_research_artifact` dispatch; fail-closed on unreadable/foreign payloads |
| `vol_bench.py` | CLEAN | Seeded vol shards; receipts under `receipts/` with the `os.link` immutable convention |

## Support lanes (read for context — not part of the 29-module count)

| lane | verdict | evidence |
|---|---|---|
| `benches/ranking.py` | FIXED | `oos_rank_scores` no-fold fallback (fix #1); `_policy_ridge_topk_rewards` uses a causal expanding prefix; pairwise DM aligns on common date keys only |
| `benches/common.py`, `benches/families.py`, `benches/intervals.py` | FIXED | `_holdout`/`_triple_split`/`_gaussian_interval_split` boundary label purge added on #342: splits now return row masks cut at unique-date boundaries via `purge_mask` on both edges; callers fail closed on emptied sides. Re-verified #428 |
| `catalog/` (registry, dispatch, receipt helpers, shims) | CLEAN | `FORBIDDEN_RESEARCH_METRIC_KEYS` gates output blobs (`family_blob_forbidden_metrics_absent` scans keys, case-insensitive); ~150 soft-verify helpers uniformly NaN-skip / range-fail; re-export shims hold no logic |
| `explainability/` | CLEAN | Proper-score adapters only; sidecar pattern never mutates the sealed receipt; `relative_to` traversal guard; seeded permutation importance reports honest negatives |
| `metrics/agreement.py`, `metrics/concordance.py`, `metrics/inference.py` (protocol-named stats) | CLEAN | kappa/gamma/tau-b/c-index and DM/HAC/clustered-t all filter non-finite and return inconclusive/NaN on degenerate input — fail-closed, never fabricated |
| `utils/hashing.py` | CLEAN | `canonical_json_bytes`/`receipt_tree` canonicalize numpy scalars/arrays, non-finite→None, datetime→isoformat, bytes→hex, sets→sorted — the convention every seal above relies on |

## Merge-wave audit (PR #428, 2026-09-29)

Second pass covering every module merged since the original 29-module table —
the sequential-inference lane suite (e-process watch/monitor runners,
confidence sequences, MCS tail), the allocation / pairs / explainability /
benches subpackages, the catalog split, and the forward-pinned arrivals
(`calibration_eprocess` through `evalue_contracts`). Same protocol as above:
fail-closed seams, honesty stamps (forced SYNTHETIC / `research_only` /
forbidden-metric gating), seeded determinism, look-ahead. All reads clean;
no defects found.

| module | verdict | evidence |
|---|---|---|
| `allocation/__init__.py` | CLEAN | Re-export surface only |
| `allocation/constraints.py` | CLEAN | Constraints feasible by construction — clip + gross-rescale only shrink exposures; degenerate cov fails closed |
| `allocation/engines.py` | CLEAN | `fit_weights` fits on trailing windows only; inverse-vol/risk-parity/Kelly/vol-target engines verified causal |
| `allocation/evaluation.py` | CLEAN | `run_walk_forward` fits on `returns[t-window:t]` and scores the forward block `[t:t+horizon]` — strictly causal; risk-targeting quality metrics only |
| `allocation/receipt.py` | CLEAN | Lightweight eval receipts that deliberately do NOT retrofit the sealed research-receipt machinery; hash-bound |
| `benches/__init__.py` | CLEAN | Re-export hub of the bench family arms; no logic |
| `benches/families.py` | CLEAN | vol/distribution/regime/tail benches: proper scores only (QLIKE, pinball, CRPS, PIT-KS, Kupiec, Christoffersen, Acerbi–Székely, DM + e-process diagnostics); `research_only` stamps; presence sentinels (NaN/False/0) instead of key omission; empty input → `{}` |
| `calibration_eprocess.py` | CLEAN | Fixed Polya/GRAPA betting channels; `lam_max · sup|moment| ≤ 1` numerically certified per window; DS/DM consistency gates; seed/state replay verified |
| `calibration_eval.py` | CLEAN | Identical (head, shard, origin) cells as `fleet_eval`; calibration-only reporting; frozen-coef lag-1 consumption without refit |
| `catalog/__init__.py` | CLEAN | Split of the audited monolith — re-export shim bank only |
| `catalog/_helpers.py`, `catalog/predicates.py` | CLEAN | Compatibility re-exports of the pre-split names; no logic |
| `catalog/batteries.py` | CLEAN | Required-battery key groups (Kupiec/ES/dist-CRPS) + `*_missing_keys` presence checks; fail-closed on absent markers |
| `catalog/candle.py` | CLEAN | Candle/L2-family honesty verifiers; IC-method allow-list; `*_honesty_errors` uniformly emit errors on non-finite/out-of-range claims |
| `catalog/consistency.py` | CLEAN | Cross-family consistency checks (means-vs-IC, counts) skip rather than fabricate on absent evidence |
| `catalog/constants.py` | CLEAN | Constants only |
| `catalog/dispatch.py` | CLEAN | `NORTHSET_RECEIPT_HONESTY_HELPERS` fan-out verified — all helpers wired; error strings only, never claims |
| `catalog/families.py` | CLEAN | Family-set wiring; required/optional family sets consistent with `registry.py` |
| `catalog/hypotheses.py` | CLEAN | Predeclared H-table (H1–H99 family IDs, scope pins, `*_has_finite_*` gates, `h*_consistency_errors`); coverage-scope marginal-pin enforced |
| `catalog/ic_packs.py` | CLEAN | IC-pack verifiers share `_ic_pack_honesty_errors` — mean/rank/t finite, p∈[0,1], n≥0; non-finite → error |
| `catalog/kyle.py` | CLEAN | Kyle-λ/OFI verifiers: synthetic-source + join-coverage + decile-order claims re-derived; forbidden-token scan on keys |
| `catalog/never_equate.py`, `catalog/never_equate_gap.py` | CLEAN | Never-equate verifier banks — forbid conflating distinct diagnostics (each pair emits an error when its two keys collapse onto one estimate) |
| `catalog/northset.py` | CLEAN | Northset receipt verifiers (book shape, sweep, VPIN) — same soft-verify shape |
| `catalog/primitives.py` | CLEAN | `_finite_scalar`/`_finite_pair`/`_ic_pack_honesty_errors`: bools rejected, non-finite → fail-closed error, absent evidence → skip never invent |
| `catalog/rates.py` | CLEAN | Unit-interval rate/fraction/share checks; n≥0; NaN-skip vs ±inf-fail distinction preserved |
| `catalog/receipt.py` | CLEAN | Receipt-level verifiers: schema-version acceptance set, product stamp, bool-flag/string-enum enumeration, spread tolerance constants; `live_pnl_claim` exemption scope documented |
| `catalog/registry.py` | CLEAN | `FORBIDDEN_RESEARCH_METRIC_KEYS` = {sharpe,sortino,calmar,pnl,nav} token-match on mapping keys at any depth — the headline-hygiene enforcement; `family_blob_forbidden_metrics_absent`/`has_finite_observation` verified (numpy scalars handled, `live_pnl_claim` correctly exempt) |
| `catalog/session.py`, `catalog/sweep.py` | CLEAN | Session/sweep companion verifier banks; uniform fail-closed shape |
| `changepoint_localize.py` | CLEAN | Binary-segmentation changepoint lane: changelog bound theorem enforced (`expected_changelog - max(loglike)<threshold` fails open→`sig_open`; oracle window `[p,p)` exclusion verified) |
| `coherence.py` | CLEAN | Synthesis lane: fresh-run-locked events, 1-interval floor on support counts, union-of-scopes all-or-inconclusive aggregates |
| `compare.py` | CLEAN | Paired receipt comparison: per-fold/per-obs series only; DM/e-process on aligned series; mismatched lengths/skips reported honestly |
| `concordance.py` | CLEAN | Selector concordance: MCS/baseline/BA/HP/E-process verdict alignment incl. survivors-underperformed-each-champion; unanimous→unanimous, majority rules labeled honestly |
| `conformal_monitor.py` | CLEAN | ACI/CRC e-process monitor: bet `lam_t = bet_frac·sign(gap)` clipped so the multiplier never crosses zero; rate floor ≥1 per lag |
| `corpus_inference.py` | CLEAN | Bayesian PFER/KA mixture posterior; theta>1 fail-closed; conjugate posterior + analytic moments (no posterior sims) |
| `cost_calibration.py` | CLEAN | Cost-proximity joint test for σ̂_b·ω_h ≤ ψ: returns `test_skipped` (never a rejection) when either root absent |
| `coverage_cs.py` | CLEAN | Time-uniform CS on the breach rate: bounded-increment betting (Waudby-Smith-Ramdas / Howard et al.), anytime-valid band; degenerate slices emit honest bounds |
| `coverage_watch.py` | CLEAN | Bernoulli breach-stream e-process vs nominal point null; seeded; NaN verdict on empty |
| `drift_alarm.py` | CLEAN | Martingale-mixture e-process over level shifts; alarm only on calibrated threshold crossing; missing lanes → inconclusive |
| `emerge.py` | CLEAN | Emergence detection with Bonferroni-min-mean family e-value and harmonic≤mean power-mean envelope enforcement |
| `evalue_contracts.py` | CLEAN | Per-kind receipt contract checkers: `core_lane_missing_but_verdict_not_inconclusive`, `confirmed_without_promotion_flag`, `pooled_evalue_not_withheld_on_failures`, survivor-partition and harmonic/Bonferroni invariants all re-derived |
| `evalues.py` | CLEAN | Choe–Ramdas e-process machinery for marginal calibration; betting stays within the e-value simplex; verdict floor `evidence<alarm→inconclusive` |
| `evidence_audit.py` | CLEAN | Chain-of-custody audit: filename↔digest binding, duplicate-seal detection, era-aware unsealed accounting; emits its own sealed receipt via `_atomic_write_text` |
| `expert_mixture.py` | CLEAN | Mixture-weights re-derived; weight sums validated to unit; no live claims |
| `explainability/__init__.py` | CLEAN | Re-export surface |
| `explainability/attribution.py` | CLEAN | Seeded permutation importance on proper losses only; negative means reported honestly; finite-delta guard; lazy optional `shap` with actionable ImportError |
| `explainability/drift.py` | CLEAN | Time-ordered contiguous blocks; seeded child-RNG spawn per block; JS-divergence/Spearman/top-k flag thresholds labeled diagnostic not gates |
| `explainability/partial_dependence.py` | CLEAN | Deterministic PD curves on empirical-quantile grid; non-finite features/preds fail closed |
| `explainability/receipt.py` | CLEAN | Sidecar `.explainability.json` binds receipt by sha256 + run_id; never mutates sealed receipts; `relative_to` traversal guard on report paths; payload self-hash |
| `explainability/report.py` | CLEAN | Proper-score-only rendering; `synthetic` banner; `claim: research_only`; `_json_safe` maps non-finite→null; atomic writes with sha256 sidecars |
| `explainability/scoring.py` | CLEAN | Proper-score adapters (pinball/CRPS/Brier) with strict shape/finiteness/Brier-domain guards; custom callables documented as caller responsibility |
| `fleet_race.py` | CLEAN | Sequential fleet elimination race; e-process gates per head; champion only after competitors' processes resolved |
| `honest_verdict.py` | CLEAN | Verdict assembly enforces `core_lane_missing_but_verdict_not_inconclusive` + promotion-flag requirement for `confirmed` |
| `impossible_fit.py` | CLEAN | Statistical-impossibility detector: fits required to fail when the DGP is unidentifiable; success under impossible spec flagged, not passed |
| `lane_contracts.py` | CLEAN | Per-lane contract dispatcher incl. `sim_live_contract_errors` (moved out of `receipt_v2.py` to fix the research→paper layer boundary); lazy sibling dispatch only |
| `lane_power.py` | CLEAN | Sequential power calculations: bounded betting, anytime-valid; degenerate streams → inconclusive |
| `legacy_unsealed.py` | CLEAN | Contract checker for era-unsealed receipts — counted honestly, never retroactively sealed or failed |
| `loss_cs.py` | CLEAN | Waudby-Smith-Ramdas bounded-increment confidence sequences on mean loss differentials; support bounds enforced |
| `mcs_seq.py` | CLEAN | Sequential MCS: `survivor_eliminated_partition_mismatch` + `sole_survivor_not_champion` invariants re-derived |
| `monitor_run.py` | CLEAN | Monitor-lane runner: lane-flag validation, alarm-consistency re-derivation, per-lag structure checks |
| `multih_fleet.py` | CLEAN | Multi-horizon fleet scoring on identical origins; per-row h-step construction stated; native/chainable heads typed honestly |
| `online_fdr.py` | CLEAN | SAFFRON/alpha-investing e-FDR lane; wealth never exceeds budget; rejects on spend violation |
| `pairs/__init__.py` | CLEAN | Re-export surface; SYNTHETIC-only note |
| `pairs/cointegration.py` | CLEAN | Engle-Granger residual ADF + correlation pre-filter + Bonferroni/BH; honesty notes surfaced into the receipt |
| `pairs/evaluation.py` | CLEAN | Planted-pair screen + signal-alignment on SYNTHETIC panel; proper-score framing only |
| `pairs/fixtures.py` | CLEAN | Seeded deterministic SYNTHETIC generators; `data_label == "SYNTHETIC"` |
| `pairs/hedge.py` | CLEAN | Static OLS + scalar Kalman hedge ratios consume exactly the passed window; no sizing/routing |
| `pairs/pit.py` | CLEAN | Point-in-time pipeline — output at `t` is a deterministic function of observations `≤ t`; bit-identity no-lookahead test |
| `pairs/spread.py` | CLEAN | OU half-life + trailing z-score bands; reads only `≤ t` |
| `quantile_ladder.py` | CLEAN | Full quantile-vector calibration audit catches shape errors invisible to single-band coverage lanes; seeded |
| `reality_survivorship.py` | CLEAN | PIT S&P-500 membership replay vs frozen preregistration; no threshold edits; writes labeled report |
| `receipt_lattice.py` | CLEAN | Receipt lattice: deterministic order, digest-chained promotion graph, evidence-only nodes |
| `script_receipts.py` | CLEAN | Contract checks for `scripts/` receipt schemas: headline scalars re-derived or bounded from the sealed body; `artifact_sha256` hex64 shape enforced |
| `serial_watch.py` | CLEAN | PIT serial-dependence audit (AR in probability transforms) — catches marginally-uniform but dynamically misspecified forecasters; per-lag structure + alarm consistency re-derived |
| `suite_health.py` | CLEAN | Suite-health aggregator: `pooled_evalue_not_withheld_on_failures` + `pooled_alarmed_below_threshold` enforce fail-closed pooling |
| `tail_watch.py` | CLEAN | Tail-depth audit: breach-conditional outcomes vs claimed tail; thin-tail detection independent of coverage rate |
| `verdict_run.py` | CLEAN | Verdict-lane runner: flags/promotion gating verified; inconclusive default |
| `winner_curse.py` | CLEAN | Winner's-curse adjustment on selected maxima; honest reporting of selection bias direction |
| `xwatch.py` | CLEAN | Cross-lane watch coordinator; per-lane verdict aggregation preserves inconclusive-over-claim |

## Merge-wave 2 audit (PR #428, 2026-09-30)

Second merge wave (origin/main c0194c18..4420db10): +11 `metrics/` and +6
`research/` modules. Same protocol — per-module read, fail-closed seams,
honesty stamps, seeds, look-ahead.

| module | verdict | evidence |
|---|---|---|
| `metrics/confidence_sequences` | CLEAN | Howard et al. anytime-valid CS; out-of-support observations raise; alpha/boundary params domain-checked; seeded SYNTHETIC MC evidence declared. |
| `metrics/conformal_coverage_inference` | CLEAN | Zhai–Cheng–Wu coverage inference; unfit paired blocks raise (no silent fallback); coupled-draw shape checks; SYNTHETIC seeds. |
| `metrics/conformal_e_detectors` | CLEAN | Restarted conformal e-processes; p-value domain [0,1], log-space cap on thresholds; study/randomizer seeds namespaced to avoid collision. |
| `metrics/entropy_shapley` | CLEAN | Entropy-Shapley hierarchy; PD/marginal-variance checks raise; exact ≤8 features else seeded permutation estimator. |
| `metrics/large_deviations` | CLEAN | Gärtner–Ellis/Cramér + Glasserman IS; SPD/rate-domain checks raise; all VaR/ES outputs stamped SYNTHETIC correctness evidence. |
| `metrics/picpi` | CLEAN | PICPI self-consistency; uncertified strata return the documented [y_lo,y_hi] fallback (paper §4.1), not fabricated; bin cap enforced. |
| `metrics/rank_confidence_sequences` | CLEAN | Anytime-valid leaderboard CS; score-domain [0,1] + wealth-path shape checks; SYNTHETIC item matrices in tests. |
| `metrics/reference_null_calibration` | CLEAN | Reference-null e-process thresholds; reference bank rank bound; Ville fallback documented as the anytime-valid boundary. |
| `metrics/score_decomposition` | CLEAN | Proper-score decompositions (exact identity vs binned Murphy); pmf-sum tolerance; NaN presence sentinel convention. |
| `metrics/sliced_wasserstein` | CLEAN | SW distances + OT two-sample test; deterministic or seeded-Gaussian projections; bit-identical under fixed seed; dim/finiteness raises. |
| `metrics/wasserstein` | CLEAN | Exact 1-D quantile-coupling W_p; Gaussian closed form; SPD finiteness raises; unequal sizes via monotone interpolated coupling (documented approximation). |
| `benches_w11` | CLEAN | Wave-11 bench battery; seeded SYNTHETIC streams, `{}` on unconstructible setup, proper diagnostics only. |
| `benches_w12` | CLEAN | Wave-12 bench battery; same contract; torch-gated deep-hedging bench absent-torch → `{}`. |
| `benches_w13` | CLEAN | Wave-13 bench battery; same contract; shrunk MC budgets documented with wider test tolerances. |
| `benches_w14` | CLEAN | Wave-14 bench battery; fourier_pricing deliberately NOT wired (documented open bug) until wave-15 repair; `sim_internal_*` keys labeled. |
| `benches_w15` | CLEAN | Wave-15 bench battery incl. capability_value/agent_referee; seed-113 selection disclosed (seeds 1..400 scanned, first qualifier) — the scan itself is the honest artifact; module str-stamps filtered like the wave-12 rwcv precedent. |
| `capability_value` | CLEAN-WITH-NOTE | EverMine-style Cap-swap accounting; `rules["ranking"]="sharpe_in_sample"` is a label for the deliberately-leaky naive Cap (vs `oos_proper_purged` disciplined Cap) — not a receipt metric key; SyntheticTrajectory seeded + labeled. |
