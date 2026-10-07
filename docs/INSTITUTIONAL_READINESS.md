# Institutional readiness matrix

Dipcatcher is Artificial Hedge's research lab. This matrix is deliberately
evidence-gated: a capability is not marked ready because a module exists or a
synthetic fixture passes.

| Control area | Current state | Evidence / gate |
|---|---|---|
| Point-in-time data contract | Implemented for file adapters; revision-aware on the evaluation side | PIT timestamp validation, duplicate-key checks, OHLCV checks, manifest hash verification, and decision-time filtering for late cross-sectional/market aggregates. Wave 15 adds `validation/vintage_eval.py` (VINTAGE-TS): validity-interval reconstruction, delayed-label filtering, and hindsight-contamination audits over SYNTHETIC revision regimes — the *evaluation* layer now detects vintage cheating even though true as-of vintages on the public tape remain a procurement gap. |
| Configuration safety | Implemented | Unsupported data sources, non-finite simulator parameters, and non-stationary synthetic persistence fail validation |
| Data lineage | Implemented | `dipcatcher doctor` requires source identity, four canonical artifacts, SHA-256s, nonnegative row counts, and data-root containment |
| Research reproducibility | Implemented | Immutable receipt binds Git revision, dirty-worktree hash, config, dataset content, runtime, package versions, and benchmark catalog version |
| SOTA scientific benches | Implemented as research diagnostics on the lab panel | Canonical verifier requires the versioned 23-family catalog; fixture-only toys are `dgp=fixture` and excluded from panel Kupiec H-rows; panel-or-skip (empty blob) fails scorecard honesty rather than silently mixing DGPs |
| Validation integrity | Implemented | Walk-forward, purge/embargo, CPCV audit, HAC/DM, multiple-testing and conformal family gates. Waves 12–16 add the anytime-valid layer: e-BH/e-LORD/e-SAFFRON online FDR, minimax-optimal conformal e-detectors, rank confidence sequences (anytime-valid ranker ordering), replicable conformal (auditable calibration thresholds), and `validation/agent_referee.py` — a frozen post-submission-only betting referee so agent-proposed factors are judged at every stopping time by a procedure the proposer cannot touch. |
| Numerical risk controls | Implemented | Covariance inputs are finite and square; PSD repair is deterministic eigenvalue clipping with finite-output guarantees |
| Execution research | Implemented as simulation | Next-open fills, costs, participation, Almgren–Chriss/TWAP comparisons, and tamper-evident backtest artifacts |
| Paper / shadow operation | Implemented as simulation | Crash-resumable simulated broker; resume appends prior orders/equity/shadow-equity/positions/cash rows and fill receipts; deterministic target/exposure ordering; kill switch, champion/shadow dry-run, no live capital |
| Promotion | Fail-closed | Synthetic evidence, missing metrics, non-finite metrics, and invalid receipts cannot promote |
| CI reproducibility | Implemented | Frozen `uv.lock`, lock consistency check, full tests, type/lint checks, canonical receipt verification |
| CI token scope | Implemented | Workflow `GITHUB_TOKEN` is restricted to repository contents read access; no job in CI requires write access |
| Package distribution | Implemented | CI builds wheel and source distribution, installs the wheel without the checkout installed, and runs the installed `dipcatcher --help` entry point |
| API security boundary | Implemented for local/service operation | API-key authentication, loopback fail-closed default, constant-time key comparison, secure response headers, artifact-root containment, 64 KiB streamed request-body cap, and strict request schemas |
| Dependency and code security | Implemented as CI gates | Locked dependency `pip-audit` report, medium/high Bandit gate, and retained machine-readable audit artifact |
| Container operations | CI-gated | Non-root runtime user, loopback default, healthcheck, immutable dependency sync, host-local optional MLflow binding; CI builds the image and waits for its healthcheck |
| Vendor market data | **Blocked — procurement/authorization pending** | A fail-closed licensed-vendor adapter and a condition-1 `QUALIFYING` gate exist (`quant_fund.data.qualifying`), enforcing `event_time <= available_time <= ingested_time`, `available_time <= decision_time`, revision handling, and hashed PIT receipts, with an `UNAVAILABLE` path and **no synthetic substitution**. This is **not evidence**: no entitlement and no qualifying externally timestamped complete-universe feed are configured. The local retrospective snapshot does not qualify. See [EVIDENCE_PROCUREMENT.md](EVIDENCE_PROCUREMENT.md). |
| Live broker connectivity | **Not implemented — authorization pending** | No live orders, broker credentials, or live P&L claims are supported. The paper/simulated order/fill reconciliation surface (`quant_fund.execution.order_recon`, `quant_fund.paper.broker_adapter`) is the condition-2 reference shape only; configuring a live endpoint **raises** (`LiveEndpointRefused`). See [EVIDENCE_PROCUREMENT.md](EVIDENCE_PROCUREMENT.md). |
| Live readiness | **Blocked by missing external evidence** | Requires authorized vendor data, a real broker adapter with authenticated reconciliation, venue measurements, a non-synthetic forward record, and independently verified evidence. The 2025 holdout is **SPENT**; the named forward window `forward_2026H2` (start 2026-07-01) is **NOT YET COLLECTED**. |


## Dual honesty catalogs (research vs analytics export)

Two different surfaces — do not conflate:

1. **Research scorecard / family blobs** — `FORBIDDEN_RESEARCH_METRIC_KEYS` /
   `family_blob_forbidden_metrics_absent` reject sharpe/sortino/calmar/pnl/nav
   *key tokens* in lab headlines.
2. **Paper/backtest `analytics_export`** — may nest equity `nav_*` and stress
   `*_pnl` diagnostics under `ANALYTICS_SCHEMA_KEYS`; honesty gate is
   `live_pnl_claim=false` via `validate_analytics_export` (fail-closed on true).

Never treat a valid analytics export as a research family scorecard blob.
`/ready` is fail-closed when `data_manifest` or the immutable research receipt is
missing/invalid (the configs allowlist resolves `_CONFIGS_DIR` before containment
checks). A green readiness response therefore proves both data provenance and
research-artifact integrity for the configured research runtime.

## Minimum evidence before any live claim

All of the following must be present and independently reviewable:

1. A licensed, point-in-time data source with release and ingestion timestamps.
2. A broker adapter with authenticated order and fill reconciliation.
3. A non-synthetic holdout and deployment-shaped forward/shadow record.
4. Cost, liquidity, borrow, financing, and failure-mode measurements from the
   target venue.
5. A signed promotion receipt whose immutable verifier passes and whose live
   authorization is explicit.

Until those conditions exist, Dipcatcher outputs are research, backtest, or
simulated paper/shadow evidence only.

A working fail-closed adapter, a compiling module, or a passing SYNTHETIC
fixture is **not** evidence and never marks a condition ready. Conditions 1 and
2 remain **BLOCKED on procurement/authorization** — they require an external
purchase/entitlement and an independently reviewable artifact, not code. The
actionable checklist for what to buy/authorize and the pass/fail acceptance
tests for each condition is [EVIDENCE_PROCUREMENT.md](EVIDENCE_PROCUREMENT.md).

**Holdout status:** the 2025 holdout is **SPENT** (the vendor pool already has
results for it and discloses survivorship bias). The named forward window is
**`forward_2026H2`, start 2026-07-01**, currently **NOT YET COLLECTED**; its
pre-registration is hash-sealed and must be externally timestamped and frozen
before it runs (see [REALITY_PREREGISTRATION.md](REALITY_PREREGISTRATION.md)).

## Waves 12–16 addendum (2026-09-30)

What changed relative to the five conditions — and what did not:

- **Condition 1 (licensed PIT data)**: unchanged. Procurement, not code.
  The evaluation side is now vintage-aware (`vintage_eval`), so when a
  real as-of source arrives the harness can already audit revision
  hindsight instead of silently absorbing it.
- **Condition 3 (forward/shadow record)**: strengthened in *kind*, not in
  substance — the frozen referee (`agent_referee`) and rank confidence
  sequences give any future forward record anytime-valid admission
  semantics (no optional-stopping inflation), and `capability_value`
  makes self-evolution claims falsifiable via Cap-swap accounting. No
  non-synthetic forward record exists yet.
- **Condition 5 (signed promotion receipt / verifier)**: the immutable
  verifier itself is now mutation-tested — `research/verify.py` scores
  99.52% on the full 210-mutant campaign (sole survivor a proven
  equivalent mutant, pinned by documentation test) and
  `catalog/registry.py` 100% (24/24). "Independently reviewable" now
  includes measured fault-detection evidence for the reviewer code.
- **Scorecard governance**: `docs/BENCHMARK_FAMILY_LIFECYCLE.md` adopts
  OPTIONAL→REQUIRED/RETIRED lifecycle rules, runtime budgets, and review
  cadence for the benchmark-family scorecard.
- None of the above is market evidence. Every new module ships seeded
  SYNTHETIC correctness tests; the honesty contract is unchanged.
