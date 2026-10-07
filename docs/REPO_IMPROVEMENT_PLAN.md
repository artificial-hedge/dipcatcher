# Repository improvement sequence

The goal is reviewable evidence of economic value and reliable operation.
Return improvement must be measured on the target market after costs.

| Order | Work | Status and acceptance evidence |
|---|---|---|
| 1 | Real-data benchmark | Executed against the tracked 424-name snapshot with sealed validation and previously inspected test scores. Zero forecast won validation MSE; ridge's small test advantage has no economic interpretation. See [REAL_DATA_BENCHMARK.md](REAL_DATA_BENCHMARK.md) and the Phase 1 evidence report. |
| 2 | Net-return tournament | Executed the frozen equal-weight, momentum-20 and reversal-1 slate with full ledgers and both impact scenarios. Validation selected momentum-20, which underperformed equal weight on the inspected test; the economic evidence gate is false. See [NET_RETURN_TOURNAMENT.md](NET_RETURN_TOURNAMENT.md) and the Phase 1 evidence report. |
| 3 | Cost-aware construction | The matched validation run retained both failed cost-aware candidates: CLARABEL returned `optimal_inaccurate`. No candidate was selected, so there is no cost-aware test or uplift estimate. Constraints remain fail-closed. See [COST_AWARE_CONSTRUCTION.md](COST_AWARE_CONSTRUCTION.md) and the Phase 1 evidence report. |
| 4 | Forward shadow record | Local simulated protocols implemented: a frozen research candidate and benchmark with atomic settlement journal, plus a bounded paired paper adapter and pre-collection power plan. No external freeze or prospective market decision has been collected. Source verification and an independently timestamped complete-universe feed remain required. See [FORWARD_SHADOW_RECORD.md](FORWARD_SHADOW_RECORD.md), [PAPER_SHADOW.md](PAPER_SHADOW.md), and [FORWARD_SHADOW_POWER.md](FORWARD_SHADOW_POWER.md). |
| 5 | Performance and operations | **Ops floor landed (research/infra only — not readiness).** Representative-workload perf budgets are tests (`tests/unit/monitoring/test_perf_budget.py`), with reference/fast-path economics preserved (idempotent-replay parity). Recovery + observability tests: broker kill/resume NAV seam parity and no-duplicate-orders (`tests/unit/paper/test_broker_adapter.py`), ingestion crash/resume (`tests/unit/data/test_ingest_recovery.py`). Family-specific calibration drift wired into `model_health_report`; `/monitoring/drift` evidence-report hash-sidecar check preserved and fail-closed (`tests/unit/monitoring/test_drift_wiring.py`). Task-8 conformal/Student-t latency rebenchmark table in [PERF.md](PERF.md) confirms `fit_student_t` ≈110 ms dominates the causal path; the fix requires editing `src/quant_fund/metrics/**` and is **REPORTed to the Lead** (metrics owner), not edited here. Required CI gates are run by the Lead at integration. See [EVIDENCE_PROCUREMENT.md](EVIDENCE_PROCUREMENT.md). |

The existing vendor pool discloses survivorship bias and already has results
for the 2025 holdout — that **2025 holdout is SPENT** and cannot be reused as
new forward or holdout evidence. It can support retrospective diagnostics only.
Stronger evidence requires verified data availability/adjustments and a fresh
externally timestamped forward period: the named window is
**`forward_2026H2` (start 2026-07-01)**, currently **NOT YET COLLECTED**, whose
pre-registration is hash-sealed and must be externally timestamped and frozen
before it runs (see [EVIDENCE_PROCUREMENT.md](EVIDENCE_PROCUREMENT.md) and
[REALITY_PREREGISTRATION.md](REALITY_PREREGISTRATION.md)). A passing forecast
benchmark alone cannot promote a strategy or authorize live execution.

The sealed Phase 1 report is `data/metadata/research/phase1_20260925.md`;
`data/metadata/research/phase1_evidence_index.json` links its complete and
blocked runs to the frozen configs and source commit. Use
`dipcatcher verify-research data/metadata/research/phase1_evidence_index.json`
in a checkout containing the tracked snapshot and published ledgers.
