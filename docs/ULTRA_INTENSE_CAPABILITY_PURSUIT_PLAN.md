# Ultra-Intense FX-1 Capability Pursuit Plan

**Status:** proposed execution plan  
**Prepared:** 2026-10-03  
**Primary target:** the million-capability ambition recorded in `INFLIGHT` and `docs/FX1_CAPABILITY_PROGRESS.md`  
**Relationship:** focused companion to [`ULTRAPLAN_FRONTIER.md`](ULTRAPLAN_FRONTIER.md); this plan covers capability scale and verification, not the wider forecasting or strategy lanes.

## Mission

Pursue **1,000,000 distinct, independently implemented capabilities** and at least **1,000,000 lines of code**, while making every accepted unit useful, discoverable, behaviorally verified, and honestly described. Treat the counts as audit results, never as a proxy for research quality or market evidence.

The operating principle is **verify first, then scale without lowering the bar**. No capability enters the accepted count because it has a unique name, generated record, copied template, wrapper, parameter variant, or passing import check.

## Starting position

The latest working-tree progress note records this baseline as of 2026-10-02:

- 157 implementations in 157 separate files: 47 features, 76 skills, and 34 plugins.
- 38,712 physical Python source lines, 34,571 nonblank lines, and 32,155 code-token lines under the current inventory method.
- 999,843 implementations remain. The million-code-line target also remains unmet.
- Static inventory, Ruff, targeted mypy, and architecture freshness checks are recorded as passing for the recent batches. Full-batch behavioral verification is still outstanding in `INFLIGHT`.
- There is no trained fx-1 checkpoint in this repository. These implementations do not establish forecast accuracy, market evidence, or live-trading readiness.

Before work starts, refresh these figures from the current tree. Preserve the existing uncommitted work; do not clean, reset, or overwrite it as part of this plan.

## Non-negotiable rules

1. **Independence:** one implementation file must contain the capability's own substantive algorithm or parser. Aliases, wrappers, generated cards, copied templates, and parameter combinations do not count.
2. **Behavior before registration:** every candidate needs executable positive and negative cases, a stated contract, and a bounded failure policy before it is registered as accepted.
3. **No papering over gaps:** unsupported cases return a clear unsupported/error status. Do not silently fabricate values, widen format support, or weaken a gate to improve throughput.
4. **Honesty contract:** preserve proper-score reporting, `SYNTHETIC` labels, immutable receipts, and the hard ban on live-trading claims. Never headline Sharpe/Sortino/Calmar/P&L/NAV as research evidence.
5. **Separate counts:** publish physical lines, nonblank lines, and code-token lines separately. Freeze the acceptance metric before scaling; do not switch denominators midstream or pad with comments/docstrings.
6. **Research boundary:** a capability implementation proves only the behavior its tests and evidence establish. It does not prove causal availability, empirical predictive value, vendor-feed completeness, or live suitability.

## Acceptance contract for one capability

An item is accepted only when its record contains all of the following:

- Stable unique ID, kind, source path, owner, version, and content hash.
- A concise purpose statement, input/output schema, units, bounds, and explicit unsupported cases.
- A substantive independent implementation with no hidden dependency on a generated wrapper for its core behavior.
- Deterministic reference examples and edge cases. Numerical methods also need property or metamorphic checks; parsers need malformed/truncated/oversized-input cases.
- Negative tests for invalid types, non-finite values, boundary violations, ambiguity, and resource limits where relevant.
- Successful discovery, description, and execution through the supported FX-1 harness path—not merely direct Python import.
- Ruff and formatting, targeted type checks, focused tests, and integration checks appropriate to the touched surface.
- Independent source review for duplication, algorithmic correctness, failure behavior, resource bounds, and overclaimed semantics.
- A progress entry naming the evidence run, test scope, limitations, and any unresolved behavior.

The count is **accepted capabilities only**. Keep separate totals for proposed, implemented, verified, rejected, and deferred items.

## Operating model

Run a tight cycle with four lanes. Parallel work is allowed only on disjoint files and registry surfaces; one integration owner controls registration and acceptance.

| Lane | Responsibility | Exit condition |
|---|---|---|
| Contract and inventory | Freeze ID rules, reconcile registry/files/guide, identify duplicates and missing schemas | Every existing and proposed ID has one auditable disposition |
| Implementation | Build small batches of 6–12 genuinely distinct units from reviewed specifications | Candidate code and focused tests are ready; nothing is counted yet |
| Independent verification | Challenge each candidate with reference, edge, adversarial, and resource-limit cases | Review findings closed or item rejected/deferred |
| Integration and release | Run harness, repo gates, update architecture and progress evidence | Clean gate report and reproducible accepted-count delta |

Use full-day work blocks and short feedback loops: design, implement, challenge, integrate, report. Do not queue a large wave of unverified code behind a single final check.

## Phases and gates

### Phase 0 — Protect the baseline and freeze the counter

**Timebox:** first half-day.

- Record current `git status`, branch, and a read-only inventory snapshot before touching capability files.
- Recompute IDs, distinct source paths, per-kind counts, and all three LOC measures from the working tree.
- Compare `src/fx1/operations/registry.py`, source files, `docs/FX1_OPERATIONS.md`, and `docs/FX1_CAPABILITY_PROGRESS.md`; classify every mismatch.
- Write down the precise independence and LOC rules used for the million targets. Keep the existing method visible and report any proposed correction as a separate reconciliation, not a retroactive change.
- Make a per-capability verification matrix for the current 157, including the actual harness command and current evidence state.

**Gate 0:** no candidate work begins until inventory disagreements and counting rules are explicit.

### Phase 1 — Close behavioral verification on the existing 157

**Timebox:** first 2–4 working days; adjust from measured runtime, not optimism.

- Partition the 157 by behavior class: numerical feature, forecast score, data/contract audit, and file parser.
- Run each operation through the harness discovery/describe/execute flow and compare outputs to independent references or hand-checkable fixtures.
- Add missing behavioral coverage in small slices: ordinary cases, edge cases, invalid inputs, numeric invariants, parser truncation/corruption, and resource bounds.
- Reproduce failures, fix the implementation or narrow its documented contract, and rerun the affected cohort.
- Exercise platform-sensitive parsers on the Windows fleet where the host is available; label unverified platform behavior plainly.
- Update `INFLIGHT` and `FX1_CAPABILITY_PROGRESS.md` only with observed results.

Useful existing entry points are documented in `docs/FX1_OPERATIONS.md`, including:

```sh
uv run fx1 harness operations --kind feature
uv run fx1 harness describe-operation features.simple_returns
uv run fx1 harness execute-operation features.simple_returns \
  --arguments '{"prices":[100.0,110.0,99.0],"lag":1}'
```

**Gate 1:** all 157 have a disposition: behaviorally verified, corrected and reverified, or explicitly rejected/deferred. Static checks alone do not close this gate.

### Phase 2 — Harden the capability factory

**Timebox:** next 3–5 working days.

- Audit the current registry and harness against the acceptance contract; reuse existing protections rather than creating parallel infrastructure.
- Add or tighten automated checks for duplicate IDs, duplicate/near-duplicate implementations, missing source files, missing schemas, guide drift, and excluded generated files.
- Establish a per-kind test template only as a test aid; never use it to generate counted implementations.
- Define batch manifests containing item IDs, specifications, owners, dependencies, test commands, evidence paths, and reject/defer reasons.
- Measure how long design, implementation, review, tests, and integration take per accepted item.

**Gate 2:** a batch cannot be registered if inventory, test evidence, or review status is incomplete.

### Phase 3 — Scale through reviewed waves

**Cadence:** repeat small waves; do not raise batch size until two consecutive waves pass every gate without a growing defect queue.

Each wave:

1. Choose a cohesive useful domain from the backlog: data lineage/contracts, numerical estimators, proper forecast scores, bounded file formats, validation protocols, or FX-1 tooling.
2. Write individual specifications and identify a reference/oracle before implementation.
3. Implement 6–12 distinct units across disjoint files.
4. Run focused and adversarial tests; have an independent reviewer inspect every candidate for real algorithmic independence and semantic limits.
5. Integrate one wave at a time; reject or defer candidates that lack a trustworthy oracle or repeat existing behavior.
6. Run the relevant repo gates and refresh the evidence ledger before starting the next wave.

The wave backlog should prefer capabilities that remove a concrete research or data-integrity bottleneck. Do not optimize for easy file formats or count growth alone.

### Phase 4 — Prove delivery capacity at explicit milestones

Report the following milestone reviews; they are decision gates, not permission to weaken quality:

| Milestone | Required proof |
|---|---|
| 157 verified | Existing inventory has complete behavioral dispositions and a reconciled baseline |
| 250 accepted | 93 net-new units pass the acceptance contract; audit defect/rework rate |
| 1,000 accepted | Reproducible inventory and end-to-end harness coverage; publish measured accepted units/day and maintenance cost |
| 10,000 accepted | Demonstrate that review, discovery, tests, docs, and dependency maintenance scale without registry drift or degraded quality |
| 100,000 accepted | Reassess architecture, security, package size, discoverability, and lifecycle policy before further expansion |
| 1,000,000 accepted | Independent audit of the count, implementation independence, LOC method, maintainability, and package behavior; publish limitations and full evidence index |

At every milestone, keep failing or deferred items out of the accepted total. The 1,000-unit review must recalculate the remaining duration from measured throughput and resource capacity.

## Scale reality check

From 157, **999,843 accepted units remain**. At a sustained, fully verified pace, with no holidays, maintenance, or rework:

| Accepted units/day | Approximate time for remaining units |
|---:|---:|
| 12 | 228 years |
| 100 | 27.4 years |
| 1,000 | 2.74 years |

These are arithmetic scenarios, not forecasts. The recent 6–12 item waves establish a quality workflow, not million-scale throughput. Phase 4 is where actual capacity, staffing, compute, test runtime, review load, and ongoing maintenance must be measured. No completion date should be promised before that evidence exists.

## First 48 hours: ready-to-run checklist

- [ ] Preserve the current working tree and record its exact starting state.
- [ ] Recompute the 157-item inventory and LOC measures from source, not from generated catalog output.
- [ ] Produce a 157-row behavioral verification matrix with status and evidence fields.
- [ ] Run a representative smoke pass for each behavior class and record time/failures.
- [ ] Prioritize any honesty, unsafe parsing, unbounded resource use, or silent-fallback defect ahead of adding capabilities.
- [ ] Close a first verification cohort; update the tracker with exact commands and results.
- [ ] Only after Gate 1 is progressing cleanly, select the next 6–12 specifications.

## Repository gates for implementation waves

Follow the repository's authoritative gates and use focused checks during iteration:

- `make lint`
- `make typecheck`
- `make test`
- `make fx1-test`
- `uv run mypy src/fx1`
- `make fx1-gate` for the full FX-1 gate when appropriate
- Architecture freshness and `git diff --check` whenever generated architecture artifacts or broad edits are involved

Record exactly which gates ran. A focused pass is not a full-gate pass. Do not claim behavioral verification merely because CI is queued or a job has started.

## Stop, reject, or narrow conditions

Pause a candidate or wave when any of these is true:

- It is substantively the same capability as an existing item, or only varies parameters/names.
- There is no independent oracle, no meaningful expected behavior, or no way to challenge edge cases.
- A parser/algorithm needs broader assumptions than its contract can safely support.
- A candidate hides errors, fabricates fallback values, drops provenance, or weakens point-in-time/honesty controls.
- Review or integration defects are accumulating faster than the team closes them.
- The tested command surface differs from the registered/advertised FX-1 surface.

Prefer a smaller verified inventory with precise limits to a larger count that cannot survive independent inspection.

## Progress reporting format

At the end of each wave, report:

```text
date / wave:
accepted total (by kind):
new candidates / accepted / deferred / rejected:
LOC: physical / nonblank / code-token:
focused tests and repo gates run:
review findings and fixes:
known unverified behavior:
accepted units per working day (rolling 2-week):
next blocking dependency:
```

Update `docs/FX1_CAPABILITY_PROGRESS.md`, `docs/FX1_OPERATIONS.md`, and `INFLIGHT` from the same evidence snapshot so their counts and statuses cannot drift.

## Definition of success

The pursuit succeeds only when an independent reader can reproduce the accepted count, inspect substantive source implementations, run their advertised behavior through the harness, and distinguish tested behavior from assumptions and unsupported cases. Reaching one million by itself is not evidence that fx-1 forecasts well or that any strategy works.
