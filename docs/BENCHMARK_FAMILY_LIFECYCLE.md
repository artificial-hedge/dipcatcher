# Benchmark family lifecycle — scorecard governance

Status: adopted 2026-09-30 (wave-16 close-out). Scope: the research scorecard
families registered in `quant_fund.research.catalog.registry`
(`REQUIRED_BENCHMARK_FAMILIES`, `OPTIONAL_BENCHMARK_FAMILIES`,
`BENCHMARK_FAMILY_ORDER`) and their bench batteries
(`research/benches*.py`).

## Why this exists

Waves 8–16 grew the scorecard from ~24 to 80 families. Every family is a
runtime cost in `dipcatcher research`, a receipt-schema surface in
`verify-research`, and a maintenance obligation (its module, its bench, its
science assertions). Growth without lifecycle rules produces sprawl: dead
families nobody reads, duplicated science across batteries, and a research
run whose runtime is dominated by diagnostics no gate consumes. This policy
is the pruning counterpart to the honesty contract's growth discipline
(REQUIRED never shrinks silently; OPTIONAL is the proving ground).

## Family states

```
PROPOSED -> OPTIONAL -> (REQUIRED | RETIRED)
```

- **PROPOSED** — a lane report proposes bench keys (every wave lane does).
  Acceptance bar to enter OPTIONAL: seeded SYNTHETIC (or receipted real
  data), flat float blob, fail-closed to `{}`, proper scores only
  (`family_blob_forbidden_metrics_absent` passes), at least one *science
  assertion* (a direction/equality that would flip if the underlying math
  broke — not merely "keys present").
- **OPTIONAL** — wired in `benches_w*.py` + registry + agent families dict.
  May appear in receipts; fully soft-verified when present; never required.
  This is where families prove they are read.
- **REQUIRED** — promoted only by explicit owner decision. Bar: the family
  guards a *standing* correctness property (not a one-time paper
  replication), has been OPTIONAL for >= 2 waves without tolerance drift,
  and a named gate consumes its keys. `BENCHMARK_FAMILY_ORDER` and the
  docs-consistency contract are updated in the same commit. Promotion is
  rare; REQUIRED is a permanent CI cost.
- **RETIRED** — removed from the registry in a commit that (a) states the
  reason, (b) keeps the module + tests (retirement is of the *scorecard
  family*, never of the science), (c) notes the last receipt schema version
  that carried it so `verify-research` can keep validating historical
  receipts. Old receipts containing retired families must still verify —
  the registry's accepted-set semantics are append-only over history.

## Qualification gate (2026-10-07)

A family may enter or remain OPTIONAL only when it is backed by a
**qualifying implementation** under the canon qualification ruleset
(`src/quant_fund/models/canon_qualification.py`, hash-pinned `RULESET`;
evidence: `quality/canon_qualification_summary.json`): a real behavioral
mechanism (a substantive bench over data), not a template copy of a shared
AST skeleton whose bench aggregates constant checks. Template-shaped
constant-score families are ineligible — generator correctness fixtures, not
scorecard science.

The 2026-10-07 audit classified 7,529 template-backed optional families and
moved them OPTIONAL -> RETIRED in one recorded action. Every entry in
`RETIRED_BENCHMARK_FAMILIES` (`quant_fund.research.catalog.registry`, generated
`catalog/retired_families.py`) carries its retirement reason and
`last_receipt_schema_version = 2`. Retirement never erases acceptance:
`OPTIONAL_BENCHMARK_FAMILIES` remains the append-only accepted set, so an
archived receipt naming a retired family still verifies (pinned by
`tests/unit/research/test_catalog_retired_families.py`; `verify.py`
accepted-set semantics unchanged). The live scorecard and runtime emission
set is `LIVE_OPTIONAL_BENCHMARK_FAMILIES = OPTIONAL_BENCHMARK_FAMILIES -
RETIRED_BENCHMARK_FAMILIES`, consumed by `research/agent.py`.

Future waves pass the same gate before wiring: a family whose backing module
classifies `NON_QUALIFYING_TEMPLATE` is rejected at registration, and the
guard tests (`tests/unit/models/test_canon_qualification.py`,
`tests/unit/research/test_catalog_retired_families.py`) fail if a
template-shaped family reappears as qualifying or as a live registration.

## Standing budgets

Budgets apply to the **LIVE** scorecard (`LIVE_OPTIONAL_BENCHMARK_FAMILIES`
= `OPTIONAL_BENCHMARK_FAMILIES - RETIRED_BENCHMARK_FAMILIES`; 2,693 of
10,222 as of 2026-10-07), never to the full accepted set — the 7,529
retired template-backed families consume no runtime and must not be read as
headroom.

- Per-bench runtime: <= 8s typical, <= 15s with justification in the bench
  docstring (wave-14 `zi_lob`/`local_stoch_vol` precedent). Total added
  battery per wave: <= 60s. A family that outgrows its budget gets shrunk
  (documented wider tolerances, wave-12 `mh_enbpi` precedent) or retired.
- Torch-gated families MUST return `{}` cleanly without the `nn` extra
  (`deep_hedging` precedent) — a receipt from a torch-less runner stays
  valid.
- Duplicated science: if a new family's core assertion duplicates an
  existing family's (e.g. two conformal-coverage watchers), the newer one
  cites the older in its docstring and states the distinct failure mode it
  catches — or merges into it.

## Review cadence

At each wave close-out the controller records in INFLIGHT: family count,
total battery runtime, families whose science assertions were *relied on*
(cited in a decision, a gate, or a finding) since the last review, and
retirement candidates. Two consecutive reviews without a family being read
=> retirement proposal in the next wave's report.

## Current census (2026-10-07, post-retirement)

- REQUIRED: **23** — unchanged since wave 8 (`BENCHMARK_FAMILY_ORDER`
  untouched since inception — the docs-consistency contract).
- OPTIONAL: **10,222** — append-only historical ACCEPTED set; never shrinks
  (archived receipts must keep verifying).
- RETIRED: **7,529** — template-backed families moved OPTIONAL -> RETIRED in
  one recorded action (2026-10-07 canon qualification audit; per-entry
  reason + `last_receipt_schema_version = 2` in `RETIRED_BENCHMARK_FAMILIES`).
- LIVE (scorecard + runtime emission): **2,693** = `OPTIONAL - RETIRED`.
- Corpus evidence: 10,388 audited modules — 7,529 `NON_QUALIFYING_TEMPLATE`
  (72.5%). See `quality/canon_qualification_summary.json` and
  [`docs/CAPABILITY_QUALIFICATION.md`](CAPABILITY_QUALIFICATION.md).

### Historical census (2026-09-30, post-wave-16)

- REQUIRED: unchanged since wave 8 (`BENCHMARK_FAMILY_ORDER` untouched
  since inception — the docs-consistency contract).
- OPTIONAL: waves 8–10 (6), wave 11 (4), wave 12 (8), wave 13 (8), wave 14
  (14), wave 15 (8), wave 16 (pending integration), plus pre-existing
  (`candle_order_book`, `robinhood_plus`, `complexity`, `roughness`,
  `serial_randomness`).
- Battery runtimes: w12 3.9s, w13 11.0s, w14 15.7s, w15 21.2s — all within
  budget; w15's `capability_value` (~20s) is the largest single bench and
  the first retirement/shrink candidate if w16+ adds overlapping governance
  diagnostics.
- Known duplication watch: `conformal_e_detectors` (w13) vs `anytime_valid`
  (w8) both watch e-process detection — distinct failure modes (minimax
  delay-optimality vs FDR/FA control), documented in the w13 bench
  docstring; re-review at next cadence.
