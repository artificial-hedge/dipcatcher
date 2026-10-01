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

## Standing budgets

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

## Current census (2026-09-30, post-wave-16)

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
