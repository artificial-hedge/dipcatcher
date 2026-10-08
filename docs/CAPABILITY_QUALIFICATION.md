# Capability qualification standard

The acceptance test for the "1,000,000 distinct, independently implemented
capabilities usable by an AI through dipcatcher" requirement. This file is
the standard; [FX1_CAPABILITY_PROGRESS.md](FX1_CAPABILITY_PROGRESS.md) is the
ledger. Source accounting only — nothing here creates market evidence, a
live-trading claim, or a research receipt.

## 1. What qualifies

A capability counts toward the 1,000,000 only if **all** of the following
hold:

- **One separate source file** per feature, skill, or plugin. The file is
  the unit of counting.
- **Independently implemented behavior** — implemented for that capability,
  not generated, copied, or mechanically derived from another capability's
  file.
- **Usable by an AI through dipcatcher** — exposed through the AI interface
  (`src/fx1/operations`), registered in `src/fx1/operations/registry.py`,
  and documented in the [operation guide](FX1_OPERATIONS.md).

## 2. What does NOT qualify

Five named failure modes — each zero credit toward both the capability count
and the line-of-code requirement. These five are exactly what inflated
`OPTIONAL_BENCHMARK_FAMILIES` to 10,222:

- **Generated catalog records** — a record in a generated catalog or
  declaration shard with no independently implemented behavior behind it.
- **Template copies** — a copy of a shared skeleton (the corpus template is
  `X_ok(a, b) -> a and b`, `X_aux(aux) -> aux`, and a `_bench_X` returning
  `float(sum(checks) / len(checks))` over constant `True`/`False` checks).
- **Aliases** — a renamed re-export of behavior implemented in another file.
- **Parameter variants** — the same implementation instantiated again with
  different constants, seeds, or arguments.
- **Wrappers** — a thin forwarding layer over behavior that is already
  counted under another file.

## 3. How qualification is decided mechanically

"Qualifies" is reproducible, not a matter of opinion. The deterministic
classifier `src/quant_fund/models/canon_qualification.py` (`RULESET`,
`ruleset_version = 1`; `ruleset_hash` recorded in
`quality/canon_qualification_summary.json`) classifies every corpus module
as `QUALIFYING` or `NON_QUALIFYING_TEMPLATE` from file bytes alone. Two runs
over the same tree are byte-identical; any rule change changes
`ruleset_hash`. A `NON_QUALIFYING_TEMPLATE` verdict means **all** of:

1. normalized AST shape shared with >= 2 corpus modules (template copying),
2. no substantive bench (constant-check aggregation only),
3. zero control flow,
4. fewer than 40 effective body lines,
5. zero data-carrying parameters.

The checked-in summary binds the audited bytes (`models_tree_sha256`) and
the retired-family list derived from the classifier; the full dump is a
derived artifact (`.dsh-24x7/canon_qualification_audit.json`, never checked
in). The regression guards
(`tests/unit/models/test_canon_qualification.py`,
`tests/unit/research/test_catalog_retired_families.py`) fail if a
template-shaped family reappears as qualifying or as a live registration.

## 4. The honest current position

**31 implementations in 31 files** (features, skills, plugins registered in
`src/fx1/operations/registry.py`) containing **4,066 physical source lines**
(3,497 nonblank). **999,969 implementations remain** toward the
million-capability requirement. The million-LOC requirement is unmet. The
ledger and pace arithmetic are in
[FX1_CAPABILITY_PROGRESS.md](FX1_CAPABILITY_PROGRESS.md).

## 5. Template volume is not progress

Growth in generated families is **actively negative evidence**. The catalog
grew from ~80 honest families (wave-16 close-out) to 10,222 optional
families while qualifying capability stayed at 31. The 2026-10-07 audit
retired **7,529** of those families as template-shaped with no behavioral
mechanism (`RETIRED_BENCHMARK_FAMILIES`; live scorecard 2,693). Family-count
growth without independent implementation is template churn: it must be
reported as such and must never move the capability ledger. When the count
and the truth diverge, this standard requires the smaller number.
