# dipcatcher maintenance — 5 October 2026

This record covers **183 confirmed merges** before the evidence PR [#2804](https://github.com/artificial-hedge/dipcatcher/pull/2804) itself merges: **14 into main and 169 into the PRs’ original feature bases**. Main was verified at `4ab99c82e6ab0d6ec72e298ae1d1edae625608a3`. The full [machine-readable ledger](maintenance-2026-10-05.json) records each reviewed head, destination branch, merge commit, branch verification and captured CI status.

The repository owner authorized merges independently of CI status. GitHub’s supported merge operation was used with expected reviewed heads. No branch protection, security control, integrity gate, or historical receipt was weakened to obtain a merge.

## Concrete fixes

| Area | Changes and PRs |
|---|---|
| Credential recovery and quotas | [#2727](https://github.com/artificial-hedge/dipcatcher/pull/2727) makes rotation durable before publishing the new key. [#2748](https://github.com/artificial-hedge/dipcatcher/pull/2748) quarantines credentials after detected journal damage, keeps authentication required through empty/zero-byte recovery, and persists request/token usage across restart. |
| Concurrent work and stored responses | [#2802](https://github.com/artificial-hedge/dipcatcher/pull/2802) protects journal state. [#2801](https://github.com/artificial-hedge/dipcatcher/pull/2801) prevents deletion resurrection and cancellation/completion regression; [#2781](https://github.com/artificial-hedge/dipcatcher/pull/2781) keeps replay cursors monotonic. #2748 serializes keyed submissions and keeps claims held while cancelled workers finish. |
| API validation and accounting | [#2803](https://github.com/artificial-hedge/dipcatcher/pull/2803), [#2796](https://github.com/artificial-hedge/dipcatcher/pull/2796) and [#2761](https://github.com/artificial-hedge/dipcatcher/pull/2761) tighten usage, expiry, authorization order and quota retry semantics. [#2736](https://github.com/artificial-hedge/dipcatcher/pull/2736) and #2748 return structured refusals for malformed values and deeply nested JSON. |
| Resource cleanup and compatibility | [#2785](https://github.com/artificial-hedge/dipcatcher/pull/2785) protects webhook recovery and deduplication; [#2797](https://github.com/artificial-hedge/dipcatcher/pull/2797) closes backends/releases capacity on failure. [#2755](https://github.com/artificial-hedge/dipcatcher/pull/2755) bounds startup; [#2767](https://github.com/artificial-hedge/dipcatcher/pull/2767) corrects SDK pagination and sampling checks. |
| Research and evidence integrity | All 168 research-wave PRs were source-reviewed and merged into their intended bases. [#2680](https://github.com/artificial-hedge/dipcatcher/pull/2680), [#2681](https://github.com/artificial-hedge/dipcatcher/pull/2681), [#2743](https://github.com/artificial-hedge/dipcatcher/pull/2743) and [#2744](https://github.com/artificial-hedge/dipcatcher/pull/2744) repair registration/generator problems and preserve older work through conflicts. [#2777](https://github.com/artificial-hedge/dipcatcher/pull/2777) addresses concrete guard/platform failures. Unsupported new receipts and inherited integrity bypasses were excluded from the streaming integration. |
| Dependency coverage | #2804 adds the omitted `clients/typescript/fx1` audit command. `make -n audit-js` now produces all four npm audit commands. |

## Checks actually performed

These are overlapping focused checks, so their counts must not be added into a single independent-test total.

| Check | Result |
|---|---|
| Research-wave isolated smoke tests | 2,016 passed across 168 PRs. These use SYNTHETIC fixtures and do not measure predictive or market performance. |
| Composed streaming/state regressions | 51 passed, plus one empty-audit-result regression. |
| Corrected buffered stream audit | 186/186 probes passed in one 49.67-second run. New SYNTHETIC receipt verified; native OpenAPI and normalized surface exactly matched. [Manifest and exact runtime](stream-validation-2026-10-05/validation_manifest.json). |
| Final fault/recovery integration | 49/49 probes passed in 12.50 seconds after validation-helper extraction. The only later probe-source-file change was a nonempty aggregation guard: its regression failed before the fix and passed afterward; all probe implementations stayed unchanged. Seven validation cases and scoped Ruff/format passed. |
| Key recovery/quota component | 50 direct tests, scoped Ruff/format and keys-only mypy passed; real HTTP recovery/administrator replacement/restart smoke passed for damaged-prefix, empty-prefix and zero-byte cases. |
| Idempotency component | 10 HTTP cases, five claim-primitive cases and seven independent cancellation cases passed. Repeated cancellation did not release a claim while its worker was active. |
| Deep validation component | Four failing baseline cases corrected; all seven focused cases passed. Bounded 6,025-byte request returned JSON with request id and did not run the backend. |
| Other focused fixes | Journal (3), usage/expiry/accounting (48), headers/quota (23), usage/scopes (47), enum validation (5), webhook recovery (16), evaluation/resource composition (26), key rotation (31), quickstart (7), Anthropic SDK (14), plus the selected checks detailed in the JSON ledger. |
| Historical CI verification | Seven current CLI smoke checks and the registry four-tag contract passed. All five current GARCH-MIDAS tests passed with both available and exact locked NumPy/SciPy. No duplicate patch was needed for those stale failures. |
| Full repository gates | Full frozen `make sync`, `make lint`, `make typecheck`, `make test`, `make fx1-test` and `make fx1-gate` were not run. Focused Ruff/format/type checks are not a claim that these gates passed. |

## Dependency findings

The initial scan completed at **17:06:40 UTC** on 5 October: **228 exact Python registry pins**, **four npm lockfile trees**, and **33 external Rust registry crates** returned no advisory matches. After #2767, the frozen Python export contained the same 228 pins plus `anthropic==1.11.0` and `docstring-parser==0.18.0`; those two additions were separately checked at **17:55:12 UTC**, also with no matches or skipped pins. The 230 Python pins were covered across two recorded scans, not one later full rescan.

[Initial scan](dependency-audit-2026-10-05.md) · [SDK dependency supplement](dependency-audit-pr2767-2026-10-05.md). GitHub alert feeds, Cargo-specific/yanked-package checks and the partly unpinned Kronos environment remain outside this coverage. No claim is made that the repository is vulnerability-free.

## CI at merge and failure investigation

Of the 183 recorded merges, **119 had a failing workflow**, **38 returned only queued/pending workflows**, **10 returned no workflow runs**, and **15 had other completed/mixed states**. The earliest PR’s CI snapshot was not retained. This comes from the helper’s first page of pull-request-triggered runs; an empty result is not proof that all checks passed.

A [representative failing FX1 run](https://github.com/artificial-hedge/dipcatcher/actions/runs/37206111435) stopped on 11 unused-ignore errors. All 11 annotations were already absent on the initial default branch, so no duplicate fix was invented. The [older 23 September CI failure](https://github.com/artificial-hedge/dipcatcher/actions/runs/35885983271) was checked against current source and targeted tests. This investigation did not inspect all 119 failing logs or rerun Torch-heavy/full-doctor tests.

## Limits and next work

The streaming battery uses a buffered ASGI transport. Real network disconnect propagation, deletion during an active network response and delivery timing remain unverified. The tested complete tree matches [#2801 head 816c8846](https://github.com/artificial-hedge/dipcatcher/commit/816c88467792cf994fa1ca24c303d74304ce6d32), tree `08785154226ecab69e5ce7e87f4b432fb5fe2b95`. The 186-probe run predates the later #2748 changes. Its measured dependency environment is explicitly recorded and is not an exact-lock/full-platform run.

Journal coordination is within one instance. Valid-history rollback or deletion of the complete journal needs an independent trusted anchor; provider completion and durable token charging still have a crash boundary. Recovery tombstones consume the existing key-store capacity. Added quota journaling and recovery compaction were not throughput-benchmarked.

Next maintenance should confirm #2804’s final merge, review subsequent commits and new PRs, exercise real-network SSE cancellation, rotate into older ingestion/execution state code, and continue generator idempotence/validation cleanup. Full quant/FX1 gates and newly completed workflow failures remain follow-up work when the complete environment is available.
