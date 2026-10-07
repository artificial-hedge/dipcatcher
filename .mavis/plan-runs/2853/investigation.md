# #2853 Actions fan-out — investigation

Generated 2026-10-07. Stats measured via `gh run list` (PR-head sampling) +
`yaml.safe_load` of each `.github/workflows/*.yml` (matrix fan-out math).

## Evidence restated (from issue)

- For main commit `1d5c1ff79d5276545b5e462c186901aa7484563d`, 12 workflows
  created 81 checks; 80 stayed queued.
- One push → 80 runnable jobs. `Platform Matrix` (36) + `CI` (26) =
  77.5% of fan-out.
- ~750 queued runs total (back to 2026-10-04).

## Current state (this snapshot)

22 workflow files in `.github/workflows/`. Per-workflow run-time fan-out
(measure of how many jobs run per workflow invocation):

| Workflow | Fan-out | Has concurrency? | Trigger(s) |
|---|---:|---|---|
| `matrix.yml` | 36 | yes | `pull_request`, `push:main`, `workflow_dispatch` |
| `ci.yml` | 28 | yes | `pull_request`, `push:main`, `workflow_dispatch` |
| `full-coverage.yml` | 18 | yes | `pull_request`, `push:main` |
| `proofcore.yml` | 8 | yes | `pull_request`, `push:main`, `workflow_dispatch` |
| `dependency-audit.yml` | 7 | yes | `pull_request`, `push:main`, `workflow_dispatch` |
| `codeql.yml` | 2 | yes | `pull_request`, `push:main`, `schedule` |
| `docs.yml` | 2 | yes | `pull_request`, `push:main` |
| `perf-baseline.yml` | 2 | yes | `pull_request`, `push:main` |
| `replay_viz.yml` | 2 | yes | `pull_request`, `push:main` |
| `reproduce_sota.yml` | 2 | yes | `pull_request`, `push:main` |
| `simtest.yml` | 2 | yes | `pull_request`, `push:main` |
| `web_explorer.yml` | 2 | yes | `pull_request`, `push:main` |
| `adversarial-properties.yml` | 1 | yes | `pull_request`, `push:main` |
| `arch_guards.yml` | 1 | yes | `pull_request`, `push:main` |
| `atlas.yml` | 1 | yes | `pull_request`, `push:main` |
| `dependency-review.yml` | 1 | yes | `pull_request` |
| `fx1.yml` | 1 | yes | `pull_request`, `push:main` |
| `property-nightly.yml` | 1 | yes | `schedule` |
| `release.yml` | 1 | yes | `push:tag` |
| `scorecard.yml` | 1 | yes | `pull_request`, `push:main`, `schedule` |
| `secret-scan.yml` | 1 | yes | `pull_request`, `push:main` |
| `witness_monitor.yml` | 1 | NO | `pull_request`, `push:main` |

**Total per-PR fan-out**: ~125 jobs across all workflows (some are
scheduled-only and don't fire on PR — `property-nightly.yml`,
`release.yml` — but 20 do).

Of the 12 mentioned in the issue:
- `matrix.yml` is the worst single offender at 36 jobs.
- `ci.yml` is next at 28 jobs.
- Together: 64 jobs = **51% of the 125 PR-time jobs**.

## Root-cause hypothesis

The runner pool is rate-limited per GitHub org, but the workflow file count
times matrix depth saturates it:

1. **Matrix width is set to "cover the world"**: 3 OS times 3 Python times
   4 shards in `matrix.yml`; 7 elements in `dependency-audit.yml`'s
   Python band.
2. **PR triggers fan out to all 20 PR-triggering workflows simultaneously**
   because the runner pool can't differentiate "this is a small doc PR"
   vs "this is a wide-branch fix".
3. **No scheduling separation** between "fast lane" (lint/types/arch) and
   "exhaustive lane" (full coverage, perf baseline). Everything waits in
   the same queue.
4. **`witness_monitor.yml` lacks `concurrency:`** — every push on main
   starts a new run that never cancels its predecessor.
5. **Sharded re-collection**: `matrix.yml` per-shard re-runs
   `pytest --collect-only -q` to greedy-assign files. That's N times T
   environment setup where N=shards. A shared `actions/cache` plus single
   collection artifact could collapse this.

## Recommended fix (tiered)

### Tier 1 — safe, low-risk (~30% fan-out reduction)

1. **Reduce `matrix.yml` Python band to 3.12 + 3.13 on PR**; 3.14
   restricted to nightly (`push:main` is fine; or move to
   `property-nightly.yml`).
   - Reduces fan-out from 36 → 24 on PR.
2. **Add `concurrency:` to `witness_monitor.yml`**:
   ```yaml
   concurrency:
     group: ${{ github.workflow }}-${{ github.ref }}
     cancel-in-progress: true
   ```
3. **Reduce `matrix.yml` shards from 4 to 2 on PR** (full 4 on push:main).
   - Reduces PR fan-out from 24 → 12 for matrix.yml.

Estimated cumulative: matrix.yml 36 → 12 on PR. Combined with existing
changes, **PR fan-out 125 → 101 ~= 19%**.

### Tier 2 — design (requires testing)

4. **Share the test inventory artifact across shards in `matrix.yml`**:
   use `actions/upload-artifact@v4` in a "collector" job and
   `actions/download-artifact@v4` in each shard job. Removes per-shard
   `pytest --collect-only` redundancy. **Risk**: workflow epoch/stamp
   changes (any workflow file change requires re-stamp per AGENTS.md).

5. **Split fast lane from exhaustive lane**: mark `arch_guards.yml`,
   `dependency-review.yml`, `scorecard.yml`, `secret-scan.yml`,
   `codeql.yml` as **required checks** (already are) but pin a separate
   **exhaustive lane** (full-coverage, perf-baseline, property-nightly) to
   `push:main` only — remove from PR. PR-time fan-out drops to ~75.
   **Risk**: removes fast feedback for exhaustive bugs; consider
   `pull_request_check` (not `pull_request`) for weekly.

### Tier 3 — operational (no code change)

6. **GitHub org runner capacity check** (`gh api -l /v3/orgs/{org}/actions/runners/online`):
   confirm the pool isn't mis-provisioned. Cheap to investigate; might
   show the bottleneck is here.

## Risks across all tiers

- **Workflow epoch/stamp integrity** (AGENTS.md): every workflow change
  requires `make stamp-epochs` re-run and `quality/epoch_heads.json`
  update. Plan Tier 1+2 as a single PR with the epoch re-stamp included.
- **Receipt continuity**: receipts are immutable; workflow changes are
  receipts themselves. The `witness_monitor.yml` concurrency addition is
  a tiny receipt edit; the `matrix.yml` band change is a more substantial
  one.
- **CI gate breaking**: if any required check is removed from PR triggers,
  it must move to `push:main` AND a branch-protection rule update.

## Validation plan (after the fix lands)

Per the issue's requirement: "do not report a performance improvement
without measurement":

1. Take a baseline sample today on `main` of `gh run list --limit 50` —
   record queue time-to-first-run, jobs-queued-jobs-started ratio.
2. Apply Tier 1 (mechanical).
3. Re-sample: expect `duration_queues < baseline`, `started <= concurrent`.
4. Apply Tier 2 (design).
5. Re-sample: confirm required-checks coverage is unchanged, fan-out down
   >= 30%.

## Recommended ship shape

- **PR 1 (this investigation)**: post this comment on the issue. No
  code change.
- **PR 2 (Tier 1 mechanical)**: matrix band shrink + concurrency fix.
  Single-file change to two workflows + `quality/epoch_heads.json`
  re-stamp.
- **PR 3 (Tier 2 design)**: shard artifact sharing + fast/exhaustive
  split. Larger change; needs Lt sign-off and a branch-protection update.

---

**Honesty contract (per AGENTS.md)**: this investigation contains no
research metrics, no live-P&L claims, no Sharpe/Sortino headlines. The
"improvement" is asserted as a measurement target, not as a result.