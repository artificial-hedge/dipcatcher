# 16 — CI/CD Engineering Excellence

> Lane: **CI/CD ENGINEERING EXCELLENCE.** Status doc — research + audit +
> adoption plan. No workflow, Makefile, or `pyproject.toml` change is made
> here; every snippet below is a proposal to land in a separate PR.
>
> Scope of audit: `.github/workflows/*.yml` (15 workflows), `.github/dependabot.yml`,
> `.github/codeql/`, `Makefile`, `pyproject.toml` (`[tool.pytest]`,
> `[tool.coverage]`, `[dependency-groups]`), `.pre-commit-config.yaml`,
> `.test_durations`.
>
> Convention note (from `AGENTS.md`): fx-1 lanes **commit directly to `main`**;
> cross-cutting repo changes go through PRs. CI parity is defined by
> `.github/workflows/ci.yml` + `fx1.yml` using `uv sync --frozen`. This doc
> respects that model and calls out where "direct-to-main" collides with
> merge-queue/branch-protection best practice.

---

## 1. Verdict up front

The CI/CD posture here is **already near-SOTA on supply-chain security and
parallelism**, and materially **behind SOTA on merge integrity, flaky-test
handling, patch coverage, and workflow DRY-ness**. Concretely:

**Strong (at or above typical OSS SOTA):**

- **100 % of `uses:` references pinned to 40-char commit SHAs** with a trailing
  `# vX.Y.Z` comment (Dependabot-compatible). Verified across all 15 workflows.
- **Least-privilege `permissions:`** on every workflow; `permissions: {}` at the
  top of `release.yml`, `codeql.yml`, `secret-scan.yml`, `dependency-review.yml`,
  `scorecard.yml`, escalating per-job only where required (`id-token: write`,
  `attestations: write`, `security-events: write`).
- **`persist-credentials: false`** on nearly every checkout; fork-safe attestation
  (`if: github.event_name == 'push'`).
- **`concurrency` group + `cancel-in-progress`** and an explicit
  **`timeout-minutes`** on every job.
- **uv caching** via `setup-uv` `enable-cache: true` + `cache-dependency-glob:
  uv.lock` everywhere, plus bespoke `actions/cache` for pip and synthetic lakes.
- **Test sharding two ways**: `ci.yml` (Python 3.12/3.13 × shard 1..4 via
  `pytest-split --splits/--group/--durations-path`) and `matrix.yml` (OS ×
  Python × 4 deterministic greedy file-shards). Coverage is combined from shard
  artifacts into one gate.
- **Full release supply chain**: CycloneDX SBOM, Sigstore signing, SLSA build
  provenance attestation, PyPI **Trusted Publishing (OIDC, no stored token)**
  behind a 3-part gate.
- **Layered scanning**: CodeQL (`security-extended`), gitleaks (SHA-pinned
  binary, full history, redacted), `pip-audit --strict`, bandit,
  dependency-review, OpenSSF Scorecard.
- **Dependabot** across `uv`, `pip`, and `github-actions` ecosystems with grouping.
- **pre-commit mirrors CI** (ruff/mypy/bandit/gitleaks/`uv lock --check`) at
  pinned versions matching `uv.lock`.
- **Determinism**: Hypothesis `ci` profile is `derandomize=True, max_examples=100`.

**Gaps (ranked by leverage):**

| # | Gap | Impact | Fix (section) |
|---|---|---|---|
| G1 | No `merge_group` trigger / merge queue / branch-protection ruleset | The tree that passes PR checks ≠ the tree that merges; direct-to-`main` bypasses gates | §4.1 |
| G2 | Pin drift (`atlas.yml` on `setup-uv v5`; all others `v10.2.0`) + no automated drift gate | A stale/unvetted action can slip in silently | §4.2 |
| G3 | No flaky-test detection, retry, or quarantine; `.test_durations` never refreshed | Red builds from nondeterminism; shard imbalance grows over time | §4.4 |
| G4 | Coverage is **global-only** (`fail_under`); no **patch/diff** coverage gate | A PR can add uncovered lines while the global % still passes | §4.3 |
| G5 | Massive duplication: 35× `uv sync --frozen`, 19 checkouts in `ci.yml`, setup preamble copy-pasted across 11 files | Drift already materialized (G2); high maintenance cost | §4.5 |
| G6 | `--frozen` (installs stale lock silently) instead of `--locked` in install jobs | Lockfile/pyproject mismatch only caught in the one `audit` job | §4.6 |
| G7 | No `cache-suffix` despite differing resolutions (`--all-groups` vs `--all-extras` vs `--group dev`); no `prune-cache` | Cache-key collisions / bloated caches | §4.6 |
| G8 | No version→tag guard, no generated changelog/release notes | `fx1.__version__` (semver source of truth) can drift from the pushed `v*.*.*` tag | §4.7 |
| G9 | No `GITHUB_STEP_SUMMARY`; Scorecard/CodeQL not enforced as required checks | Results buried in logs; security scans advisory-only | §4.8 |

---

## 2. SOTA practice summaries (2024–2026) with citations

### 2.1 Concurrency, queues, and merge integrity

- **Concurrency groups.** `concurrency.group` bounds a workflow to one active run
  per key; a new run cancels the pending one by default. The newer
  **`concurrency.queue: max`** (up to 100 queued runs) lets builds *queue*
  instead of cancel — useful for `main`/release refs where cancelling is
  destructive. Allowed expression contexts are `github`, `inputs`, `vars`,
  `needs`, `strategy`, `matrix`. [1]
- **Merge queue + `merge_group`.** A merge queue re-tests each PR **combined with
  the base branch and the PRs ahead of it** before merging, so what merges is
  what passed. Required checks **must** be reported on the `merge_group` event —
  a workflow triggered only on `pull_request`/`push` will **not** satisfy a
  merge-queue required check and the merge fails. A `push` still fires for the
  merge-group branch but does not carry the base-branch target. Add
  `merge_group: branches: [main]` alongside `pull_request`. The queue can be set
  to "only merge non-failing PRs" or to tolerate intermittent failures. [2][3][4]
- **Best practice:** keep `cancel-in-progress: true` for PR refs (cheap
  superseding) but set **`cancel-in-progress: false`** for `main` and release
  refs so an in-flight release/post-merge run is never cancelled mid-flight.

### 2.2 uv caching and dependency locking

- **`setup-uv` caching.** `enable-cache` defaults to `auto` (on for hosted
  runners, **off** for `release`/tag-push/`pull_request_target`/`workflow_run`
  to avoid poisoned caches). `cache-dependency-glob` controls invalidation
  (default already includes `**/pyproject.toml` and `**/uv.lock`). Two inputs
  this repo does **not** use yet: **`cache-suffix`** (disambiguates jobs that
  share dependency files but resolve differently — otherwise the first uploader
  wins the key and others fail to save) and **`prune-cache`** (runs
  `uv cache prune --ci`, dropping re-downloadable pre-built wheels before save;
  can shrink a 137 MB cache to KB when all deps ship wheels). [5][6]
- **`--locked` vs `--frozen`.** They are near-opposites: **`uv sync --locked`**
  *fails* if `uv.lock` is out of date vs `pyproject.toml` (the CI guard you
  want); **`uv sync --frozen`** skips the check and installs against a possibly
  stale lock without comment. Best practice for gates is `--locked`, or `--frozen`
  **plus** an explicit `uv lock --check` that runs before install. [7][8]

### 2.3 Reusable workflows vs composite actions (DRY)

- **Composite action** = a bundle of *steps* that runs as **one step** inside a
  caller job; fast startup, no secrets, no `if:`/jobs, logs collapsed to one
  step. Ideal for the repeated "checkout → setup-uv → sync" preamble. [9][10]
- **Reusable workflow** (`on: workflow_call`) = whole *jobs/pipelines* called as
  a job; supports multiple jobs, `secrets`, `if:`, per-step live logs, up to 4
  nesting levels. Ideal for the shared sharded-test or lint pipeline. [9][11]
- **Rule of thumb:** composite for setup/caching/notification steps; reusable for
  an entire multi-job pipeline that must run on a specific runner or take
  secrets. Centralizing third-party action usage behind one reviewed composite /
  reusable that itself pins to vetted SHAs is a recommended pattern to shrink the
  supply-chain surface and make bumps a single diff. [12][13]

### 2.4 Matrix strategies and test sharding / xdist tuning

- **`pytest-xdist` distribution.** `-n auto` scales to CPUs (`-n logical` to
  logical cores). `--dist` modes: `load` (default, unordered), `loadscope`
  (group by module/class), **`loadfile`** (all tests in a file on one worker —
  what this repo uses; good when files share expensive fixtures), `loadgroup`
  (`xdist_group` marker), and **`worksteal`** (idle workers steal from busy ones;
  better than `load` when durations vary widely while still reusing fixtures).
  `--maxprocesses` caps workers; `--max-worker-restart` bounds crash restarts.
  Avoid `--pdb`/`--looponfail` under xdist (auto-disables distribution /
  deprecated). [14]
- **`pytest-split` sharding.** Store durations once (`--store-durations
  --durations-path .test_durations`), then run `--splits N --group k`. Tests
  absent from the durations file get the *average* known duration, so staleness
  degrades gracefully — but balance drifts as the suite changes.
  **`--clean-durations`** removes entries for deleted tests and is the
  recommended way to regenerate; run it on a schedule (weekly) and merge the
  refreshed file. `--splitting-algorithm` (default `duration_based_chunks`)
  trades balance vs locality. [15][16]

### 2.5 Coverage enforcement (diff/patch coverage)

- **Global vs patch.** A single project-wide threshold lets a PR merge uncovered
  new code as long as the average holds. SOTA adds a **patch/diff-coverage**
  gate that holds only the *changed lines* to a threshold. [17][18]
- **In-runner options (no data leaves CI):** **`diff-cover`** (and wrappers like
  `DavidDeSloovere/diff-cover-action`, `swantron/difftron`) intersect `git diff`
  with a Cobertura/lcov report and fail the PR below `--fail-under`; they post
  idempotent PR comments + inline annotations + step summaries, and skip
  gracefully on fork PRs (read-only token). `diff-cover-action` also wraps
  `diff-quality` for ruff/mypy/eslint on changed lines. Needs `fetch-depth: 0` to
  diff against base. [17][18]
- **SaaS option:** Codecov/Coveralls `codecov/patch` status with
  `target-patch` (default 80) enforced via branch protection; choose when you
  want historical trend graphs and org dashboards. [19]

### 2.6 Flaky-test detection and quarantine

- **Detection is a property of history, not a single run.** The strongest signal
  is **pass-on-retry within the same run** (same code, near-identical env); the
  next is **variance across identical commits** (same SHA passes in one run,
  fails in another). Collect per-run results keyed by commit SHA, compute a flake
  rate per test, and flag tests whose outcome varies on unchanged code. At scale
  this is automated (e.g. Atlassian's Flakinator over 350 M executions/day). [20][21]
- **Quarantine, don't delete.** Move known-flaky tests out of the *blocking*
  path into a non-blocking suite so they still run (and keep collecting evidence)
  but stop failing the gate. Track owner, failure rate, and an expiry. [20][22][23]
- **SLA + "fix or remove".** GitLab: fast quarantine ≤ 3 days → long-term ≤ 3
  months → auto-delete. Microsoft: a flaky test not fixed within ~2 weeks is
  removed (credited with ~18 % flakiness reduction in 6 months). Quarantined
  tests that *break* (not just flake) should be treated as urgent. [22][23]
- **Tooling.** `pytest-rerunfailures` (`--reruns N`, per-test
  `@pytest.mark.flaky(reruns=..)`, `--force-reruns`, `--reruns-mode=append`) is
  xdist-compatible and can rerun crashed workers; `xfail(strict=False)` is a
  lightweight manual quarantine. Reruns mask flakes — pair them with a flake
  registry so retries are *measured*, not silently absorbed. [24]

### 2.7 Supply-chain CI security

- **Pin by SHA.** Tags (`@v4`) are mutable and have been abused (the 2022/2024
  `tj-actions` / `reviewdog` compromises). Pin every `uses:` to the **full
  40-char SHA** with the semver in a comment so Dependabot/Renovate can still
  bump; never a short SHA. Enforce with a PR check
  (`zgosalvez/github-actions-ensure-sha-pinned-actions`) and a linter. [12][25][26]
- **Lint/audit the workflows themselves.** `actionlint` (YAML/expression/permission
  errors) and **`zizmor`** (Actions-specific security audit: unpinned refs,
  `pull_request_target` misuse, injection, excessive permissions) run in CI as a
  pre-merge gate; OpenSSF **Scorecard** scores the repo (its `Pinned-Dependencies`
  check is what SHA-pinning satisfies). [25][26][27]
- **2026 Actions security roadmap (forward-looking).** GitHub is adding a
  **`dependencies:` block in workflow YAML** that locks all direct + transitive
  action/composite deps by commit SHA (deterministic runs, hash-mismatch
  fail-fast, composite nested deps no longer hidden), **scoped secrets** bound to
  explicit contexts (no implicit inheritance into reusable workflows), and
  **workflow execution protections** built on rulesets (actor/event allow-lists;
  restrict `pull_request_target`/`workflow_dispatch`). Plan for these now by
  keeping pins + least privilege + allow-listed actions. [28][29]

### 2.8 Release automation (semver from `fx1.__version__`, changelogs)

- **Single source of truth.** `docs/FX1_API_STABILITY.md` makes **`fx1.__version__`**
  canonical semver, read by hatch via `[tool.hatch.version] path =
  "src/fx1/__init__.py"` — the distribution version cannot drift from the module.
- **python-semantic-release (PSR).** Derives the next version from
  **Conventional Commits**, stamps it into project files, generates a Jinja
  changelog (with `exclude_commit_patterns` to drop `ci`/`test`/`chore` noise),
  posts release notes to the VCS, and ships a GitHub Action. Fits this repo's
  existing "terse conventional-commit style." [30][31]
- **Guard even without PSR:** a CI job that reads `fx1.__version__` and asserts
  the pushed tag equals `v{version}` closes the "tag ≠ module version" hole;
  `softprops/action-gh-release` (SHA-pinned) can create the GitHub Release from
  the SBOM/checksums the `release.yml` already produces.

---

## 3. Audit of current workflows

### 3.1 Inventory

| Workflow | Triggers | Notable jobs | State |
|---|---|---|---|
| `ci.yml` | push(main), PR, schedule(full), dispatch | lint, audit, package(SLSA), container, test(2×4 shard), coverage(combine), perf, smoke, mc-engine, rust-accel, market-sim, parity, examples, audit-obs, diffbacktest, robustness, formal, pretrade, stress | Rich; heavy duplication |
| `fx1.yml` | push(main, fx-1/**), PR | lint, mypy, tests, honesty gate, corpus smoke | Clean, single job |
| `matrix.yml` | PR, push(main), dispatch | OS×Python×4 file-shard | Deterministic greedy sharding |
| `release.yml` | tag `v*.*.*`, PR, dispatch | build, SBOM, Sigstore, SLSA, Trusted-Publish | Exemplary |
| `codeql.yml` | push, PR, schedule | Python `security-extended` | Clean |
| `secret-scan.yml` | push, PR | gitleaks (SHA-pinned binary, full history) | Clean |
| `dependency-review.yml` | PR | dependency-review + graph gate | Clean |
| `scorecard.yml` | push, PR, schedule, branch_protection_rule | OpenSSF Scorecard → SARIF | Clean |
| `docs.yml` | push(main), PR | mkdocs `--strict` → Pages | Clean |
| `proofcore.yml` | push(main), PR | layering, leakage-scan(warn), reality-filter, per-pkg coverage floors, fx1 coverage ratchet | Clean |
| `adversarial-properties.yml` | push(main), PR | Hypothesis `ci` profile | Clean |
| `property-nightly.yml` | schedule, dispatch | Hypothesis `nightly` profile | Clean |
| `simtest.yml` | PR, dispatch | deterministic sim swarm | Clean |
| `web_explorer.yml` | push/PR (path-filtered) | vitest + Playwright | Clean; **path-filtered** ✓ |
| `replay_viz.yml` | push/PR (path-filtered) | Playwright + fixture tests | Clean; **path-filtered** ✓ |

### 3.2 What is already correct (do not regress)

- SHA pinning with semver comments — **all** `uses:` lines (verified).
- Least-privilege + `permissions: {}` escalation model.
- `concurrency` + `timeout-minutes` on every job; path filters on the two web
  lanes; `HYPOTHESIS_PROFILE=ci` derandomized.
- Combined-coverage pipeline (shard artifacts → `coverage combine` → threshold in
  `[tool.coverage.report] fail_under`, the single source of truth; no inline
  `--cov-fail-under` drift between CI and `make coverage`).
- Release supply chain (SBOM + Sigstore + SLSA + OIDC Trusted Publishing) and the
  3-part PyPI gate with **no stored token**.
- Dependabot on all three ecosystems with grouping; pre-commit parity with
  `uv.lock` versions.

### 3.3 Findings (evidence-backed)

- **G1 — No merge integrity.** No workflow has a `merge_group` trigger; there is
  no CODEOWNERS and no branch-protection/ruleset is asserted (the `scorecard.yml`
  header itself notes rulesets are "inconclusive until the owner adds" one). With
  fx-1 lanes committing **directly to `main`**, required checks can be bypassed
  and the merged tree is not the tested tree. This is the single highest-leverage
  gap. [2][3]
- **G2 — Pin drift, no drift gate.** `atlas.yml` pins
  `astral-sh/setup-uv@d4b2f3b6…  # v5`; every other workflow pins
  `…c18668ad…  # v10.2.0`. Nothing in CI fails on an out-of-date or unpinned
  ref (no `actionlint`, no `zizmor`, no `ensure-sha-pinned`). This is exactly the
  class of drift a composite action + a linter gate prevents. [12][25]
- **G3 — No flaky handling; stale shard durations.** No `pytest-rerunfailures`,
  no `--reruns`, no quarantine marker/suite, no flake registry. `.test_durations`
  is committed and consumed by `ci.yml` (`--durations-path`) but **never
  regenerated** (`--store-durations`/`--clean-durations` appear nowhere), so shard
  balance silently drifts as tests are added/removed. [15][20][24]
- **G4 — Global coverage only.** The gate is `[tool.coverage.report] fail_under`
  (committed value `80`; note the working tree currently shows `70` — itself a
  drift signal). `proofcore.yml` adds *per-package floors* and an *fx1 ratchet*
  (`--cov-fail-under=60`), which is good, but there is **no patch/diff-coverage**
  gate: a PR touching hot paths can ship uncovered new lines while the global %
  holds. [17][18]
- **G5 — Duplication.** `uv sync --frozen` appears **35×**; `ci.yml` alone has
  **19** checkouts; the `checkout → setup-uv(python, enable-cache,
  cache-dependency-glob) → sync` preamble is copy-pasted across 11 workflows.
  Every copy is a place pins/flags can (and did — G2) diverge. [9][10]
- **G6 — `--frozen` in install jobs.** Most gates run `uv sync --frozen`, which
  installs a stale lock silently; only `ci.yml`'s `audit` job and `docs.yml` run
  `uv lock --check`. Best practice is `--locked` in the install step (or
  `uv lock --check` before *every* install). [7][8]
- **G7 — Cache keys can collide / bloat.** Jobs resolve differently
  (`--all-groups`, `--all-groups --all-extras`, `--group dev`, `--only-group
  docs`, `--extra jax`) but none set `cache-suffix`, and none set `prune-cache`.
  `setup-uv` warns that same-glob/different-resolution jobs collide on the cache
  key (first save wins; others fail to upload). [5][6]
- **G8 — Release version/changelog not automated.** `release.yml` is excellent at
  *packaging* but nothing derives the tag from `fx1.__version__` or verifies
  `v{tag} == v{fx1.__version__}`; `CHANGELOG.md` is hand-maintained (Keep a
  Changelog) and no GitHub Release/notes are produced. [30]
- **G9 — Low signal surface.** No `GITHUB_STEP_SUMMARY` anywhere (coverage %,
  perf-regression flags, flake counts, Scorecard summary stay in raw logs);
  Scorecard/CodeQL publish SARIF but are not wired as **required** checks.
- **G10 — `cancel-in-progress: true` on release/main.** `release.yml` uses a
  dedicated group but still cancels in-progress runs; a superseding tag push
  could cancel a release mid-flight. Prefer `false` for release and `main`
  post-merge runs (`concurrency.queue: max` where queueing is desired). [1]

---

## 4. Adoption plan (concrete snippets)

Sequenced by leverage ÷ risk. Each is independently mergeable and additive — none
weakens an existing gate or the honesty contract. **These are proposals; land
each in its own PR.**

### 4.1 (G1) Merge queue + `merge_group` + ruleset — highest leverage

Add the `merge_group` trigger to the required-check workflows so a merge queue
can gate on them. For fx-1 lanes that commit directly to `main`, either (a) route
them through the queue too, or (b) keep direct-push but add a **post-merge
`push`-triggered run** (already present) *and* a ruleset that requires the check
on the PR path for cross-cutting changes.

```yaml
# ci.yml / fx1.yml / matrix.yml — add merge_group alongside pull_request
on:
  pull_request:
    branches: [main]
  merge_group:               # NEW: lets a merge queue report required checks
    branches: [main]
  push:
    branches: [main]
```

Then, in repo **Settings → Rules → Rulesets** (or `gh api`), for `main`:
require the checks (`test`, `coverage`, `lint`, `fx1/test`, `scorecard`,
`codeql`), enable **"Require a merge queue"**, and (recommended)
**"Require branches to be up to date"**. Merge-queue behaviour ("only merge
non-failing PRs") can be relaxed if G3's flake handling lands first. [2][3][4]

```bash
# one-time, owner-run: enforce the merge queue + required checks on main
gh api -X POST repos/:owner/:repo/rulesets -f name=main-gates \
  -f target=branch --input .github/rulesets/main.json   # see GitHub rulesets API
```

### 4.2 (G2) Workflow linting + pin-drift gate

A small pre-merge job that fails on unpinned refs, drift, and common
misconfigurations. This is what would have caught `atlas.yml`'s `v5` pin.

```yaml
# .github/workflows/ci-lint.yml  (new)
name: ci-lint
on:
  pull_request:
    paths: [".github/**", "**/action.yml"]
  merge_group:
    branches: [main]
permissions:
  contents: read
concurrency:
  group: ci-lint-${{ github.ref }}
  cancel-in-progress: true
jobs:
  lint:
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1
        with:
          persist-credentials: false
      # actionlint: YAML/expression/permission correctness (resolve current SHA + tag comment)
      - uses: rhysd/actionlint@<pin-sha> # v1.x.y
      # zizmor: Actions security audit (unpinned refs, injection, excess perms)
      - uses: zizmorcore/zizmor-action@<pin-sha> # vX.Y.Z
      # fail if any third-party action is not SHA-pinned
      - uses: zgosalvez/github-actions-ensure-sha-pinned-actions@<pin-sha> # vX.Y.Z
```

Also add a Dependabot-driven **consistency check**: a tiny script (or the
`ensure-sha-pinned` action's report) that asserts every `astral-sh/setup-uv` pin
is identical across workflows — the composite action in §4.5 makes this moot by
having exactly one pin site. [12][25][26]

### 4.3 (G4) Patch/diff-coverage gate

Runs in-runner; no coverage data leaves CI; skips gracefully on fork PRs.

```yaml
# add a job to ci.yml, needs: [coverage]
diff-cover:
  needs: [coverage]
  if: ${{ !cancelled() && github.event_name != 'push' }}
  runs-on: ubuntu-latest
  timeout-minutes: 15
  permissions:
    contents: read
    pull-requests: write        # for the idempotent PR comment
  steps:
    - uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1
      with:
        fetch-depth: 0          # required to diff against the base branch
    - uses: actions/download-artifact@3e5f45b2cfb9172054b4087a40e8e0b5a5461e7c # v8.0.1
      with:
        name: dipcatcher-coverage
        path: .
    - uses: astral-sh/setup-uv@c18668ad3cf93ea998bef934396af7bb5c839dc7 # v10.2.0
      with: { python-version: "3.12" }
    - name: Diff coverage on changed lines
      run: |
        set -euo pipefail
        uvx diff-cover coverage.xml \
          --compare-branch=origin/${{ github.base_ref || 'main' }} \
          --fail-under=85 \
          --markdown-report "$GITHUB_STEP_SUMMARY"
```

Equivalent SaaS path: Codecov `codecov/patch` with `target-patch: 85` enforced
via the §4.1 ruleset (adds trend graphs/org dashboards). Keep the *global*
`fail_under` as-is; the patch gate is additive. [17][18][19]

### 4.4 (G3) Flaky detection, retry, and quarantine

Three parts: measure, contain, and keep shard durations fresh.

```toml
# pyproject.toml [dependency-groups] dev  (proposal)
#   "pytest-rerunfailures>=15",
```

```yaml
# ci.yml test job — bounded retry + flake signal to the step summary
- name: Pytest
  run: |
    set -euo pipefail
    args=( -q -n auto --dist worksteal -m "$marker"
           --splits 4 --group "${{ matrix.shard }}"
           --durations-path .test_durations
           --reruns 2 --reruns-delay 1 )        # NEW: mask transient flakes, keep the signal
    uv run pytest "${args[@]}"
```

Quarantine (non-blocking, still runs, still collects evidence): a
`@pytest.mark.flaky_quarantine` marker + a nightly job that runs quarantined
tests with `-p no:randomly`, reports the flake rate to `GITHUB_STEP_SUMMARY`, and
files/updates a tracking issue. Enforce an SLA (GitLab-style: fix/remove within
~2 weeks) in the tracking file so quarantine never becomes permanent. [20][22][23][24]

```yaml
# .github/workflows/refresh-durations.yml  (new, weekly) — keeps shards balanced
name: refresh-durations
on:
  schedule: [{ cron: "40 3 * * 1" }]
  workflow_dispatch:
permissions:
  contents: write
  pull-requests: write
jobs:
  refresh:
    runs-on: ubuntu-latest
    timeout-minutes: 60
    steps:
      - uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1
      - uses: astral-sh/setup-uv@c18668ad3cf93ea998bef934396af7bb5c839dc7 # v10.2.0
        with: { python-version: "3.12", enable-cache: true }
      - run: uv sync --locked --all-groups --all-extras
      - name: Regenerate .test_durations (--clean-durations drops deleted tests)
        run: uv run pytest -n auto -m "not network and not slow" \
               --store-durations --clean-durations --durations-path .test_durations
      - name: Open PR if changed
        run: |
          set -euo pipefail
          git diff --quiet -- .test_durations && exit 0
          git checkout -b chore/refresh-test-durations
          git add .test_durations
          git commit -m "ci: refresh pytest-split durations"
          git push -f origin chore/refresh-test-durations
          gh pr create --fill || true
        env: { GH_TOKEN: "${{ github.token }}" }
```

Also switch `--dist loadfile` → **`--dist worksteal`** in the long shards: better
balance under wide duration spread while still reusing fixtures. [14][15][16]

### 4.5 (G5) Composite action + reusable workflow (DRY, one pin site)

Fold the repeated preamble into a **composite action** so there is exactly one
`setup-uv` pin and one sync policy; convert the sharded test lane into a
**reusable workflow** both `ci.yml` and `matrix.yml` can call.

```yaml
# .github/actions/uv-setup/action.yml  (new composite)
name: uv-setup
description: Checkout-free uv setup with cache-suffix and lock policy
inputs:
  python-version: { default: "3.12" }
  sync-args:      { default: "--all-groups --all-extras" }
  cache-suffix:   { default: "" }
runs:
  using: composite
  steps:
    - uses: astral-sh/setup-uv@c18668ad3cf93ea998bef934396af7bb5c839dc7 # v10.2.0
      with:
        python-version: ${{ inputs.python-version }}
        enable-cache: true
        cache-dependency-glob: uv.lock
        cache-suffix: ${{ inputs.cache-suffix }}   # G7: disambiguate resolutions
        prune-cache: true                          # G7: shrink cache before save
    - name: Sync (locked)
      shell: bash
      run: uv sync --locked ${{ inputs.sync-args }} # G6: fail on stale lock
```

```yaml
# usage in any job — one line replaces checkout+setup+sync duplication
- uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1
  with: { persist-credentials: false }
- uses: ./.github/actions/uv-setup
  with:
    python-version: ${{ matrix.python }}
    cache-suffix: "py${{ matrix.python }}-${{ matrix.shard }}"
```

For the whole sharded-test pipeline, promote it to a reusable workflow
(`on: workflow_call`) that `ci.yml` invokes with the matrix as an input — so
`matrix.yml` and `ci.yml` share one implementation. Reusable (not composite)
because it owns jobs, the runner, and can take `secrets`. [9][10][11][13]

### 4.6 (G6/G7) Lock policy + cache hygiene

- Replace `uv sync --frozen` with **`uv sync --locked`** in install/gate jobs (or
  keep `--frozen` but run `uv lock --check` immediately before every sync, not
  only in `audit`). `--locked` fails fast on a `pyproject.toml`/`uv.lock`
  mismatch instead of installing stale. [7][8]
- Add **`cache-suffix`** per distinct resolution (`--all-extras` vs `--group dev`
  vs `--extra jax` vs `--only-group docs`) and **`prune-cache: true`** on hosted
  runners. Both are inputs to the composite in §4.5, so this is a one-place
  change. [5][6]

### 4.7 (G8) Version guard + changelog/release automation

Cheapest safe step first — a **tag ↔ `fx1.__version__` guard** in `release.yml`:

```yaml
# release.yml build job — assert the pushed tag equals the module version
- name: Verify tag matches fx1.__version__
  run: |
    set -euo pipefail
    mod_version="$(uv run --no-project python -c 'import sys; sys.path.insert(0,"src"); import fx1; print(fx1.__version__)')"
    tag="${GITHUB_REF#refs/tags/v}"
    if [ "$tag" != "$mod_version" ]; then
      echo "::error::tag v${tag} != fx1.__version__ ${mod_version}"; exit 1
    fi
```

Fuller automation with **python-semantic-release** (derives the bump from the
existing conventional-commit style, stamps `src/fx1/__init__.py`, regenerates
`CHANGELOG.md`, and posts GitHub Release notes). Configure in `pyproject.toml`:

```toml
# pyproject.toml (proposal)
[tool.semantic_release]
version_toml = ["src/fx1/__init__.py:__version__"]
commit_parser = "conventional"
tag_format = "v{version}"
[tool.semantic_release.changelog]
mode = "update"                       # prepend into the existing CHANGELOG.md
exclude_commit_patterns = [
  '''^ci(\(.*\))?:''', '''^test(\(.*\))?:''', '''^chore(\(.*\))?:''',
  '''^docs(\(.*\))?:''', '''^style(\(.*\))?:''',
]
```

Keep PSR's publish path **behind the existing 3-part PyPI gate** — PSR computes
version/changelog/tag; `release.yml` still owns SBOM/Sigstore/SLSA/Trusted
Publishing. This preserves the "no stored token" property. [30][31]

### 4.8 (G9) Surface results + enforce security checks

- Write coverage %, the perf-regression verdict, flake counts, and the Scorecard
  summary to **`GITHUB_STEP_SUMMARY`** so they show in the run UI instead of raw
  logs (the §4.3 diff-cover and §4.4 flake jobs already do this).
- Make **Scorecard** and **CodeQL** **required checks** in the §4.1 ruleset so
  the security scans are blocking, not advisory; add a **CODEOWNERS** for
  `.github/workflows/` so workflow changes need review (mitigates G2/G5 drift at
  the human layer).

---

## 5. Suggested landing order

1. **§4.5 composite action + §4.6 lock/cache hygiene** — low risk, removes the
   duplication that caused G2, one pin site. (Cross-cutting → PR.)
2. **§4.2 ci-lint gate** — catches pin drift/regressions from here on.
3. **§4.4 durations refresh + `--dist worksteal` + bounded reruns** — cheap CI
   reliability win; prerequisite for tolerating flakes in a merge queue.
4. **§4.3 diff-cover** — additive coverage gate (start informational, then
   `--fail-under`).
5. **§4.7 version guard**, then PSR/changelog automation.
6. **§4.1 merge queue + ruleset + §4.8 required checks/CODEOWNERS** — owner-run,
   highest leverage, do last once checks are stable and fast.

---

## 6. References

[1] GitHub Docs — *Control the concurrency of workflows and jobs* (incl.
`concurrency.queue: max`): https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/control-workflow-concurrency
[2] GitHub Docs — *Managing a merge queue*: https://docs.github.com/repositories/configuring-branches-and-merges-in-your-repository/configuring-pull-request-merges/managing-a-merge-queue
[3] GitHub Changelog — *Merge group webhook event and Actions trigger*: https://github.blog/changelog/2022-08-18-merge-group-webhook-event-and-github-actions-workflow-trigger/
[4] GitHub Docs — *Troubleshooting required status checks* (merge_group): https://docs.github.com/en/pull-requests/how-tos/merge-and-close-pull-requests/troubleshooting-required-status-checks
[5] `astral-sh/setup-uv` — caching docs (`enable-cache: auto`, `cache-suffix`, `prune-cache`, `cache-dependency-glob`): https://github.com/astral-sh/setup-uv/blob/main/docs/caching.md
[6] pydevtools — *How to Cache uv Dependencies in CI* (`uv cache prune --ci`): https://pydevtools.com/handbook/how-to/how-to-cache-uv-dependencies-in-ci.md
[7] uv Python cheatsheet — *`--locked` vs `--frozen`*: https://dev.to/extractdata/uv-python-cheatsheet-what-changed-in-012-and-what-still-trips-you-up-5b38
[8] uv — *GitHub Actions integration* (`uv sync --locked`): https://mintlify.wiki/astral-sh/uv/integrations/github-actions
[9] GitHub Blog — *How to start using reusable workflows*: https://github.blog/developer-skills/github/using-reusable-workflows-github-actions/
[10] GitHub Docs — *Reusing workflow configurations* (reusable vs composite): https://github.com/github/docs/blob/main/content/actions/concepts/workflows-and-actions/reusing-workflow-configurations.md
[11] GitHub Docs — *Reusing workflows* (jobs/secrets/nesting limits): https://docs.github.com/en/actions/sharing-automations/reusing-workflows
[12] StepSecurity — *Pinning GitHub Actions for Enhanced Security*: https://www.stepsecurity.io/blog/pinning-github-actions-for-enhanced-security-a-complete-guide
[13] Sourcetrail — *Practical Security Hardening for GitHub Actions* (centralize third-party actions behind reviewed composites): https://www.sourcetrail.com/javascript/practical-security-hardening-for-github-actions/
[14] `pytest-xdist` — *Running tests across multiple CPUs* (`--dist loadfile/loadgroup/worksteal`, `--maxprocesses`, `--max-worker-restart`): https://pytest-xdist.readthedocs.io/en/latest/distribution.html
[15] `pytest-split` — README (`--store-durations`, `--clean-durations`, `--durations-path`, splitting algorithms): https://github.com/jerry-git/pytest-split/
[16] `pytest-split` issue #20 — *Using `--store-durations` in GitHub Actions* (combine shard durations): https://github.com/jerry-git/pytest-split/issues/20
[17] GitHub Marketplace — *Diff Cover Action* (`DavidDeSloovere/diff-cover-action`): https://github.com/marketplace/actions/diff-cover-action
[18] GitHub Marketplace — *Difftron Delta Coverage Gate*: https://github.com/marketplace/actions/difftron-delta-coverage-gate
[19] `getsentry/codecov-action` — `target-patch` / `codecov/patch` status: https://github.com/getsentry/codecov-action/blob/main/README.md
[20] Qualflare — *Flaky test detection* (history/pass-on-retry scoring): https://qualflare.com/flaky-test-detection/
[21] TestDino — *Flaky Test Benchmark Report 2026* (Atlassian Flakinator; Microsoft 2-week rule): https://testdino.com/blog/flaky-test-benchmark
[22] Trunk — *Eradicating flaky tests* (quarantine, SLA): https://trunk.io/blog/eradicating-flaky-tests
[23] GitLab Handbook — *Test Quarantine Process* (3-day / 3-month / auto-delete): https://handbook.gitlab.com/handbook/engineering/testing/quarantine-process/
[24] `pytest-rerunfailures` — README (`--reruns`, `--force-reruns`, `--reruns-mode=append`, `flaky` mark, xdist compatibility): https://github.com/pytest-dev/pytest-rerunfailures/blob/master/README.rst
[25] safeguard.sh — *Pin GitHub Actions to SHAs: A Practical Guide* (`ensure-sha-pinned-actions`, actionlint): https://safeguard.sh/resources/blog/how-to-pin-github-actions-to-shas-correctly
[26] freeCodeCamp — *How to Prevent Poisoned GitHub Actions Dependencies* (actionlint PR gate): https://www.freecodecamp.org/news/how-to-prevent-poisoned-github-actions-dependencies/
[27] NearForm — *Why You Should Pin Your GitHub Actions by commit-hash* (OpenSSF Scorecard `Pinned-Dependencies`): https://nearform.com/insights/why-you-should-pin-your-github-actions-by-commit-hash/
[28] GitHub Blog — *What's coming to our GitHub Actions 2026 security roadmap* (`dependencies:` SHA locking, scoped secrets, execution protections): https://github.blog/news-insights/product-news/whats-coming-to-our-github-actions-2026-security-roadmap/
[29] GitHub Docs — *Workflow execution protections* (ruleset-based actor/event allow-lists): https://docs.github.com/en/enterprise-cloud@latest/admin/enforcing-policies/enforcing-policies-for-your-enterprise/actions-policies/workflow-execution-protections
[30] Python Semantic Release — docs (Conventional Commits, changelog templates, GitHub Action): https://python-semantic-release.readthedocs.io/en/stable/
[31] Python Semantic Release — *Configuration* (`version_toml`, `commit_parser`, `exclude_commit_patterns`, `tag_format`): https://python-semantic-release.readthedocs.io/en/latest/configuration/configuration.html
