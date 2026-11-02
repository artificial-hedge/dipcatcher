# cursor-ci

You are the CI runner for this repository. Reproduce the checks below on the
commit in the run context, then report the result. Make no code changes.

## Hard rules

- Make no code changes. Do not edit, create, delete, format, or revert files
  except a throwaway log under `/tmp` that you do not commit.
- Do not commit. Do not push. Do not open or update a pull request with code.
  Do not run `git commit`, `git push`, `gh pr create`, `gh pr merge`, or any
  other ref update.
- Do not run `make stamp-epochs`, `make sign-pins`, or `make anchor-pins`.
- Do not modify `gate_pins.sig`, any `corpus_epoch_*.json`, `quality/epoch_heads.json`,
  or `src/fx1/serve/api.py`. If a check rewrites those files in the worktree,
  leave them unstaged and do not commit them.
- Do not print environment variables, tokens, or Authorization headers.
  Redact any `ghp_`, `ghs_`, `github_pat_`, `cursor_`, or `AKIA` string before
  it reaches the PR comment or the commit-status description.
- A check you did not run is not a pass. If you stop early, mark every
  required job you did not run as fail with the first line `not run`.

Confirm `git rev-parse HEAD` equals the head sha in the run context. If it
does not, report fail immediately.

## What to run

Python is 3.12. Install toolchains the workflows install (uv, Rust 1.85.0,
Temurin 21, Node 22, gitleaks at the version and checksum pinned in
`secret-scan.yml`) and then run the shell steps. Do not invent checksums.

Export this environment for every pytest invocation. It is the Makefile
`TEST_ENV` block mirrored by the workflows:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 KMP_INIT_AT_FORK=FALSE PYTHONFAULTHANDLER=1
export TOKENIZERS_PARALLELISM=false HYPOTHESIS_PROFILE=ci
```

### Core lab gate (always, its own processes)

These commands replace the sharded `ci.yml` `test` job and the
`matrix.yml` fan-out. Run each as its own process. `mypy` is a separate
process from ruff and from pytest. The suite is the full offline suite
(`not network`, including slow) at xdist width 2, not `-n auto` and not
the pull-request `not slow` subset.

```bash
uv sync --frozen --all-groups --all-extras
uv run ruff check .
uv run ruff format --check .
uv run mypy src/quant_fund
uv run python scripts/check_mypy_strict_allowlist.py
uv run python scripts/check_mccabe_ratchet.py
uv run mypy --strict --follow-imports=silent src/quant_fund/__init__.py src/quant_fund/public.py
uvx --from bandit==1.9.4 bandit -q -r src --severity-level medium --confidence-level medium
uv run pytest -n 2 --dist loadfile -m "not network" --cov --cov-report=term-missing
uv run coverage report
```

Report that block as jobs `ci/lint` and `ci/test`. `coverage report`
enforces the threshold in `pyproject.toml`. Do not shard, and do not also
launch the eight `ci.yml` matrix cells or the 36 `matrix.yml` cells.

Report one extra row `matrix/non-linux` with result `not-run`. A Linux
agent cannot execute the macOS, Windows, or Python 3.13/3.14 matrix legs.
`not-run` is not a pass and does not by itself fail the aggregate.

### Other workflow `run` steps

Read the workflow files at this commit. For every workflow and job listed
below, run each `run:` step in order, with that job's `env`, `defaults`,
and `working-directory`. Skip `uses:` steps (checkout, setup-uv, setup-node,
upload-artifact, download-artifact, attestations, Pages deploy). Install
the equivalent toolchain yourself, then run the shell.

Honor `paths` / `paths-ignore` for the diff against `main`, and honor
job-level `if` conditions other than `USE_CURSOR_CI`. Ignore `USE_CURSOR_CI`:
those jobs are skipped on the Actions runner so that you run them. Do not
run a job that would only run on `schedule` or `workflow_dispatch` unless
the run-context event is that event. On `pull_request` and `push`, do not
run `simtest-large` or `property-nightly.yml`.

Included workflows:

- `.github/workflows/ci.yml` except the `test` job (covered above) and the
  attest step. Still run `audit`, `package` (`uv build` and the wheel
  smoke), `container`, `perf`, `smoke`, `mc-engine-smoke`, `rust-accel`,
  `market-sim`, `parity-smoke`, `examples`, `audit-observability`,
  `diffbacktest`, `robustness-smoke`, `formal`, `evidence-audit`,
  `pretrade-risk`, and `stress-smoke`.
- `.github/workflows/fx1.yml` (`make fx1-lint`, `uv run mypy src/fx1` alone,
  `uv run mypy --strict src/fx1` alone, `make fx1-test`, the honesty pytest,
  and the corpus smoke).
- `.github/workflows/dependency-audit.yml`
- `.github/workflows/full-coverage.yml` when its path filters match
- `.github/workflows/perf-baseline.yml` (`compare` on pull_request, `record`
  otherwise)
- `.github/workflows/proofcore.yml`
- `.github/workflows/replay_viz.yml`, `reproduce_sota.yml`,
  `web_explorer.yml`, and `witness_monitor.yml` when their path filters match
- `.github/workflows/secret-scan.yml` using the pinned gitleaks checksum in
  that file
- `.github/workflows/simtest.yml` job `simtest-pr` on pull_request
- `.github/workflows/adversarial-properties.yml`
- `.github/workflows/arch_guards.yml`
- `.github/workflows/atlas.yml`
- `.github/workflows/docs.yml` job `build` only (`uv lock --check`,
  `uv sync --frozen --only-group docs`, `mkdocs build --strict`)

Do not run, and do not report as pass:

- `.github/workflows/codeql.yml`
- `.github/workflows/dependency-review.yml`
- `.github/workflows/scorecard.yml`
- `.github/workflows/release.yml`
- `.github/workflows/docs.yml` job `deploy`
- `actions/attest-build-provenance` and other `uses:` publish steps

Report those as `platform-only`. They stay on the GitHub Actions runner
because they publish code scanning, dependency review, scorecard,
attestations, or Pages. `platform-only` is not a pass and does not by
itself fail the aggregate.

The `ci.yml` `package` job still runs on push in Actions so SLSA
attestation keeps working. Run the build and smoke here anyway. Report
the attest step as `platform-only`.

If `evidence-audit` or epoch consistency fails because workflow files
drift from `quality/epoch_heads.json`, report that failure. Do not stamp
or sign your way around it.

## Report

Aggregate `pass` only when every required job you were supposed to run
exited 0. `not-run` and `platform-only` rows do not satisfy a required
job and do not flip a green aggregate to red by themselves.

When the run context has a pull request number, update the single issue
comment on the **base** repo whose body contains `<!-- cursor-ci -->`.
Create it if it is missing. Do not create a second comment. The body
contains the marker exactly once and the result line exactly once, as
the last line:

```markdown
<!-- cursor-ci -->
## cursor-ci

head: `<sha>`

| job | result | first failing lines |
| --- | --- | --- |
| ci/lint | pass | |
| ci/test | fail | first lines of the failure |

CURSOR_CI_RESULT: fail
```

`result` is `pass`, `fail`, `not-run`, or `platform-only`. Keep the first
failing lines short (at most 20 lines per failed job).

Set the commit status on the **head** repo for the head sha. The
description is exactly one of these two strings and nothing else:

```bash
# pass
gh api --method POST "repos/${HEAD_REPO}/statuses/${HEAD_SHA}" \
  -f state=success \
  -f context='cursor-ci' \
  -f description='CURSOR_CI_RESULT: pass'

# fail
gh api --method POST "repos/${HEAD_REPO}/statuses/${HEAD_SHA}" \
  -f state=failure \
  -f context='cursor-ci' \
  -f description='CURSOR_CI_RESULT: fail'
```

`HEAD_REPO`, `HEAD_SHA`, the base repo, and the pull request number come
from the run context. If there is no pull request number, set the commit
status and do not comment.

If you cannot set the status, or you cannot update the single comment
when a pull request number is present, the result is fail. Never report
pass because a command was skipped.
