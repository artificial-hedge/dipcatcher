# Full offline coverage and validation

This is an execution/measurement lane, not a declaration that the repository
has 100% coverage, no vulnerabilities, or production readiness. The ordinary
PR gates and their 81% coverage floor are unchanged.

## Scope

The `Full offline coverage` workflow collects branch and line coverage from
four balanced lab shards plus separate FX1, examples, performance, and native
Python suites. Slow and `perf_full` tests are included. Suites remain separate
to preserve their existing import/discovery behavior. The native extension is
built before its Python wrapper suite.

Every tracked Python file under `src/`, `scripts/`, and `examples/` is in the
measurement inventory, including entrypoints and files never imported by a
test. Untouched files get zero-hit entries, not silently omitted denominators.
This is not quantitative JavaScript or Rust source coverage, coverage of
vendored code, or coverage of Python files outside those three roots.

Independent jobs run the repository's lint/type/FX1/security/documentation
gates, locked Python and four npm dependency audits, browser/JavaScript
behavioral checks, Rust tests/audit, and the separate Kronos dependency audit.
A failed job does not cancel its siblings. Missing client `test` scripts fail
rather than being excused with `--if-present` or `--passWithNoTests`.

No deployment, hosted evaluation, broker integration, or real signing keys
are configured. Tests marked `network` remain excluded and are an explicit
gap; this marker filter is not a network sandbox. CodeQL and secret scanning
remain separate existing workflows, and their status must be inspected too.

## Run from a complete checkout

```sh
make sync
make native
uv run --frozen python scripts/full_coverage.py all \
  --output data/metadata/full-coverage/local-run-001
```

Output directories must be new. The `all` command runs lanes sequentially to
avoid multiplying xdist's worker count. Actions runs them on separate runners.
The Python command does not run the independent language/security jobs.

For a single isolated worker, use `run --lane python-1 --output <new-directory>`.
Copy the complete worker result into a directory named after its lane, then
combine all eight directories with:

```sh
uv run --frozen python scripts/full_coverage.py combine \
  --parts <parts-directory> --output <new-report-directory>
```

All workers and the reporting checkout must refer to the same commit and
coverage configuration. The caller must provide a complete, unmodified
checkout and its frozen environment; the artifact hashes are consistency
checks, not independent attestations against a malicious producer.

## Result contract

Each lane retains `pytest.log`, JUnit XML, a manifest, explicit configuration,
and controller coverage data. Aggregation requires every named lane, matching
commit/configuration identities and hashes, nonempty branch data, and actual
JUnit results. Worker fragments cannot substitute for a missing controller.
Failed/all-skipped/empty lanes cannot produce a passing aggregate. Individual
skipped tests are listed and `has_skipped_tests` remains visible.

The combined output contains raw coverage data, `coverage.json`,
`coverage.xml`, browsable HTML, and `summary.json`. The inherited project floor
is applied using an explicit configuration from the checkout root. Reports
are retained even when the floor or tests fail. Missing inputs produce a failed
summary instead of a smaller, apparently complete result.

Passing this floor is not 100% line/branch coverage. Review the actual totals,
per-file missing lines/branches, skipped tests, and failed/blocked jobs before
making any stronger statement. Installing dependencies, typechecking clients,
and running empty Rust suites do not establish behavioral coverage.

## Workflow integrity and initial integration

Workflow files belong to the corpus-epoch chain. This change requires a new
workflow epoch and the corresponding `quality/epoch_heads.json` update before
integration; historical receipts must not be rewritten. The read-only
`epoch-candidate` job runs the repository's original writer in a full checkout
and uploads `workflow-epoch-candidate` for review. It does not push commits or
make existing evidence checks pass. Apply and verify that candidate against
an unchanged workflow snapshot, or regenerate after concurrent edits.

The initial authoring environment could not resolve GitHub for a full clone
and did not have the frozen dependencies, Ruff, or mypy. Local regression
results validate this runner and synthetic miniature projects only, not the
full application. The first full workflow result is still required.

Known gaps at authoring: the research TypeScript client's TypeScript 7 lock
conflicts with openapi-typescript's TypeScript 5 peer requirement; both typed
clients declare typechecking but no behavioral `test` script. These are real
follow-up defects, not permission to weaken dependency resolution or tests.
