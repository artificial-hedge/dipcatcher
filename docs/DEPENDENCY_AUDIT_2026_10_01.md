# Dependency cleanup and vulnerability fixes — 2026-10-01

Scope: root Python runtime, development, documentation and optional extras;
all three npm projects; optional Rust extension; vendored Kronos requirements.
The artificial `tests/fx1/_test_uv.lock` is a test fixture, not an installable
dependency manifest. Existing unrelated working-tree changes were preserved.

## Fixes

| Package | Before | After | Advisory family |
|---|---|---|---|
| urllib3 | 2.7.0 | 2.8.0 | CVE-2026-97687, CVE-2026-97688, CVE-2026-97689 |
| virtualenv | 21.7.9 | 21.14.2 | PYSEC-2026-4011 through PYSEC-2026-4014 |
| Flask (Kronos UI) | 2.3.3 | 3.1.3 | PYSEC-2026-2151 |
| Flask-CORS (Kronos UI) | 4.0.0 | 6.0.5 | PYSEC-2024-71, PYSEC-2024-271, PYSEC-2026-1383/1384/1385 |
| PyO3 | 0.23.5 | 0.29.3 | RUSTSEC-2025-0020, RUSTSEC-2026-0177, GHSA-36hh-v3qg-5jq4 |

Primary references include the [urllib3 proxy TLS advisory](https://github.com/urllib3/urllib3/security/advisories/GHSA-8988-9cw3-xx77),
the [PyO3 buffer overflow advisory](https://rustsec.org/advisories/RUSTSEC-2025-0020.html),
and the [PyO3 thread safety advisory](https://rustsec.org/advisories/RUSTSEC-2026-0177.html).
Python audit counts contained duplicate advisory IDs; the table reports
advisory families rather than inflating the count with duplicated records.

The Rust NumPy binding moves from 0.23.0 to 0.29.0 with PyO3. The only source
compatibility change is `Bound.downcast` to `Bound.cast` in `hash_many`.
The Rust lock regeneration removes four obsolete transitive crates.
Python NumPy stays at 2.5.3. Root and Kronos Torch minimums are raised to 2.6;
the root lock's existing Torch version is unchanged.

Root transitive security floors are resolver constraints (`urllib3>=2.8.0`,
`virtualenv>=21.7.13`), avoiding artificial direct runtime dependencies.
Kronos UI requirements include the shared parent requirements rather than
duplicating pandas, NumPy, Torch and Hugging Face Hub declarations. The shared
NumPy minimum preserves the previous UI constraint.

## Ongoing coverage

- The TypeScript API client now has a lockfile and uses `npm ci` and its
  declared scripts instead of fetching independent versions with `npx`.
- Dependabot manages root Python exclusively through uv, covers both Kronos
  requirements files through pip, and also covers npm and Cargo.
- `make audit-all` combines Python, npm, Rust and Kronos audits. Individual
  commands and the cargo-audit installation command are in `SECURITY.md`.
- Root Python audits inspect pinned versions without pip resolution and strip
  platform markers to include Windows/Linux extras on every audit host.
- A dependency-change and weekly CI workflow covers all four ecosystems,
  including client typechecking. Actions are pinned to existing repo SHAs.

Kronos has independent requirements files rather than a full transitive lock;
its audit checks a fresh Python 3.12 resolution. No model weights were fetched,
and no remote fleet jobs or research receipts were changed by this cleanup.

## Validation

The broad checks below ran in the original working tree based on `f4c8ac801b`,
including its unrelated local changes. For publication, the dependency-only
patch was applied to current remote main (`2a0253d7ed`) in an isolated worktree.
Lock consistency and `make audit-all` were checked again there. The complete
lab suite has not been rerun against that newer base; CI must validate the PR.

Audit results: no known vulnerabilities in all 220 locked third-party Python
distributions, all three npm trees, the Rust lock (cargo-audit and OSV), or
the freshly resolved Kronos UI/model requirements. The audits query current
advisory databases and do not establish the absence of unknown vulnerabilities.

Passed: frozen Python sync, lock consistency, Ruff check and formatting,
fx1 tests and types, strict public-facade types, Rust formatting and Clippy, native extension release
build, web production build, replay typecheck/build, client typecheck, and
Kronos HTTP smoke checks for `/`, data files, available models and model status
(including CORS headers).

Full offline lab gate: **15,797 passed, 6 failed, 190 skipped, 1 xfailed** in
28 minutes. Failed tests:

- `tests/unit/cli/test_cli_api.py::test_api_key_comparison_is_constant_time`
  expects strings while the already modified API compares encoded bytes.
- `tests/unit/docs/test_arch_atlas.py::test_committed_artifacts_are_fresh`
  and `test_check_cli_exit_zero`: existing modified architecture artifacts
  are stale.
- `tests/unit/test_quality_ratchet.py::test_mypy_strict_allowlist_only_grows`:
  the existing modified allowlist refers to missing files.
- `tests/unit/cli/test_cli_startup.py::test_package_and_cli_imports_skip_heavy_modules`:
  NumPy and SciPy enter the CLI import path in the checkout with existing CLI
  changes.
- `tests/unit/pretrade/test_latency.py::test_allow_path_meets_latency_gate`:
  the 5-microsecond latency threshold fails during the concurrent suite;
  the pretrade latency tests pass on the final isolated recheck.

Additional validation issues:

- `make lint`'s allowlist check references five missing files in the already
  modified `quality/mypy_strict_modules.txt`.
- Web unit tests lack `adaptive_mix_20asset_1d_20260922.json` and have an
  existing fixture-index mismatch for an untracked cost-calibration receipt.
- Harness mypy reports one missing annotation for `pooled` at line 244 of
  the existing untracked `blueprint_cmds.py` (888 files checked).
- `uv pip check` rejects the existing z3-solver wheel's platform metadata.
- Native numeric parity passes, but the timed `hash_many` benchmark initially
  failed its speed floor under concurrent load. The committed PyO3 0.23.5
  baseline was built separately, but macOS rejects its binary with a
  misaligned LINKEDIT string pool, preventing a reliable baseline comparison.
  The upgraded extension builds and loads successfully; no timing gate was
  weakened. Reruns also failed; after the large suite finished, `hash_many`
  measured 0.95x versus the 1.5x speed floor. This performance gate remains
  unresolved, despite passing native correctness checks.
