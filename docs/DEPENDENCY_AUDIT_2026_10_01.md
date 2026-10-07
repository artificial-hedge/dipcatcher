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
The initial Rust lock regeneration removes four obsolete transitive crates.
The follow-up enables runtime-detected SHA-256 acceleration on AArch64,
with the existing software fallback and unchanged timing thresholds.
Hash parity now also covers padding and compression-block boundaries.
Python NumPy stays at 2.5.3. Root and Kronos Torch minimums are raised to 2.6;
the root lock's existing Torch version is unchanged.

Root transitive security floors are resolver constraints (`urllib3>=2.8.0`,
`virtualenv>=21.7.13`), avoiding artificial direct runtime dependencies.
Z3 is constrained to `<5` (locked at 4.16.0.0): the 5.0.0.0 and 5.1.0.0
macOS ARM wheels have unsupported internal `macosx_13_3_arm64` tags,
despite their `macosx_13_0_arm64` filenames. The validated 4.16 wheel
passes `uv pip check`; no installed metadata is patched.
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
- `make sync` and CI frozen-install steps also run `uv pip check`, so
  incompatible wheel metadata fails setup rather than going unnoticed.
- Root Python audits inspect pinned versions without pip resolution and strip
  platform markers to include Windows/Linux extras on every audit host.
- A dependency-change and weekly CI workflow covers all four ecosystems,
  including client typechecking. Actions are pinned to existing repo SHAs.

Kronos has independent requirements files rather than a full transitive lock;
its audit checks a fresh Python 3.12 resolution. No model weights were fetched,
and no remote fleet jobs or research receipts were changed by this cleanup.

## Validation follow-up

The first run found baseline validation failures in both the original
working tree (`f4c8ac801b` plus existing local feature work) and the PR's
isolated remote-main base (`2a0253d7ed`). The follow-up fixes the causes:

- Remove a byte-identical duplicate measurement validator without removing
  any receipt schema or weakening the shared contract.
- Refresh generated architecture artifacts and document/pin the exact
  existing forecast-audit and atomic-write imports in ADR-0002.
- Restore the primary checkout's five missing strict-allowlist modules from
  remote main; keep its existing blueprint command work lazy and typed;
  align its API comparison test with the existing encoded-byte comparison.
  These feature-work repairs stay in the original checkout.
- Check web coverage against committed receipts, matching the exporter;
  read legacy unsealed evidence from its separate archive; locate the
  fast-replay fixture through the generated index instead of an old date.
  Refresh fixture copies from committed evidence without changing receipts.
- Fail closed for a near-zero crash-gate wealth anchor; the existing
  branch-regression test now passes.
- Use a compatible Z3 wheel and enable ARM SHA-256 acceleration rather
  than altering installed metadata or lowering performance thresholds.

Final check results will be recorded after the complete recheck.
