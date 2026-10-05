# Dependency advisory scan — 2026-10-05

Completed at **17:06:40 UTC on 2026-10-05** against repository commit
`f7f94b700dab770e968c761736a98f6256801754`.

The queried advisory services returned **no known vulnerability matches** for
the locked package versions in the scopes below. These results describe an
advisory lookup at the recorded time; they do not establish that the repository
or its dependencies are vulnerability-free.

| Scope | Versions or dependency positions queried | Tool / service | Result |
|---|---:|---|---|
| Root Python lock, all extras, groups and platforms | 228 exact registry pins | pip-audit 2.10.1 / PyPI | Exit 0; no matches; no skipped packages |
| `web/package-lock.json` | 91 dependency positions | npm 11.9.0 audit | Exit 0; no matches |
| `replay/package-lock.json` | 26 dependency positions | npm 11.9.0 audit | Exit 0; no matches |
| `clients/typescript/package-lock.json` | 53 dependency positions | npm 11.9.0 audit | Exit 0; no matches |
| `clients/typescript/fx1/package-lock.json` | 33 dependency positions | npm 11.9.0 audit | Exit 0; no matches |
| `rust/quant_core/Cargo.lock` | 33 external registry crates | OSV querybatch API | HTTP 200; 33 results; no matches |

npm counts refer to positions within each lock tree and are not a count of
distinct packages across the repository.

## Commands and scope

The Python export used the frozen lock without installing the project or its
dependencies:

```sh
uv export --frozen --all-groups --all-extras --no-emit-project \
  --format requirements.txt --no-hashes --no-header --no-annotate \
  -o locked_with_markers.txt
sed 's/ ;.*//' locked_with_markers.txt | sort -u > locked_all_platforms.txt
uvx --no-env-file --from pip-audit==2.10.1 pip-audit \
  --strict --disable-pip --no-deps --progress-spinner off --timeout 10 \
  -r locked_all_platforms.txt --format json --output pip_audit.json
```

The run removed environment markers and deduplicated pins in Python; the `sed`
command above reproduces that normalization for this export. The resulting
`name==version` set was checked against all 228 registry packages in `uv.lock`.
The editable root project was excluded. Removing markers ensured that packages
locked for other supported platforms were queried too. `pip-audit` warned that
hashes were omitted; package artifact integrity was outside this scan.

For each of the four npm directories listed above, the run executed:

```sh
npm audit --package-lock-only --ignore-scripts --json \
  --fetch-timeout=15000 --fetch-retries=0
```

These commands ran on scratch copies of the exact committed package manifests
and lockfiles, without package installation or lifecycle-script execution.

For Rust, the run parsed the committed `Cargo.lock`, selected its 33
`registry+` packages, and posted their exact names and versions to
`https://api.osv.dev/v1/querybatch` as
`{"queries":[{"package":{"name":"<crate>","ecosystem":"crates.io"},"version":"<locked version>"}]}`.
The complete request and response are included in the evidence file.

## Maintenance change

`make audit-js` previously audited three npm trees and omitted the independently
locked client in `clients/typescript/fx1`. The target now includes that fourth
tree. No dependency constraints or lockfiles were changed.

## Coverage limits

- `cargo` and `cargo-audit` were unavailable. The Rust check queried OSV;
  it did not run `cargo audit`, check yanked crates, or evaluate Cargo-specific
  lock consistency and unmaintained-package warnings.
- Vendored `third_party/kronos` requirements are separate from the root lock.
  Its web UI requirements include `../requirements.txt`. This run did not
  resolve or audit that independent, partly unpinned environment. The existing
  `make audit-kronos` target remains the separate command for that scope.
- Advisory lookup does not assess source-code defects, exploit reachability,
  malicious packages, artifact integrity, or unpublished vulnerabilities.
- Results apply to the recorded snapshot and pins; subsequent dependency
  changes need another scan.

## Evidence

[Raw results and provenance](dependency-audit-2026-10-05.json) include exact
Python pins, all four npm responses, the Rust request and response, tool
versions, the root lock SHA-256, and Git blob IDs for the npm and Rust locks.
The scan made no repository or GitHub changes.
