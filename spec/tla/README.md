# Order-lifecycle model checks

Run `bash scripts/run_tlc.sh` with Java 21, Bash, curl and GNU coreutils.
The script checks safety and liveness separately without changing their bounds,
invariants or properties. No Python application code is required.

## Pinned TLC artifact

We use stable **TLA+ tools v1.7.4 / TLC 2.19** (revision `5a47802`).
The upstream `v1.8.0` tag is a rolling prerelease: its build workflow replaces
`tla2tools.jar` and force-moves the tag. `run_tlc.sh` therefore downloads the
**stable** `v1.7.4` release asset, never `v1.8.0`. The tag URL is not trusted
by itself: `verify_jar` requires the pinned SHA-256 and byte size before the
file is cached or executed. A replaced upload fails closed.

- Official release: <https://github.com/tlaplus/tlaplus/releases/tag/v1.7.4>
- Download used by `scripts/run_tlc.sh` (stable v1.7.4, not the rolling v1.8.0 tag):
  <https://github.com/tlaplus/tlaplus/releases/download/v1.7.4/tla2tools.jar>
- Historical immutable asset id `184694200` (metadata only; the asset API
  returns HTTP 403 to unauthenticated CI as of 2026-10-13):
  <https://api.github.com/repos/tlaplus/tlaplus/releases/assets/184694200>
- Size: `2274532` bytes
- SHA-256: `936a262061c914694dfd669a543be24573c45d5aa0ff20a8b96b23d01e050e88`
- Upstream release's published SHA-1:
  `bee4a54f3ee3d4afc347c3240ec2d9e93b075104`

On 2026-10-13 the v1.7.4 tag download was fetched again over HTTPS and matched
this SHA-256 and size (the unauthenticated asset API returned HTTP 403 in CI
run 38343446613, so the tag URL is what runners can actually retrieve).
SHA-1 is provenance evidence only; runtime verification requires the pinned
SHA-256 **and** size, for both downloads and `TLA_TOOLS_JAR` overrides.

GitHub's asset API can change metadata without changing bytes, but unauthenticated
runners cannot download that API. Integrity is the pinned SHA-256 and size. If
the tag is moved or the asset is deleted, download or `verify_jar` fails closed;
there is no fallback and no automatic checksum update.

Compatibility was checked with both existing configs on Java 21: safety generated
3,523 states (345 distinct); liveness generated 428 (76 distinct). Both completed
without errors. v1.7.4 includes upstream's multi-worker liveness soundness fix
(<https://github.com/tlaplus/tlaplus/issues/971>). The model uses standard integer,
set, record and temporal operators; it does not require prerelease extensions.

Downloads use a unique temporary file beside the cache and are verified before
rename. Failed/partial downloads are removed and never cached or executed.
`TLA_TOOLS_JAR` must name a file, not a directory; a directory target fails
without retaining the temporary download. GNU `mv -T` prevents directory targets
from absorbing the downloaded file.
An invalid existing cache is reported with actual/expected digests and sizes,
not silently replaced. Remove that file deliberately after investigating it.
The default versioned filename avoids reusing older `.tla/tla2tools.jar` caches.

For upgrades, establish official artifact provenance and a fixed asset ID,
verify bytes independently before running them, update SHA-256/size/version
alongside this record, and rerun both model configs and the offline downloader
regression tests (`python scripts/test_run_tlc.py`). Never repair a checksum
failure by trusting the failed download or weakening verification.
