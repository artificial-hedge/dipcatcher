# CI optional blobs, one fill per stamp

Pytest sets `DIP_SCORECARD_BUDGET=ci`. The live-optional scorecard benches do
not read the research panel, so their results depend only on the code and the
interpreter. `quant_fund.research.ci_blob_cache` stores them at

```text
${DIP_SESSION_CACHE_DIR:-~/.cache/dipcatcher}/ci-optional/<stamp>/ci.pkl
```

`<stamp>` is `source_stamp()`: a hash of every `src/quant_fund/**/*.py` file,
`pyproject.toml`, `uv.lock`, and `sys.version`, with a fixed salt. Production
runs leave the budget unset and never read or write this file. A monkeypatched
`bench_*` or `DIP_DISABLE_SESSION_CACHE=1` also skips it.

## Why each shard still cold-fills

The pytest job is a four-way `pytest-split` matrix on `instabox-large`. The
shards start together. `actions/cache` restores `~/.cache/dipcatcher` and
saves it again only when the job ends, so a shard cannot see a pickle another
shard is still building.

The cache key includes `hashFiles('src/quant_fund/**/*.py')`. A new commit
misses that exact key. `restore-keys` may restore an older directory. The
loader ignores it, because the pickle has to live under the new stamp. Each
shard then builds the 1409 families itself. On this tree that fill measured
about 610 seconds, and a hit of the same stamp, including import, was about
3 seconds. Four shards therefore pay the fill four times.

`runs-on` stays `instabox-large`. This change does not edit the workflow.

## Fill once, install on every shard

Run the script with the same Python the shard will use. The stamp includes
`sys.version`, so 3.12 and 3.13 are different artifacts.

```bash
export DIP_SCORECARD_BUDGET=ci
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
uv run python scripts/ci_optional_blobs.py stamp
uv run python scripts/ci_optional_blobs.py build
uv run python scripts/ci_optional_blobs.py pack --out "ci-optional-$(uv run python scripts/ci_optional_blobs.py stamp).tar.gz"
```

`build` is a no-op when that stamp's `ci.pkl` is already present. `pack`
refuses to run without it. Put the tarball where every shard can see it
(a job artifact whose name is the stamp, or a directory on the runner that
outlives the shard). Before pytest, with the same interpreter:

```bash
export DIP_SESSION_CACHE_DIR="${DIP_SESSION_CACHE_DIR:-$HOME/.cache/dipcatcher}"
uv run python scripts/ci_optional_blobs.py install --archive ci-optional-<stamp>.tar.gz
```

`install` extracts a single member, `ci-optional/<stamp>/ci.pkl`. If the
stamp in the archive is not the stamp of this checkout and interpreter, it
exits 2 and leaves any existing file in place. After that, the research CLI
echo and `run_research` under the CI budget load the pickle instead of
rebuilding it.

A workflow sketch, not applied here:

```yaml
# runs-on stays instabox-large. One build job per python version, then the
# existing shard matrix installs the artifact before pytest.
- name: Stamp
  id: stamp
  run: echo "stamp=$(uv run python scripts/ci_optional_blobs.py stamp)" >> "$GITHUB_OUTPUT"
- name: Build CI optional blobs
  run: uv run python scripts/ci_optional_blobs.py build
- name: Pack
  run: uv run python scripts/ci_optional_blobs.py pack --out ci-optional.tar.gz
- uses: actions/upload-artifact@v4
  with:
    name: ci-optional-${{ steps.stamp.outputs.stamp }}
    path: ci-optional.tar.gz
```

Each shard downloads that artifact and runs `install` before `uv run pytest`.
Shards that share a machine can instead share `DIP_SESSION_CACHE_DIR` and
call `build` once; the loader's file lock already makes the other processes
wait. Separate runner disks still need the tarball.
