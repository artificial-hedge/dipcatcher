# Wire the CI optional-blob fill in front of the shard matrix

`docs/CI_OPTIONAL_BLOBS.md` describes the stamp, the pickle, and why four
`pytest-split` shards each cold-fill it. This note is the wiring. It is
not applied: `.github/workflows/ci.yml` stays as it is, and the `test` job
keeps `runs-on: instabox-large`.

The fill is one `build` per Python version. The shard matrix only
`install`s the tarball. `install` exits 2 when the archive stamp is not
the stamp of this checkout and this interpreter, and it does not replace
an existing pickle in that case. Do not ignore that exit code. A refused
install means the shard would otherwise cold-fill, which is the miss this
job exists to prevent.

## What the stamp binds

`scripts/ci_optional_blobs.py stamp` hashes salt `ci-optional-blobs-v1`,
`sys.version`, every `src/quant_fund/**/*.py` file, `pyproject.toml`, and
`uv.lock`. Production runs leave `DIP_SCORECARD_BUDGET` unset and never
read the pickle. The build job and the shard job must use the same
interpreter, so 3.12 and 3.13 are separate artifacts. A docs-only commit
reuses the artifact. A `src/quant_fund` edit does not.

`build` on this tree is about 610 seconds. A later process that finds the
same stamp loads it in about 3 seconds, including the import. The loader
ignores a pickle that lives under any other stamp, including one restored
by the `restore-keys` prefix on `~/.cache/dipcatcher`.

## Job order

Add one job, then make `test` wait for it. Inside `test`, restore
`~/.cache/dipcatcher` first, sync, download the artifact, `install`, and
only then run pytest. `install` writes

```text
${DIP_SESSION_CACHE_DIR:-~/.cache/dipcatcher}/ci-optional/<stamp>/ci.pkl
```

which is the path `quant_fund.research.ci_blob_cache` reads. The existing
`actions/cache` step may restore an older stamp beside it. That is
harmless: the loader looks up the current stamp only.

The artifact name is the Python version, not the stamp. The stamp is
inside the tarball, and `install` is what checks it. The `test` matrix
can download `ci-optional-py${{ matrix.python }}` without a job output.

```yaml
  ci-optional-blobs:
    runs-on: instabox-large
    timeout-minutes: 25
    strategy:
      fail-fast: false
      matrix:
        python: ["3.12", "3.13"]
    env:
      DIP_SCORECARD_BUDGET: ci
      OMP_NUM_THREADS: "1"
      MKL_NUM_THREADS: "1"
      NUMEXPR_NUM_THREADS: "1"
      VECLIB_MAXIMUM_THREADS: "1"
      OPENBLAS_NUM_THREADS: "1"
      TOKENIZERS_PARALLELISM: "false"
    steps:
      - uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1
        with:
          persist-credentials: false
      - uses: astral-sh/setup-uv@c18668ad3cf93ea998bef934396af7bb5c839dc7 # v10.2.0
        with:
          python-version: ${{ matrix.python }}
          enable-cache: true
          cache-dependency-glob: uv.lock
      - name: Sync
        run: uv sync --frozen --all-groups --all-extras
      - name: Build CI optional blobs
        run: uv run python scripts/ci_optional_blobs.py build
      - name: Pack
        run: uv run python scripts/ci_optional_blobs.py pack --out ci-optional.tar.gz
      - uses: actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a # v7.0.1
        with:
          name: ci-optional-py${{ matrix.python }}
          path: ci-optional.tar.gz
          retention-days: 1

  test:
    needs: ci-optional-blobs
    runs-on: instabox-large
    # The rest of the job is the existing test job: env, strategy
    # (python × shard), checkout, setup-uv, and the dipcatcher cache.
    steps:
      # ... checkout, setup-uv, Cache synthetic lakes and Northset benches,
      # Sync, unchanged ...
      - name: Install CI optional blobs
        run: |
          set -euo pipefail
          uv run python scripts/ci_optional_blobs.py install \
            --archive ci-optional-artifact/ci-optional.tar.gz
      # ... the existing Pytest step, unchanged, including
      # --splits 4 --group ${{ matrix.shard }} ...
```

The install step sits after Sync and after this download, which itself
sits after the cache restore:

```yaml
      - name: Download CI optional blobs
        uses: actions/download-artifact@3e5f45b2cfb9172054b4087a40e8e0b5a5461e7c # v8.0.1
        with:
          name: ci-optional-py${{ matrix.python }}
          path: ci-optional-artifact
```

`DIP_SCORECARD_BUDGET` does not have to be set on the shard. Pytest sets
it. `install` only checks the stamp and writes the pickle. `build` is the
step that requires the budget: the script sets it, and a non-empty
`load_blobs("ci")` makes `build` a no-op.

Shards that share a disk can skip the artifact and call `build` once.
The loader's file lock makes the others wait. The GitHub matrix does not
share a disk across `instabox-large` runners, so each shard needs the
tarball. Four shards then pay the install, not four fills.
