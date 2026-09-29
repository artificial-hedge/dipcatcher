# 13 — Data engine & backtest performance (SOTA lane)

Status: research notes + adoption plan, 2026-09-28. Honesty contract applies:
every timing in this document is **infrastructure throughput on SYNTHETIC or
matched lab workloads**, never market evidence and never a live-P&L claim
(`live_pnl_claim=false`, `research_only=true`). Speedups are correctness-
neutral engineering; they do not create edge.

Related in-tree docs: `docs/PERF.md` (wave-by-wave rebench ledger),
`docs/FAST_REPLAY_P42.md` (vectorized replay contract),
`docs/DATA_LAKE.md` (content-addressed lake + DuckDB query layer),
`tests/perf/` (benchmark suite + regression gate),
`scripts/perf_baseline.py` (standalone baseline harness).

---

## 1. Technique summaries (with citations)

### 1.1 Polars lazy/streaming vs pandas for financial time series

- **Lazy API + query planning.** `scan_parquet` builds a logical plan; Polars
  applies predicate/projection pushdown and parallel execution before
  materializing. Eager `read_parquet` forfeits pushdown — the whole file must
  fit in RAM before the plan runs.
  <https://realpython.com/polars-lazyframe/>
- **Streaming engine.** `collect(engine="streaming")` executes in batches and
  is reported to be *more* performant than the in-memory engine in addition to
  enabling >RAM datasets; `sink_parquet` writes batches without ever
  materializing the full result. Fallbacks are silent — inspect
  `explain(streaming=True)` for `STREAMING:` markers.
  <https://docs.pola.rs/user-guide/concepts/streaming/>
- **Measured regime split (2026, 14 GB workload).** pandas 2.2 (numpy) 24m /
  28.4 GB peak RSS with OOMs; pandas-pyarrow backend 18m / 19.6 GB; Polars
  *eager* 28m / 26.1 GB (worse than pandas-pyarrow); Polars lazy + streaming
  3m38s / 8.9 GB. Rule of thumb: under a few GB that fits in RAM, pandas-pyarrow
  is within ~2× and has the larger ecosystem; for >RAM batch jobs, lazy+streaming
  Polars "is not close — it's a different category".
  <https://pythondatabench.com/article/polars-vs-pandas-2026-benchmark>
- **Caveats.** Streaming silently falls back to in-memory for non-streamable
  ops (some joins, certain window functions); sorted "merge join" is still
  hash-join-based in the streaming engine (pola-rs/polars#24714). A cast inside
  a `scan_parquet` schema override killed pushdown (~20% regression) in one
  report. Verify plans, don't assume them.

### 1.2 DuckDB for analytical queries over Parquet

- **Automatic pushdown + streaming reads.** `read_parquet` queries run in
  parallel with filters pushed into the scan and only required columns read;
  row-group min/max zonemaps skip non-matching chunks, sometimes whole files.
  In the canonical NYC-taxi demo DuckDB beat manual pyarrow projection+filter
  code with zero tuning (70 ms vs seconds).
  <https://duckdb.org/docs/lts/guides/file_formats/query_parquet>,
  <https://www.duckdb.org/2021/06/25/querying-parquet>
- **Row-group sizing.** Keep row groups ~100K–1M rows (Parquet docs suggest
  128 MB–512 MB uncompressed) to balance parallelism vs metadata overhead.
  <https://duckdb.org/docs/lts/data/parquet/overview.html>,
  <https://www.velodb.io/glossary/par-1>
- **ASOF JOIN** (v1.2+) is the point-in-time-correct temporal join: nearest
  preceding match per row, with `asof_loop_join_threshold` (default 64)
  controlling nested-loop plans for small probe sets. Sorting cannot yet be
  skipped for pre-sorted inputs.
  <https://duckdb.org/2025/02/19/asof-plans>
- **Async I/O (v2.0, fall 2026).** Parallel range fetches with read-ahead
  saturate bandwidth on remote storage: ~3× faster on a 22 GB S3 Parquet scan
  (8.23s → 2.84s). Relevant if the lake ever moves off local NVMe.
  <https://duckdb.org/2026/07/31/asynchronous-io>

### 1.3 Arrow zero-copy patterns

- **PyCapsule interface.** Polars ≥1.3 implements the Arrow PyCapsule
  Interface: zero-copy exchange of `Series`/`DataFrame` with any
  PyCapsule-supporting library, no pyarrow dependency required. Also available:
  raw Arrow C Data Interface (`_export_arrow_to_c`).
  <https://docs.pola.rs/user-guide/misc/arrow/>
- **Arrow → pandas.** Zero-copy `to_pandas`/`to_numpy` only for primitive
  types without nulls; otherwise memory doubles. Mitigations:
  `split_blocks=True, self_destruct=True` (self-destruct renders the source
  table unusable), `zero_copy_only=True` to fail loudly instead of copying.
  <https://arrow.apache.org/docs/python/pandas.html>
- **Parquet reader.** `pl.read_parquet(..., memory_map=True)` is the zero-copy
  local path (already explicit in `data/adapters/parquet.py`).

### 1.4 Vectorized backtesting: vectorbt & numba JIT

- **vectorbt model.** Represents strategy instances as NumPy matrices;
  thousands of configurations are packed into one multi-dimensional array and
  evaluated at once — grid search becomes seconds instead of hours. Path
  dependency is handled by compiled backends: Numba JIT plus (v1.0.0, 2026)
  an **optional precompiled Rust engine** with auto-dispatch and per-call
  override (`engine="rust"`), eliminating JIT warmup for repeated runs.
  <https://vectorbt.dev/>, <https://github.com/polakowo/vectorbt/releases/tag/v1.0.0>
- **numba JIT discipline.** `@njit(cache=True)` persists compiled code to
  `__pycache__` (or `NUMBA_CACHE_DIR`) so restarts skip compilation. Known
  traps: cache invalidation does **not** see changes in functions imported from
  other modules; globals are frozen as compile-time constants; the cache is
  CPU-model-specific. JIT amortizes over large arrays — for small or single
  runs it is overhead (this is why vectorbt added the Rust engine).
  <https://numba.readthedocs.io/en/stable/user/jit.html>,
  <https://numba.readthedocs.io/en/stable/developer/caching.html>

### 1.5 Caching: content-addressed stores & joblib

- **joblib.Memory** hashes arguments (pickle → MD5) and persists results to
  disk; `memoize` (in-RAM) is only for small objects. Known failure modes:
  hashing large arrays can cost more than the recomputation it saves;
  `joblib.hash()` is **not stable across joblib/numpy versions**, so upgrades
  silently invalidate a whole cache. Put the expensive step in a pure function;
  `/dev/shm` as location on Linux for RAM-backed caching.
  <https://joblib.readthedocs.io/en/stable/memory.html>,
  <https://github.com/scikit-learn/scikit-learn/issues/12323>
- **Content-addressable storage (CAS).** The decache backend stores large
  arrays by hash-of-content in a shared `blobs/` dir, deduplicating identical
  arrays across cache entries; drop-in for joblib.
  <https://github.com/POFK/decache>
- Design rule: keys must include (input content digest, code version, config,
  library version) or the cache will serve stale science — the honesty contract
  here already treats receipts as immutable evidence, and caches should inherit
  the same digest discipline.

### 1.6 Chunked / incremental pipelines

- **Partition + sort for pruning.** Partition by low-cardinality filter columns
  (`source=/symbol=/date=` — already the lake layout); avoid over-partitioning
  (small-file problem). Sort data by frequent filter columns so row-group
  min/max statistics actually prune. Page-level stats/bloom filters give
  finer skip resolution.
  <https://dev.to/alexmercedcoder/all-about-parquet-part-10-performance-tuning-and-best-practices-with-parquet-1ib1>,
  <https://stackoverflow.com/questions/76782018/>
- **Streaming writes.** Buffer to ≥1 row group before flush; 2-pass writes
  (small scratch row groups compacted into large ones) keep peak memory bounded.
  <https://estuary.dev/blog/memory-efficient-streaming-parquet/>
- **Incrementality via checkpoints**, not re-scanning: append-only lineage
  records keyed by input snapshot IDs + code hash (the lake's lineage model)
  are the right shape — rebuild only partitions whose inputs changed.

### 1.7 Benchmarking methodology with statistical rigor

- **hyperfine** (CLI): auto run count (≥10 runs / ≥3 s), warmups,
  cache-clearing prepare commands, outlier detection, JSON/MD export. Good for
  process-level wall time of scripts.
  <https://github.com/sharkdp/hyperfine>
- **pytest-benchmark**: calibration + pedantic mode, `--benchmark-precision`
  (rounds until relative margin of error < fraction), `--benchmark-compare-fail
  min:5%` regression gating, JSON export.
  <https://pytest-benchmark.readthedocs.io/en/latest/usage.html>
- **ASV (airspeed velocity)**: benchmarks a project across its *commit
  history*; `asv publish` runs statistical change-point detection and names the
  commit where a regression landed; `asv find` bisects. SciPy uses it as the
  reference continuous-benchmarking setup.
  <https://asv.readthedocs.io/en/stable/using.html>,
  <https://docs.scipy.org/doc/scipy-1.16.0/dev/contributor/benchmarking.html>
- **pyperf**: multi-worker process isolation, auto-calibration, stability
  detection, `compare_to` significance testing between suites.
  <https://github.com/psf/pyperf>
- **CI regression gates.** Wall time on shared runners is noisy; two credible
  fixes: (a) *calibrated ratios* — divide by a same-machine reference kernel
  (what `tests/perf/check_regression.py` does) or instruction counting
  (CodSpeed/callgrind, deterministic across heterogeneous VMs, ~20× slower);
  (b) *statistical thresholds* — fail when delta exceeds 2–3× observed CV, or
  Welch t-test / z-score against a rolling baseline (Bencher). Cumulative
  drift is the real enemy: 2%/PR invisible per-PR is +22% after 10 PRs.
  <https://codspeed.io/docs/features/customization>,
  <https://buildwithaitoday.com/track/performance-engineering/learn/pee-benchmark_regression_ci>

---

## 2. Current-state profile

### 2.1 What the stack already does (don't re-buy it)

| Capability | Where | State |
|---|---|---|
| Polars-first dataframes | `pyproject.toml` (`polars>=1.17`), all of `data/`, `features/`, `pipeline/` | Done; eager reads dominate |
| DuckDB over lake Parquet | `data/lakehouse/query.py` (`read_parquet`, PIT `asof_bars`) | Done for lakehouse; not used for gold/panel analytics |
| Arrow zero-copy reads | `data/adapters/parquet.py` (`memory_map=True`), `data/lake.py` | Done |
| numba JIT kernels | `features/_kernels.py`, `backtest/_fast_kernel.py` (`@njit(cache=True)`, bit-identical contract, pure-Python fallback) | Done |
| Vectorized replay engine | `backtest/fast_replay.py` + `docs/FAST_REPLAY_P42.md` (fail-closed refusals, bit-identical vs reference loop) | Done |
| Content-addressed lake + lineage | `data/lakehouse/` (`objects/sha256/<aa>/<digest>`, snapshot manifests, lineage records) | Done |
| Process-local content-digest caches | `pipeline/dataset.py` `_PANEL_CACHE` (sampled head/tail invalidator), `compute/cache.py` `FitCache` (SHA-256 LRU), forecast caches (conformal/ranker/wrappee/GARCH) | Done |
| Benchmark gate (calibration-normalized) | `tests/perf/` + `check_regression.py` (ratio vs `test_calibration_matmul`, 50% threshold, fail-closed on missing benches), CI `perf` job | Done |
| Standalone baseline harness | `scripts/perf_baseline.py` (median-of-5, 1.25× pybroker-style threshold, git_revision provenance) | Done; CI wiring deferred (INFLIGHT, 2026-09-27) |
| Incumbent comparison | `scripts/incumbent_bench_vectorbt.py` / `_qlib.py` / `_zipline.py`, `scripts/run_incumbent_benchmarks.py` | Done |

### 2.2 Where time goes (evidence-backed)

**Causal research loop (`pipeline.forecast`).** From `docs/PERF.md` waves
5–39 (SYNTHETIC lab panel, 25 dates, 885 rows):

- gold `panel()`: cold ≈ 0.022 s → hot ≈ 4.5e-5 s (~500×; parquet+PIT+join avoided).
- `optimize_asof`: cold ≈ 0.20–0.29 s → hot ≈ 0.010 s (~19–29×).
- conformal full result-cache hit ≈ 0.0018 s (near-free); wrappee-only
  same-asof ≈ 1.56–1.6× — residual Student-t MLE + Mondrian/CQR
  `design_matrix`/row-hash work dominates when the full cache misses.
- causal 25-date panel ≈ **3.13–3.15 s**, stuck ~0.3 s above the Wave-5 best
  (2.84 s). The date loop is **sequential by requirement** (`w_prev` turnover
  causality) — parallelism across dates is blocked, so per-date constant cost
  is the only lever.
- Wave 12 lesson (profiled): a day-index `pl.concat` "optimization" was ~16×
  *slower* than `filter(event_time <= cutoff)` and cost ~1.1 s of causal wall;
  the fix was O(log n) `history_prefix_upto` (Series binary search + `slice`,
  ~0.023 ms vs filter ~0.49 ms vs concat ~7.8 ms). **Micro-optimizations here
  must be profiled and rebenched, not asserted.**

**Backtest replay.** Incumbent evidence
(`.dsh-24x7/evidence-incumbent-vectorbt-fast-11a.json`, read-only): on real
Binance daily bars, 11 assets × 999 bars, matched economics and fills
(decision close t → exec open t+1), 10 reps:

- dipcatcher **fast** engine median **84.8 ms** (76.6–104.1) vs vectorbt 1.1.0
  median **76.6 ms** (70.3–80.7) — ~11% slower, single-process wall time.
- NAV parity to 1.5e-15 relative; identical fills (5,734) and fees; dipcatcher
  fails closed on duplicate weights / stale marks / kill-switch where vectorbt
  accepts. The disclaimer in the receipt is explicit: latency parity on one
  matched workload, not a superiority claim.

Interpretation: the vectorized replay is already in the incumbent's latency
band. The remaining gap is small and the differentiator is fail-closed
semantics, not speed — further engine micro-tuning has low leverage. A cProfile
sample of the fast path (`_prof2.txt`, root) shows the *Python driver* around
the numba kernel still spending: `run_backtest_fast` frame prep 0.206 s
internal, 16 × `LazyFrame.collect` 0.129 s, plus `iter_rows`/`row_tuples`/
`sum`-genexpr per-row costs — i.e., matrix construction and per-date Polars
materialization, not the arithmetic kernel.

**Data engine I/O.** Gold/silver reads are eager `read_parquet` of whole files
(`data/lake.py`); fine at current panel sizes (cold panel ≈ 22 ms), but there
is no lazy/streaming path for the HF minute-bar corpus
(`data/adapters/hf_ohlcv_1m.py` does use `scan_parquet` + pushdown in
`_scan_vendor`, then `collect()` eagerly). Minute-scale corpora are where
>RAM streaming and DuckDB pushdown will matter first.

**Parallel sweeps.** `compute/parallel.py` `process_map` uses the `fork`
context and falls back to serial on failure — on Windows (this workstation) it
is effectively always serial.

### 2.3 Benchmark instrumentation gaps

1. Two harnesses coexist: `tests/perf` (pytest-benchmark, calibration-normalized,
   CI-gated at 50%) and `scripts/perf_baseline.py` (median-of-5, 1.25× threshold,
   not yet CI-wired — deferred because baselines must be recorded on CI runners).
   Threshold philosophies differ (50% calibrated vs 25% absolute); neither does
   significance testing.
2. `perf_bench.json` wave artifacts (`data/metadata/perf_bench.json`) are cited
   throughout `docs/PERF.md` but no tracked script currently regenerates them —
   wave history is not reproducible from the tree alone.
3. No cross-commit trend tracking (ASV-style) — regressions are caught only
   when a single run exceeds a static baseline ratio.

---

## 3. Adoption plan

Prioritized by leverage per unit of risk. Every phase keeps the honesty
contract: no headline metric changes, receipts stay reproducible, SYNTHETIC
labels preserved.

### Phase 1 — Benchmark harness consolidation + statistical rigor (highest leverage)

1. **Unify on `tests/perf` as the single gate.** Keep calibration-normalized
   ratios (correct for heterogeneous runners); fold `scripts/perf_baseline.py`'s
   six research kernels (CRPS, stationary bootstrap, reality check, conformal
   quantile, e-BH, energy score) in as `tests/perf/test_research_kernels.py`
   cases so one gate, one baseline, one CI job covers both suites. Retire the
   second harness or demote it to a local-only diagnostic.
2. **Add significance testing to `check_regression.py`.** pytest-benchmark JSON
   already carries rounds + stats; compute CV per bench, and fail only when the
   calibrated ratio exceeds baseline by `max(threshold, 3×CV)` — or implement a
   Welch t-test between the run's round samples and stored baseline rounds
   (Bencher-style). This kills the "jitter looks like regression" noise that
   `docs/PERF.md` repeatedly disclaims, without loosening real detection.
3. **Make the wave artifact reproducible.** Add a tracked
   `scripts/perf_wave.py` (or a pytest hook) that writes
   `data/metadata/perf_bench.json` with the existing schema, so PERF.md wave
   numbers are regenerable and citable from a receipt hash.
4. **Add data-engine benches to the gate** (fast tier): `panel()` cold/hot,
   `build_features[24sym-252d]` (exists), `hf resample` (exists), plus
   `lakehouse asof_bars` and gold join at minute-scale synthetic sizes.
5. Optional later: ASV over commit history for trend/bisection (`asv find`),
   and/or CodSpeed-style instruction counting if runner noise ever defeats
   calibration. Do not adopt both — calibration ratio + significance test is
   the cheap 80%.

### Phase 2 — Cache design: promote process-local caches to a disk CAS

1. **Content-addressed disk tier for the expensive fits.** The Student-t /
   Gaussian wrappee MLE and GARCH as-of fits are the dominant cold-path cost in
   the causal loop (PERF.md: full conformal hit 0.0018 s vs wrappee-only
   ~0.085 s). Back `FitCache` with a joblib-style disk store under
   `data/cache/fits/`, keyed by the *existing* digest discipline:
   `sha256(train target/scale arrays) + family + alpha + label + config +
   code_hash + joblib/numpy versions`. Reuse `utils.reproducibility.content_address`
   so cache keys and receipt keys share one hashing story.
2. **Version-tag every key** (joblib lesson: hashes are not stable across
   library upgrades). A library bump must invalidate, never silently serve.
3. **CAS dedup for payloads**: store arrays by content hash in a shared
   `blobs/` dir (decache pattern) so identical train windows across as-of dates
   are written once.
4. **Bound + LRU everywhere** (already true for `FitCache`/`_WRAPPEE_CACHE`);
   document eviction in `docs/PERF.md`. Never cache across the honesty boundary:
   research-only artifacts must not be readable as production state.

### Phase 3 — Polars/DuckDB opportunities (do these only where profiled)

1. **Lazy column projection on gold reads.** `dataset.panel()` cold path reads
   whole feature/label parquets eagerly; when `feature_names` is given, use
   `scan_parquet(...).select(keys + feats + label)` so projection pushdown
   reads only needed columns before the PIT validation. Cheap, semantics-preserving
   (validation still runs on the projected frame), measurable via the Phase-1 bench.
2. **Streaming engine for >RAM builds.** Convert `build_gold` /
   `hf_ohlcv_1m` month-file scans to `collect(engine="streaming")` /
   `sink_parquet` **only** for corpus sizes that exceed RAM (minute-scale HF
   data). Verify with `explain(streaming=True)` that no silent fallback occurs
   (joins in the pipeline are the risk). At current daily-panel sizes this is a
   no-op — do not pay batching overhead for frames that fit in RAM.
3. **DuckDB ASOF JOIN for PIT joins.** The lakehouse query layer already speaks
   DuckDB; express "latest knowable row per (security_id, decision_time)"
   memberships/marks as `ASOF JOIN` instead of correlated filters where
   correctness tests can prove equivalence against the Polars path. Keep the
   Polars path as reference (fail-closed philosophy).
4. **Arrow PyCapsule / numpy zero-copy in replay frame prep.** `_prof2.txt`
   shows `_bars_to_matrices` + `LazyFrame.collect` + `iter_rows` dominating the
   fast-replay driver. Replace per-row Python extraction with
   `Series.to_numpy(zero_copy_only=False)`-style columnar pulls (or
   `__arrow_c_array__`) into the pre-indexed matrices the numba kernel already
   expects. This targets the measured 0.2 s driver cost without touching the
   bit-identical kernel contract; the conformance suite is the gate.
5. **Windows parallelism.** `compute/parallel.py` should support `spawn`
   (picklable payloads) so process sweeps actually parallelize on this
   workstation; keep `fork` on Linux. Order-preserving map + `derive_seed`
   already make results worker-independent.

### Phase 4 — Explicitly deferred / rejected

- **vectorbt-style Rust kernels for the replay engine.** The incumbent bench
  shows ~11% latency gap at parity NAV with fail-closed advantages; a Rust port
  of `_fast_kernel` buys little and doubles the bit-identical conformance
  surface. Revisit only if the workload class grows (intraday, >100 names) and
  profiling shows the numba kernel (not the driver) dominating.
- **Parallel causal dates.** Blocked by `w_prev` turnover causality (PERF.md,
  standing deferral). Do not "fix" with speculation that could reorder TC.
- **pandas migration anywhere.** The repo is Polars-first by design; pandas
  stays only where vendor interop requires it.
- **GPU / differentiable engines** (`diffbacktest` JAX lane exists separately) —
  out of scope for the data-engine lane.

### Sequencing & success criteria

| Phase | Change | Gate / success criterion |
|---|---|---|
| 1 | Single benchmark gate + significance test + reproducible wave artifact | CI `perf` job green; false-positive rate on 10 no-op runs ≈ 0; `perf_bench.json` regenerable from tree |
| 2 | Disk CAS for fit caches | Causal 25d wall across a *fresh process* approaches the hot-cache band (~3.1 s) without in-RAM warmup; stale-reuse tests still fail closed |
| 3a/3b | Lazy projection; streaming for minute corpora | `panel(feature_names=…)` cold time drops measurably (benched, calibrated ratio); HF month scan peak RSS bounded on >RAM synthetic corpus |
| 3c/3d | DuckDB ASOF equivalence; zero-copy matrix prep | Equivalence tests vs Polars reference; fast-replay driver internal time down vs `_prof2.txt` baseline with bit-identical outputs |
| 3e | spawn-safe `process_map` | Order + seed determinism tests on Windows and Linux |

All phases land with tests first (fail-closed on contract violations), and any
timing claim in `docs/PERF.md` must come from the Phase-1 harness with its
SYNTHETIC/research-only labels intact.
