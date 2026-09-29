# SOTA 06 — Point-in-Time Correctness & Data Leakage Prevention

Lane: **POINT-IN-TIME CORRECTNESS & DATA LEAKAGE PREVENTION**. Status:
research notes + adoption plan, 2026-09-28. This document summarizes
state-of-the-art practice (1978–2026) for building leakage-proof research
data engines; audits the dipcatcher / fx-1 data pipeline (`src/quant_fund/data`,
`src/quant_fund/pit`, `src/quant_fund/leakage`, `src/fx1/data`) against that
practice; and proposes a concrete adoption plan. It modifies no code — it is
the design record for the next data-contract generation.

Honesty contract applies: nothing in this document is market evidence. All
guard failures discussed are **correctness** failures, not performance claims;
no Sharpe/Sortino/Calmar/P&L/NAV is headlined anywhere, per the repo's
`FORBIDDEN_RESEARCH_METRIC_KEYS` / `fx1.honesty.FORBIDDEN_HEADLINE_TOKENS`
contract. Synthetic leaky-oracle results (from
`src/quant_fund/validation/leakage_redteam.py`) are labeled SYNTHETIC wherever
referenced and are correctness tests, never edge claims.

Related in-tree docs: `docs/DATA_CONTRACTS.md` (PIT schema contract),
`docs/FX1_DATA.md` / `docs/FX1_DATASOURCES.md` (fx-1 ingest lanes),
`docs/AUDIT_P63_DATA.md` (findings F4/F8/F9 referenced by the LH rules),
`docs/SOTA/10-reproducibility.md` (receipt/Merkle side of the same proof
stack), `INFLIGHT` §leakage-redteam.

---

## 1. Verdict up front

The repo's **guard stack is unusually strong for a lab harness**: a write-once
bitemporal vault with hash-chained manifests (`pit/`), a runtime watchdog that
fails closed on any read whose `max_known_at` exceeds the decision time
(`leakage/watchdog.py`), a 14-rule AST leakage linter with cited allowlists
(`leakage/rules.py`, LH001–LH014), PIT universe construction with delisting
and ticker-change handling (`data/universe.py`), a temporal security master
(`data/security_master.py`), a Gençay-style leaky-oracle red team
(`validation/leakage_redteam.py`), and an fx-1 ingest gate that rejects
un-dated market data (`src/fx1/data/sources/ingest.py`, `requires_as_of`).
This is at or above published SOTA for most buy-side research shops, where
PIT correctness is usually enforced by convention and vendor data alone.

Three structural gaps remain (details in §3.3):

- **G1 — Two data architectures.** The vault enforces PIT physically; the
  legacy lake enforces it *post hoc* (`validate_feature_frame`) and LH009
  (direct `pl.read_parquet`/`pl.scan_parquet` outside `data/`/`pit/`) is still
  **warning-severity** with ~37 call sites in 20 files unmigrated. A read path
  that can silently bypass the choke point is the single largest residual leak
  surface.
- **G2 — Knowledge-time is collection-time for most public sources.** Adapters
  stamp `available_time = utc_now()` at ingest; `pit_frame()` *defaults*
  `available_time` to `event_time` when a source omits it. Both destroy the
  publication lag that look-ahead-bias prevention depends on: backfilled
  fundamentals/macro become "known" at their event date or at ingest date,
  not at their historical release date. There is no ALFRED-style vintage store
  for revised macro series.
- **G3 — No unified as-of join utility with tolerance semantics.** Polars
  `join_asof` is called ad hoc across sleeves/portfolios/microstructure; LH004
  only catches the worst form (as-of on `event_time` instead of publish time).
  Tolerance, `by`-grouping, and `known_at`-keyed joins are left to each call
  site, and restatement semantics (`pit/corrections.py`) apply only inside the
  vault.

Fixing G2 is the highest-leverage change: the vault, watchdog, and lint stack
can only enforce `known_at <= decision_time` if `known_at` *means the
historical release time*. Today, for most non-exchange sources, it doesn't.

---

## 2. SOTA practice summaries (with citations)

### 2.1 Bitemporal databases & point-in-time read semantics

**The core model.** Snodgrass & Ahn's taxonomy [1] separates **valid time**
(when a fact was true in the world) from **transaction time** (when it was
recorded in the database). A **bitemporal** relation carries both; a
point-in-time query pins transaction time at the decision instant `t_known`
and valid time at the observation instant `t_event`:

\[ \{(e, v) \mid \text{valid\_from} \le t_{\text{event}} < \text{valid\_to} \;\wedge\; \text{known\_at} \le t_{\text{known}}\} \]

TSQL2 [2] operationalized this; **SQL:2011** standardized it as `SYSTEM_TIME`
(transaction time) vs `APPLICATION_TIME` (valid time) periods, with
`FOR SYSTEM_TIME AS OF` syntax [3]. Commercial engines (Oracle Flashback /
Temporal Validity, SQL Server temporal tables, MariaDB system-versioning)
implement exactly this pair of axes. The quant-ML ecosystem converged on the
same shape independently: Microsoft's **qlib** ships a dedicated PIT database
where every fundamental field is a `(value, date, period, quarter,
release_date)` tuple and reads resolve to the latest `release_date <= asof`
[4]; QuantConnect Lean ships point-in-time financial statements keyed by
filing date rather than period end [5].

**Why both axes, not one.** Valid-time-only data silently restates history
(the classic backfill leak: today's corrected revenue replaces the number a
trader saw in 2019). Transaction-time-only data cannot answer "what was true
in the world as of decision date" across vintages. Restatement-safe research
needs the pair: append-only rows `(event_time, known_at, value)` where
corrections are **new rows with later `known_at`**, never overwrites — the
`pit/corrections.py` design in this repo [2].

### 2.2 Look-ahead bias: publication lags, restatements, and compounding

**Publication lag is a first-order effect.** US periodic reports are due on
SEC-mandated deadlines *after* period end: Form 10-K within 60/75/90 days
(large accelerated / accelerated / other filers) and Form 10-Q within 40/45
days [6]. Banz & Breen [7] showed as early as 1978 that prediction studies
using accounting data leak badly unless announcement dates — not period ends —
drive availability. Modern anomaly work (e.g., Novy-Marx & Velikov [8]) treats
PIT construction as a prerequisite, and Asness & Frazzini's quality factor
[9] is explicitly built on *published* data with conservative lags precisely
because restated/lagged fundamentals are the difference between an anomaly
and an artifact. Fundamental dissemination lags of 60–90 days mean a backtest
that assumes instant availability effectively trades on information ~1–2
quarters early.

**Compounding lags.** Lag compounds across joins: a signal built from
(period-end fundamentals) × (index membership) × (share count) × (macro
release) leaks if *any* leg uses the wrong timestamp. The standard defense is
a single choke point — every frame carries `(event_time, available_time)` and
every read filters `available_time <= t_asof` at the storage boundary, not in
strategy code [1][2]. This repo encodes exactly that contract
(`docs/DATA_CONTRACTS.md`, `data/point_in_time.py`).

**Restatements.** A restatement makes the *old* value historically correct and
the *new* value currently correct; only an append-only bitemporal store can
serve both. Overwriting in place retroactively upgrades the backtest with
hindsight-corrected data — a leak invisible to any single-axis schema [3][4].

### 2.3 Survivorship bias: the defunct-fund / delisted-security problem

Elton, Gruber & Blake [10] measured survivorship bias in mutual-fund databases
at roughly **0.9% per year** of reported performance — the same order of
magnitude as many published "anomalies". The equity analogue is the
delisting-return problem: Shumway [11] showed that ignoring delistings biases
size-strategy results materially because performance-related delistings
(bankruptcy, distress) are systematically negative. Carhart [12] and
Goetzmann [13] confirmed persistence results are heavily contaminated by
survivorship in fund data. The fix is a **PIT universe**: membership is a
function of the decision date, built from listing/delisting events, ticker
changes, and (for equities) trailing-liquidity screens computed on data
available at that date — the CRSP/Sharadar model of survivorship-free
universes [14]. A hard-coded ticker list is survivorship bias by
construction (this repo's LH005 rule exists precisely to forbid it).

### 2.4 Revision-aware (vintage) macro data

Macroeconomic series are **revised**: the first estimate of GDP/CPI/payrolls
differs from the third, and annual benchmark revisions rewrite years of
history. Croushore & Stark's Real-Time Data Set for Macroeconomists (RTDSM)
[15] and the St. Louis Fed's **ALFRED** archive [16] store every published
*vintage* — the entire series as it looked on each release date. Diebold &
Rudebusch [17] demonstrated that forecasting evaluation against revised data
overstates real-time performance; Faust & Wright's inflation-forecasting
survey [18] makes vintage discipline standard practice. The engineering
pattern: a macro observation is `(series_id, reference_period, vintage_date,
value)` with `vintage_date` as the knowledge axis; a PIT read resolves
`latest vintage_date <= t_asof`. Backfilling from a vendor's *current* series
without vintages silently injects restatement look-ahead into every macro
feature downstream.

### 2.5 Leakage detection in ML pipelines: the taxonomy

Kapoor & Narayanan [19] (*Leakage and the reproducibility crisis in
machine-learning-based science*, Patterns 2023) audited >300 papers using
standard ML libraries and found **leakage or reproducibility problems in most
of them**, proposing an 8-type taxonomy: (1) training-set leakage into test,
(2) test-set leakage into training (incl. preprocessing fit on full data),
(3) random-sampling artifacts, (4) grouping/duplication failures,
(5) temporal leakage (the finance case: future information in features),
(6) data-collection/annotation leakage, (7) evaluation-protocol leakage
(metric computed with oracle knowledge), (8) **leakage in the pipeline itself**
(default library behaviors — e.g., pandas `.shift(-1)`, centered windows,
imputation with full-sample statistics). Kaufman et al. [20] (*TKDD* 2012)
gave the foundational formulation — leakage = information from *outside the
legitimate training set* used to create the model — and the detection toolkit:
**permutation/shuffle tests** (shuffling the label-time mapping should destroy
skill; if it doesn't, something is leaking), **windowed evaluation**, and
feature-legitimacy review. Zliobaitė [21] formalized the validation-design
view: the test protocol must replicate the operational information timeline.
Static analysis is the complement: the industry pattern (LeakLint-class tools,
H2O Driverless AI's leakage guards, feature-store contract checks) is to
**lint for known leaky idioms** (`shift(-k)`, `center=True`, `bfill`,
fit-before-split, `.mean()` over full frames) rather than rely on reviewer
vigilance. This repo's `leakage/ast_scan.py` + LH001–LH014 registry is a
direct instance of that pattern, and its LH001/002/003/010 map onto Kapoor &
Narayanan types 8/5/2/8 respectively.

### 2.6 Timestamp alignment & as-of joins

Event data (trades, filings, releases) is irregular; decision grids are
regular. The **as-of join** — for each left row, take the most recent right
row with `key <= left_key` — is the primitive that aligns them without
introducing look-ahead, provided it keys on the *knowledge* axis. Semantics
that matter in practice [22][23]:

- `strategy="backward"` (default; last value at-or-before) is the only
  PIT-safe default. `forward`/`nearest` pull future rows into past decisions
  and must be explicitly justified (e.g., matching a trade to the *next*
  quote for TCA, never for features).
- **Tolerance**: an unbounded backward join silently resurrects stale values
  (a security-master attribute from 3 years ago). Production joins carry
  `tolerance=` and treat misses as null → explicit staleness policy.
- **`by` grouping**: as-of without `by=` across a multi-security frame is a
  cross-sectional leak (matching security A's timestamp to security B's row).
- DuckDB's `ASOF JOIN` and Polars' `join_asof` implement the same algebra;
  both require sorted keys and both apply the join *after* any filter, so the
  `available_time <= t_asof` filter must be applied to the right frame
  **before** the join, or the join itself becomes the leak vector (this is
  why this repo's LH004 forbids as-of on `event_time` and demands
  `known_at`-keyed joins).

### 2.7 López de Prado: data structures, labeling, purging (AFML)

*Advances in Financial Machine Learning* [24] is the reference treatment of
the ML-side of this lane:

- **Ch. 2 — Financial data structures**: bars should be sampled by
  information content (volume/dollar/run bars, or volume/imbalance clocks),
  because time-sampled bars of irregularly-arriving information are
  non-stationary; sampling scheme choice is itself a leakage surface (a bar
  "closes" only when its information arrives — computing it mid-bar uses
  future-of-the-tick data).
- **Ch. 3 — Labeling**: the **triple-barrier** method (volatility-scaled
  upper/lower profit-taking/stop barriers + vertical time barrier) replaces
  fixed-horizon returns; **meta-labeling** separates direction (primary
  model) from sizing/confidence (secondary model), which confines label
  leakage to one stage.
- **Ch. 4 — Sample weights**: label **concurrency** — overlapping labels
  mean one observation's outcome informs several samples; average uniqueness
  and return-attribution weights prevent a single event from dominating.
- **Ch. 7 — Purged & embargoed CV**: because labels look forward, any train
  sample whose label window overlaps a test window leaks; **purging** removes
  those train samples, and **embargo** additionally drops a buffer *after*
  each test window to suppress serial-autocorrelation leakage.
- **Ch. 12 — CPCV**: combinatorial purged cross-validation generates
  backtest-path distributions from a single dataset, feeding DSR/PBO
  deflation (the statistics side lives in `docs/SOTA/03`-lane material and
  `quant_fund/reality`).

The data-engine obligation: **labels and features must share the PIT
contract**, and purging/embargo must be derived from the *actual* label
horizons stored with the data — which requires the labeling layer to record
its forward window in the frame, not in tribal knowledge.

### 2.8 The leaky-oracle blind spot (Gençay 2026)

Gençay [25] (arXiv:2608.27734) showed that **statistical deflation cannot
detect structural look-ahead**: a strategy with even a half-normal future-
information leak passes DSR/PBO because those methods correct for *search*
over trials, not for illegitimate information inside every trial. The fixes
are structural: (a) **leaky-oracle red teaming** — inject a deliberate future
leak into a synthetic control and verify the evaluation stack flags it, and
(b) **search-trial-count deflation with immutable trial logging**. This repo
already implements both in `src/quant_fund/validation/leakage_redteam.py`
(SYNTHETIC correctness harness, labeled as such) and tracks it in
`docs/RESEARCH_REFERENCES.md`. The residual exposure is that the red team
validates the *evaluation* stack; it cannot catch a leak introduced upstream
in *data construction* — which is exactly what gaps G1–G3 below would allow.

---

## 3. Audit of the current data pipeline

### 3.1 Data-flow map (verified by inspection)

```
external sources                       research / eval consumers
───────────────                        ─────────────────────────
data/sources/adapters.py               pipeline/, research/, paper/,
  (binance, kraken, coinbase,          hedge_lab/, microstructure/,
   fred, treasury, sec-edgar,          sleeves/, portfolio/ ...
   coingecko, universe snapshots)         ▲                ▲
        │ normalize.py (schema +          │ guarded reads  │ direct parquet
        │ available_time enforcement)     │ (pit/shim.py)  │ reads (~37 sites,
        ▼                                 │                │ LH009 warning)
data/sources/storage.py ──► data/lake.py (content-addressed bronze parquet)
        │                         │
        │                         ▼
        │                 pit/vault.py  PitVault.asof(name, t)
        │                   • write-once appends (.append-lock.sqlite3)
        │                   • SHA-256 hash-chained manifests (pit/manifest.py)
        │                   • restatements as appends (pit/corrections.py)
        │                   • fail-closed VaultUnavailableError
        │                         │
        │                         ▼
        │                 pit/frame.py PitFrame(max_known_at, content_sha256,
        │                         │        validate(decision_time))
        │                         ▼
        │                 leakage/watchdog.py LeakageWatchdog.observe()
        │                   • asserts max known_at <= decision_time per read
        │                   • strict mode: missing watermark fails closed
        │                   • max_known_at_consumed → proof bundle manifest
        │                         ▼
        │                 proofcore DataAccessRecord → Merkle proof bundles
        │                   (receipts/ — immutable evidence)
        ▼
fx1/data/sources/ingest.py
  • spec.requires_as_of gate: rejects un-dated market data
  • FX1_DATA.md / FX1_DATASOURCES.md contracts
```

Parallel post-hoc validation for the legacy path: `data/point_in_time.py`
(`require_pit_columns`, `filter_available`, `filter_trailing_returns_asof`,
`validate_feature_frame`) and `data/universe.py`'s
`assert_universe_not_from_future`.

### 3.2 What is already correct (do not regress)

| Guard | Location | SOTA alignment |
|---|---|---|
| Bitemporal vault, `asof(t)` filters `known_at <= t` before caller access | `pit/vault.py` | §2.1 SQL:2011 `SYSTEM_TIME AS OF` semantics; qlib PIT [3][4] |
| Restatements as appends, `RestatementPolicy.LATEST_KNOWN` / `STRICT_FIRST`, lazy `select_asof` pushdown | `pit/corrections.py` | §2.1/§2.4 vintage pattern [15][16] |
| Hash-chained manifests, part-hash verification, atomic append lock, crash recovery | `pit/manifest.py` | tamper-evident evidence; ties into receipt immutability (SOTA 10) |
| Runtime watchdog: per-read `max_known_at <= decision_time`, strict fail-closed, watermark exported to proof bundles | `leakage/watchdog.py` | §2.1 choke-point enforcement; exceeds most published setups |
| 14-rule AST leakage linter with severity levels and cited allowlists; clean-src gate (`tests/unit/test_leakage_clean_src.py`) | `leakage/rules.py`, `leakage/ast_scan.py` | §2.5 static-analysis pattern [19][20]; LH001↔type 8, LH004↔§2.6, LH005↔§2.3, LH007/008↔honesty contract |
| PIT universe: membership as-of decision date, delistings, ticker changes, trailing-ADV liquidity screens, `assert_universe_not_from_future` | `data/universe.py` | §2.3 survivorship-free construction [10][11][14] |
| Temporal security master: `valid_from`/`valid_to`/`available_time`, `snapshot_asof` | `data/security_master.py` | §2.1 valid-time axis |
| Source-frame invariant `event_time <= available_time <= ingested_time`, fail-closed `SourceError` | `data/sources/base.py::pit_frame` | §2.2 timestamp-chain contract |
| fx-1 `requires_as_of` ingest gate: un-dated market data rejected with explicit error | `src/fx1/data/sources/ingest.py` | §2.5 type-8 pipeline leakage blocked at ingest |
| Leaky-oracle red team + trial-count deflation (SYNTHETIC, labeled) | `validation/leakage_redteam.py` | §2.8 [25] — rare in practice; ahead of most shops |
| Cryptographic read receipts: every vault read is a `DataAccessRecord` in Merkle proof bundles | `pit/vault.py` + `proofcore/` | §2.5 immutability of trial logging [25] |

### 3.3 Findings (evidence-backed gaps)

**G1 (high) — The lake path is guard-by-convention, not guard-by-construction.**
`LH009` is warning-severity "until the call-site migration wave lands"
(`leakage/rules.py`). Measured at HEAD: **~37 `pl.read_parquet`/`pl.scan_parquet`
call sites across 20 files** outside `data/`+`pit/` (heaviest:
`paper/ledger.py` ×8, `cli/report_cmds.py` ×4, `cli/micro_cmds.py` ×4,
`cli/book_cmds.py` ×3; plus `pipeline/`, `research/`, `hedge_lab/`,
`microstructure/`, `simtest/`, `validation/gates.py`, `api/app.py`).
`pit/shim.py::guarded_read_parquet` exists and wraps these reads in a
`PitFrame` with `validate(decision_time)`, but nothing forces its use. Any of
these sites can load a frame containing rows published after the decision time
and no runtime guard fires — the watchdog only observes vault reads. This is
the Kapoor & Narayanan type-8 failure mode [19]: the leak isn't in the
strategy, it's in an unguarded I/O default.

**G2 (high) — `available_time` is not historical release time for public
sources.** Three compounding weaknesses:

1. `data/sources/base.py::pit_frame` defaults `available = event` when a row
   omits `available_time` (only `normalize.py`'s release path raises
   "release time must be explicit"). For any lagged series (fundamentals,
   filings, macro), the default silently asserts zero publication lag — the
   exact Banz–Breen failure [7] — and the chain check
   `event <= available <= ingested` passes.
2. `data/sources/adapters.py` stamps `available_time = utc_now()` at ~9
   non-exchange sites (FRED/Treasury/SEC-EDGAR-class adapters, universe
   snapshots, reference lists). On a historical backfill this means
   knowledge-time = ingest-time: either too late (live) or, after a re-ingest
   of old data, *later than any decision date under test* — the vault will
   happily hide such rows forever, or worse, a lake-path read (G1) sees them
   with the wrong stamp. Neither reproduces the historical release calendar
   (§2.2 [6]) or revision vintages (§2.4 [15][16]).
3. The universe-snapshot adapter is self-documented as
   "survivorship-truncated … stamped `available_time = now`"
   (`adapters.py` ~L338): today's listing snapshot used at past decision
   dates is §2.3 survivorship bias [10][14]. `data/universe.py` does the right
   thing when fed event-driven listing/delisting data; the gap is that the
   *source* of membership for public adapters is a now-stamped snapshot.

**G3 (medium) — As-of joins are ad hoc; no tolerance/lag utility, incomplete
lint coverage.** `join_asof` appears in `data/universe.py`, sleeves,
portfolios, and microstructure with per-site `by=`/`strategy=` choices; only
`on="event_time"` misuse is linted (LH004). There is no repo-level helper
enforcing: backward-only default, mandatory `by=` on multi-security frames,
mandatory `tolerance` (or explicit `tolerance=None` acknowledgment of
unbounded staleness), and pre-join `available_time <= t_asof` filtering
(§2.6 [22][23]). The allowlist-based LH004 exemptions
("frames that are already PIT-filtered") are exactly the tribal-knowledge
pattern a typed utility would replace.

**G4 (medium) — Purge/embargo and label horizons are not carried by the data
layer.** Labels record their forward windows inside `labels/` engines, but
the feature side has no schema field linking a feature frame to the label
horizon it will be purged against; purged-CV construction (AFML Ch. 7/12
[24]) is re-derived per backtest lane. Concurrency weights (Ch. 4) likewise
live per-lane. This is a correctness-adjacent duplication risk rather than a
live leak, but it is the natural next contract after G1–G3.

**G5 (low) — Enforcement asymmetry between `normalize.py` and `base.py`.**
The two stamping paths disagree on whether `available_time` is mandatory
(§G2.1). One contract, one implementation: `pit_frame` should fail closed and
require an explicit, auditable lag policy (see P2 snippet).

---

## 4. Adoption plan (concrete)

Landing order: **P1 → P2 → P3 → P4**, with P5/P6 in parallel. P1 and P2 are
prerequisites for each other's value: guarded reads without true release
times enforce the wrong invariant faithfully; true release times without a
choke point can still be bypassed.

### 4.1 P1 (G1) — Close the lake read path; flip LH009 to error

Mechanics, in order:

1. Migrate the 37 sites file-by-file to `pit.guarded_read_parquet(path, t,
   dataset=...)` (or `pit.lake_asof` where a `Lake` handle is in scope).
   Reads that are genuinely non-temporal (ops tooling, `pipeline/doctor.py`
   diagnostics) move to an explicit `LH009_EXEMPT_GLOBS` entry with a cited
   justification, matching the existing allowlist discipline.
2. Add a **migration-burndown gate**: a test that asserts the count of
   LH009 findings is ≤ the previous recorded count (ratchet), so the wave
   cannot regress while in flight.
3. Flip `LH009` severity `warning → error` in `leakage/rules.py` the moment
   the ratchet hits zero outside exempt globs (the rule docstring already
   promises this).

```python
# pit/shim.py — proposed strict-mode addition (fail closed on missing watermark)
def guarded_read_parquet(
    path: Path, t: datetime, *, dataset: str,
    watchdog: WatchdogProtocol | None = None,   # NEW: same observer as vault
    recorder: DataAccessRecorder | None = None, # NEW: reads enter proof bundles
) -> PitFrame:
    frame = _to_pit_frame(pl.read_parquet(path), dataset=dataset, t=t)
    frame.validate(t)
    if watchdog is not None:
        watchdog.observe(_record(frame, dataset, t, path), t)
    return frame
```

Attaching the same watchdog/recorder the vault uses extends the
`max_known_at_consumed` watermark and Merkle read-receipt coverage to lake
reads — one evidence channel for both architectures.

### 4.2 P2 (G2/G5) — True release times: explicit lag policy + vintage store

**(a) Fail-closed stamping.** Remove the silent `available = event` default;
require each adapter to declare a `ReleaseLag` policy per dataset:

```python
# data/sources/base.py — proposed contract
@dataclass(frozen=True)
class ReleaseLag:
    """How event_time maps to historical release (knowledge) time."""
    kind: Literal["instant", "fixed", "calendar", "vintage"]
    lag: timedelta | None = None            # kind="fixed": e.g. 10-Q => 40d [6]
    calendar: Callable[[datetime], datetime] | None = None  # release schedules
    assume_at_event: bool = False           # ONLY for exchange-realtime legs;
                                            # must be cited in DATA_SOURCES

def resolve_available(event: datetime, policy: ReleaseLag) -> datetime: ...
```

Rows arriving without `available_time` then resolve through the policy;
`assume_at_event=True` is allowed only for exchange-realtime adapters
(trades/quotes/book), audited in `docs/DATA_SOURCES.md` per dataset. This
closes G5 by making `normalize.py`'s rule the universal rule.

**(b) Vintage tracking for revised series (ALFRED pattern [15][16]).**
Macro datasets gain a vintage axis stored as ordinary vault appends —
`pit/corrections.py` already gives the right semantics
(`(event_time, known_at)` append-only with `select_asof`): each release
snapshot is a restatement row with `known_at = release_timestamp`,
`event_time = reference_period`. Contract test: for any macro dataset,
`pit.stats` must show ≥1 revision per revised series per year, or the
adapter must declare `revisions=False` explicitly (first-print storage).

**(c) Survivorship-safe universe feeds [10][14].** Universe membership for
public sources must come from listing/delisting *events* (which
`data/corporate_actions.py` + `data/universe.py` already consume), not
now-stamped snapshots. The snapshot adapter keeps `available_time = now` but
is relabeled in `docs/DATA_SOURCES.md` as **live-ops-only** (paper trading
forward), barred from historical research by a dataset-level flag checked in
`guarded_read_parquet`.

### 4.3 P3 (G3) — Unified as-of join utility + LH015

```python
# data/asof.py — proposed single implementation
def asof_join(
    left: pl.DataFrame, right: pl.DataFrame, *,
    left_on: str, right_on: str = "known_at",   # knowledge axis by default
    by: str | list[str] | None = None,          # mandatory for multi-security
    tolerance: timedelta | None = ...,          # sentinel forces explicit None
    strategy: Literal["backward"] = "backward", # forward/nearest need pit.asof_forward
    right_asof: datetime | None = None,         # pre-join PIT filter (§2.6)
) -> pl.DataFrame:
    """Backward as-of join with PIT-safe defaults.

    - Filters right on ``known_at <= right_asof`` BEFORE joining.
    - Raises on multi-security right frames without ``by``.
    - Raises on missing ``tolerance`` (pass ``None`` deliberately for
      unbounded staleness; the decision is then recorded in the column
      ``{left_on}_staleness``).
    """
```

Then:

- **LH015 (error)**: raw `.join_asof(` outside `data/asof.py` and the
  existing cited allowlist — the same allowlist-with-justification pattern
  as LH001–LH014. Existing legitimate sites (`data/universe.py`'s vectorized
  membership join) migrate to the utility or the allowlist with citations.
- **LH016 (warning)**: `strategy="forward"`/`"nearest"` anywhere — human
  review, mirroring LH013's severity philosophy.
- Contract tests: (i) `asof_join` with `right_asof=t` never returns a right
  row with `known_at > t` (property test over synthetic frames — labeled
  SYNTHETIC); (ii) tolerance miss → null, not resurrection; (iii) `by=`
  omission on multi-security frames raises.

### 4.4 P4 — Contract-test battery (schema-level invariants)

Add to the existing gate lanes (`make test` / `make fx1-gate`):

1. **Chain invariant** (already in `pit_frame`; extend to vault reads):
   every PIT frame satisfies `event_time <= known_at/available_time <=
   ingested_time`, tz-aware everywhere (watchdog already fails closed on
   naive timestamps).
2. **Decision-time round trip**: for a sample of vault datasets, read at
   `t`, assert `PitFrame.validate(t)` passes and
   `validate(t - epsilon)` fails iff any row has `known_at ∈ (t-ε, t]` —
   pins the boundary semantics of `asof` (inclusive).
3. **Lag-policy audit**: table-driven test asserting every registered source
   adapter declares a `ReleaseLag` and that no non-realtime adapter uses
   `assume_at_event=True`.
4. **Vintage presence**: P2(b) macro-vintage test.
5. **Red-team extension [25]**: feed the leaky-oracle harness a *data-layer*
   leak (a frame with one row stamped `known_at` after decision time) through
   `guarded_read_parquet` and assert the watchdog fires — extending the
   existing eval-side red team upstream into data construction (SYNTHETIC).

### 4.5 P5 (G4) — Label-horizon metadata & shared purge/embargo

Record the forward horizon on labeled frames (`label_horizon: timedelta`
column or frame schema field) at labeling time (`labels/` engines), then ship
one utility — `research/purge.py::purged_folds(labels, horizon, n_splits,
embargo)` implementing AFML Ch. 7 purging + embargo and Ch. 12 CPCV
combinatorics [24] — consumed by all backtest lanes. Average-uniqueness
weights (Ch. 4) fall out of the same concurrency computation. This removes
per-lane re-derivation and makes the purge window a function of *data*, not
of each lane's memory.

### 4.6 P6 — Reporting & burndown

`leakage/report.py` gains: LH009 site count (ratchet target 0), adapters
without declared `ReleaseLag` (target 0), `join_asof` sites outside
`data/asof.py` (target = allowlist size). Surfaced in the existing leakage
scan gate so the burndown is visible in CI, matching the repo's
work-tracking conventions (`INFLIGHT`, `day_grind_progress.md`).

### 4.7 Suggested landing order

| Step | Change | Gate touched |
|---|---|---|
| 1 | P2(a) `ReleaseLag` contract + remove silent default (G2.1/G5) | `make test`, `make lint` |
| 2 | P3 `data/asof.py` + LH015/LH016 + allowlist migration | `make test`, leakage clean-src gate |
| 3 | P1 guarded-read migration wave + ratchet test | LH009 burndown |
| 4 | P1 flip LH009 → error | clean-src gate (zero errors) |
| 5 | P2(b) vintage storage for macro; P2(c) universe relabel | `make test`, contract battery |
| 6 | P4 battery incl. data-layer red-team extension | `make test-full` |
| 7 | P5 purge/embargo utility; P6 reporting | lane adoption PRs |

Steps 1–2 are small and independent; 3–4 are the long wave but fully
mechanical; 5–7 build on the contract from 1–2.

---

## 5. References

[1] Snodgrass, R. & Ahn, I. — *A Taxonomy of Time in Databases*, ACM TODS
11(3), 1986; and Snodgrass, R. — *Developing Time-Oriented Database
Applications in TSQL2*, Morgan Kaufmann, 1999.
[2] This repo — `src/quant_fund/pit/` design: `vault.py` (as-of reads,
append-only writes), `corrections.py` (restatement append semantics),
`manifest.py` (hash-chained evidence); `docs/DATA_CONTRACTS.md`.
[3] ISO/IEC 9075-2:2011 (SQL:2011) — temporal support: `SYSTEM_TIME`
(transaction time) vs `APPLICATION_TIME` (valid time), `FOR SYSTEM_TIME AS
OF`. Vendor implementations: Oracle Flashback/Temporal Validity, SQL Server
temporal tables, MariaDB system-versioning.
[4] Microsoft qlib — Point-in-Time database documentation:
https://qlib.readthedocs.io/en/latest/component/data.html#point-in-time-pit-database
[5] QuantConnect Lean — Point-in-time financial statements / fundamentals
data model: https://www.quantconnect.com/docs/v2/lean-engine/datasets
[6] US SEC — periodic report deadlines: Form 10-K 60/75/90 days by filer
status (Release 33-8128), Form 10-Q 40/45 days (Rule 13a-13 / Reg S-X
Art. 10); late-filing relief Rule 12b-25.
[7] Banz, R. & Breen, W. — *Prediction of Corporate Events Using Securities
Prices*, Journal of Financial Economics, 1978 (announcement-date vs
period-end availability).
[8] Novy-Marx, R. & Velikov, M. — *A Taxonomy of Anomalies and Their Trading
Costs*, Review of Financial Studies, 2016 (PIT data as prerequisite).
[9] Asness, C. & Frazzini, A. — *Quality Minus Junk*, Review of Accounting
Studies, 2013 (published-data construction discipline).
[10] Elton, E., Gruber, M. & Blake, C. — *Survivorship Bias in Mutual Fund
Performance*, Review of Financial Studies 9(4), 1996 (~0.9%/yr bias).
[11] Shumway, T. — *The Delisting Bias in CRSP Data*, Journal of Finance,
1997.
[12] Carhart, M. — *On Persistence in Mutual Fund Performance*, Journal of
Finance, 1997.
[13] Goetzmann, W. — *Performance, Timing, and Survivorship Bias*, Journal of
Portfolio Management, 1999 (survivorship in long-horizon fund data).
[14] CRSP survivorship-free US databases; Sharadar (Nasdaq Data Link) SEP/
SF1 point-in-time fundamentals & ticker-trading-history universe
documentation.
[15] Croushore, D. & Stark, T. — *A Real-Time Data Set for Macroeconomists*,
Journal of Business & Economic Statistics 19(1), 2001 (RTDSM).
[16] Federal Reserve Bank of St. Louis — ALFRED (Archival Federal Reserve
Economic Data): https://alfred.stlouisfed.org/
[17] Diebold, F. & Rudebusch, G. — *Forecasting Output with the Composite
Leading Index: A Real-Time Analysis*, JBES, 1991.
[18] Faust, J. & Wright, J. — *Forecasting Inflation*, Handbook of Economic
Forecasting Vol. 2A, 2013 (vintage discipline in forecast evaluation).
[19] Kapoor, S. & Narayanan, A. — *Leakage and the Reproducibility Crisis in
Machine-Learning-Based Science*, Patterns 4(9), 2023 (arXiv:2109.07138);
8-type leakage taxonomy.
[20] Kaufman, S., Rosset, S., Perlich, C. & Stitelman, O. — *Leakage in Data
Mining: Formulation, Detection, and Avoidance*, ACM TKDD 6(4), 2012
(permutation/shuffle tests, windowed evaluation).
[21] Zliobaitė, I. — *How Good Is the Evidence? Formalising the Validation
Protocol of Predictive Models* (validation must replicate the operational
information timeline), 2017.
[22] Polars — `DataFrame.join_asof` / `LazyFrame.join_asof` documentation
(strategy backward/forward/nearest, `tolerance`, `by`):
https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.join_asof.html
[23] DuckDB — `ASOF JOIN` documentation (irregular-interval alignment):
https://duckdb.org/docs/sql/query_syntax/from.html#as-of-joins
[24] López de Prado, M. — *Advances in Financial Machine Learning*, Wiley,
2018: Ch. 2 (financial data structures / information-sampled bars), Ch. 3
(triple-barrier labeling, meta-labeling), Ch. 4 (sample uniqueness,
concurrency weights), Ch. 7 (purged k-fold CV + embargo), Ch. 12 (CPCV).
[25] Gençay (2026), arXiv:2608.27734 — leakage-safe, search-aware
evaluation: leaky oracles survive DSR/PBO; structural look-ahead exclusion +
search-trial-count deflation as fixes. Implemented in this repo at
`src/quant_fund/validation/leakage_redteam.py` (SYNTHETIC harness; see
`docs/RESEARCH_REFERENCES.md`).
