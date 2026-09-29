# 19 — Deep Code Quality Review: `src/quant_fund`

**Lane:** read-only code-quality audit of the dipcatcher harness (data engine,
evaluation, verification). No source file was modified; this document is the
only deliverable.

**Scope measured:** 670 modules, 152,578 lines, 48 top-level subpackages,
1,632 intra-package import edges.

**Date:** 2026-09-28 · **Commit at review start:** `3dafeb7`

---

## 0. Critical caveat — the working tree is mid-restore and unstable

This must be read before any finding below, because it changes the
interpretation of the gate results.

During this review the working tree **changed underneath the audit**. A
concurrent restore process was active:

| Path | State | Tracked? |
|---|---|---|
| `pyproject.toml` | flapped between two versions (see below) | tracked |
| `src/src/quant_fund/` | 151 stale pre-split files | **untracked** |
| `tests/tests/` | 457 stale test files | **untracked** |
| `configs/configs/` | 5 stale YAMLs | **tracked** |
| `scripts/scripts/` | 8 files, 3 with no counterpart | **tracked** |
| `.backup-prerestore/` | full mirror incl. `mirrors/tests_tests-pass2/` | untracked |

`pyproject.toml` was observed at two distinct contents within ~30 minutes
(mtime 15:09:29 → 15:22:31; SHA-256 `7146394…` for the short version):

| | Working tree (regressed, 147 lines) | `HEAD` (369 lines) |
|---|---|---|
| `name` | `"dipcatcher"` | `"fx-1"` |
| `version` | `version = "1.0.0"` (literal) | `dynamic = ["version"]` → `src/fx1/__init__.py` |
| `numba>=0.60` | **absent** | present (line 37) |
| `exchange-calendars==4.13.2` | **absent** | present (line 38) |
| ruff `select` | no `C901` | `["E","F","I","UP","B","SIM","C901"]` |
| `mccabe.max-complexity` | **absent** | `74` |
| `warn_return_any` | `false` | `true` |
| `disable_error_code = ["override"]` | **absent** | present |
| `testpaths` | `["tests"]` | 5 explicit dirs, `tests/tests` excluded |
| coverage `fail_under` | `70` | `80` |

**Consequence:** if the 147-line variant lands, the wheel silently loses two
*actually imported* dependencies (`numba` at 5 import sites,
`exchange_calendars` at 7), the coverage gate drops 10 points, the complexity
ceiling disappears, and pytest begins collecting the 457-file stale
`tests/tests/` mirror. `AGENTS.md` says `uv.lock` is authoritative and
`--frozen` must pass; a `[project]` rename plus dropped deps breaks that.

This is recorded as **P0-1** because it is not a style issue — it is a
packaging/gate integrity issue, and it makes every other gate result in this
document time-dependent.

---

## 1. Tool results (recorded, nothing fixed)

### 1.1 ruff

```
uv run ruff check src --statistics
→ (empty; exit 0)          # with the 369-line HEAD config
→ 1  F841 unused-variable  # earlier, before _fast_kernel.py was rewritten at 15:10
```

`ruff check src` passes. Two notes:

- ruff **skips** `src/src/` entirely (confirmed: `ruff check src/src` → *"No
  Python files found under the given path(s)"*, while
  `--no-respect-gitignore` → `1 I001`). So the stale 151-file tree is invisible
  to lint yet still shipped inside `src/`.
- Under HEAD's ruleset plus `RUF100`, **21 `noqa` directives are unused**
  because they suppress rules that are not in `select`: `BLE001` (16×),
  `F401` (5×), `C901` (2×), plus `S310`, `N802`, `S101`, `E402`, `I001`,
  `F841`. See §P1-4.

### 1.2 mypy

```
uv run mypy src/quant_fund
→ Found 15 errors in 2 files (checked 670 source files)
```

| Count | Location | Code | Nature |
|---|---|---|---|
| 14 | `data/sources/adapters.py:111, 214, 268, 346, 421, 454, 495, 563, 607, 647, 676, 706, 749, 758` | `[override]` | `fetch(*, series_id: str, …)` incompatible with `SourceAdapter.fetch(**kwargs: Any)` — **suppressed in HEAD config**, surfaces only in the regressed variant |
| 1 | `backtest/_fast_kernel.py:28` | `[no-redef]` | genuine defect: `# type: ignore[misc]` does not cover `no-redef` |

The single real error is a mislabeled suppression. Compare the two identical
numba-fallback shims:

```22:22:D:/dipcatcher/src/quant_fund/models/volatility.py
    def njit(*_args: Any, **_kwargs: Any) -> Any:  # type: ignore[no-redef]
```

```28:29:D:/dipcatcher/src/quant_fund/backtest/_fast_kernel.py
    def njit(*_a, **_k):  # type: ignore[misc]
        def _deco(fn):
```

`_fast_kernel.py` is **untracked** (new since HEAD). The same shim in
`fast_replay.py:64-65` also lacks annotations. This is the only file where the
numba fallback was written without the established pattern.

With `--disallow-untyped-defs` layered on, only **18 untyped defs** remain
across all 670 modules — 8 of them in `backtest/_fast_kernel.py` and
`backtest/fast_replay.py`, the rest scattered
(`simtest/faults.py:105`, `mc_engine/report.py:202`, `models/arch_fit.py:15`,
`paper/quantile_signals.py:69,160`, `pit/vault.py:74`,
`pipeline/train/splits.py:125`, `pipeline/train/ranking.py:350,363`).
The global flag is therefore ~97% affordable.

### 1.3 Complexity

```
ruff --select C901 --config "lint.mccabe.max-complexity=74"
→ 2 errors, both in src/src/ (validate_ledger_schema 78, verify_research_artifact 196)
```

The live tree satisfies the ceiling of 74 — but a ceiling of 74 is itself a
finding: it exists solely to grandfather `paper/ledger.py:validate_ledger_schema`
(see the comment at `pyproject.toml:133-136`). At a conventional ceiling of 15,
**106 functions** violate it.

---

## 2. Findings

Severity: **P0** blocks correctness/evidence integrity · **P1** systemic,
fix soon · **P2** structural debt · **P3** polish.

### P0-1 · Build/gate identity instability (see §0)
`pyproject.toml` (working tree vs `HEAD`). Two committed duplicate trees
(`configs/configs/`, `scripts/scripts/`) prove this has already happened once
before: `git log -- configs/configs` → `621023e "chore: commit leftover nested
source, tests, lockfile, and wide tape"`, `811dffc "chore: commit leftover
nested source and test trees"`.

`configs/configs/base.yaml` is a **drifted** copy, not a mirror: it lacks the
entire `robinhood_plus:` block that `configs/base.yaml:83-98` carries
(ADR-023). Anyone resolving the nested path gets silently different defaults.

### P0-2 · Honesty gates fail **open** on import failure

The repo's hard rule #4 states receipts are immutable evidence and
`verify-research` gates claims. Four catalog soft-verify functions return
`[]` — meaning *"no errors found"* — when a **deferred import** raises:

```588:590:D:/dipcatcher/src/quant_fund/research/catalog/receipt.py
    try:
        from quant_fund.microstructure.book_metrics import METRICS_REQUIRED_FINITE_KEYS
    except Exception:
        return []
```

```435:437:D:/dipcatcher/src/quant_fund/research/catalog/candle.py
    try:
        from quant_fund.microstructure.bench import FEATURE_COLS
    except Exception:
        return []
```

Same pattern at `research/catalog/candle.py:1055-1057` and
`research/fleet_eval.py:698-701`.

An empty error list is indistinguishable from a passing check. If
`microstructure` fails to import for *any* reason (typo, partial checkout,
dependency skew), every receipt passes `northset_metrics_required_keys_finite…`
and `candle_*` verification without a single assertion being evaluated. In a
system whose entire value proposition is fail-closed verification, this is the
one place it fails open — and it does so silently, with no log line (see P1-2).

Fix is ~4 lines per site: `except ImportError` (narrow), and return a
non-empty sentinel error such as `["<check>_unverifiable_import_failed"]`.

### P1-1 · `_atomic_write_*` is implemented 10 times with divergent durability

| Site | Name | fsync file | **fsync dir** | Overwrite semantics |
|---|---|---|---|---|
| `paper/ledger.py:78` | `_atomic_write_text` | yes | **yes** (`:87`) | overwrite |
| `paper/ledger.py:92` | `_atomic_write_parquet` | yes | **yes** (`:101`) | overwrite |
| `backtest/engine.py:749` | `_atomic_write_text` | yes (`:766`) | no | overwrite |
| `research/agent.py:95` | `_atomic_write_text` | yes (`:111`) | no | overwrite |
| `research/fleet_eval.py:768` | `_atomic_write_text` | yes | no | **refuses**: symlink check `:771`, content-compare `:773-776`, `FileExistsError` |
| `research/vol_bench.py:586` | `_atomic_write_text` | yes | no | overwrite |
| `data/lakehouse/store.py:316` | `_atomic_write` | — | no | bytes |
| `pit/manifest.py:81` | `_atomic_write` | — | no | bytes |
| `proof/bundle.py:232` | `_atomic_write_bytes` | — | no | bytes |
| `research/explainability/report.py:616` | `_atomic_write` | — | no | bytes |

`_fsync_directory` exists only in `paper/ledger.py:66`. Without an
`fsync` on the parent directory, the `os.replace` rename is not guaranteed
durable across a crash — exactly the failure mode receipts are supposed to
survive. Eight of ten "atomic" writers are atomic-but-not-durable.

Three *different behaviors* share one name, so a reader cannot predict whether
a call site overwrites, refuses, or checks symlinks. `receipt_v2.seal_receipt`
relies on the refusing variant; `run_backtest`'s metrics export relies on the
overwriting one.

### P1-2 · Observability is effectively absent (6 of 670 modules)

`utils/logging.py` is a competent structlog setup: JSON renderer, ISO
timestamps, `merge_contextvars`, level filtering, `PrintLoggerFactory`
(`:11-33`). It is used by exactly **6 modules**:

```
cli/data_cmds.py            models/covariance.py
pipeline/forecast/covariance.py   pipeline/forecast/decide.py
pipeline/forecast/garch.py        pipeline/forecast/realized.py
```

Zero `logging.getLogger`, zero stdlib `logging.<level>()` calls. In 152k lines
of numeric research code — including the entire `backtest/`, `research/`,
`paper/`, `portfolio/`, `validation/`, `northset/`, and `microstructure/` trees
— there is no log statement. Failures surface only as raised exceptions or, in
26 places, `print()`:

```154:156:D:/dipcatcher/src/quant_fund/hedge_lab/v2_slate.py
    print(f"CACHED {engine} fp={fp}", flush=True)
    print(f"FIT {engine} label={label} fp={fp}", flush=True)
```

Other `print()` in library (non-`__main__`, non-CLI) code:
`hedge_lab/mirror.py:286`, `hedge_lab/v2_slate.py:885,899`,
`pretrade/bench.py:279,285`, `quant_models/greeks.py:192`,
`research/forward_shadow.py:595`, `research/net_tournament.py:421`,
`research/prospective_sota.py:722`, `research/ranker_probability.py:423`,
`research/reality_sweep.py:789,974`, `research/real_benchmark.py:434`,
`robustness/smoke.py:119`.

**Two non-convergent stacks.** `observe/` is a second, complete observability
system (868 lines: `flags.py`, `install.py`, `logging.py`, `metrics.py`,
`overhead.py`, `serve.py`, `slo.py`, `tracing.py`) with its own JSON logger,
correlation IDs, secret redaction, Prometheus renderer, OTLP tracing, and SLO
histograms. It never touches `utils/logging.py`. It is opt-in via
`DIPCATCHER_OBSERVE` (`observe/flags.py:21-23`) and installs by
**monkeypatching** 12 entry points (`observe/install.py:86-100`), off by
default. So the richer stack is dark in every default run, while the wired-up
stack covers 6 modules. `observe/logging.py:34-53` duplicates the redaction +
JSON formatting that `structlog` already provides.

### P1-3 · Four parallel exception taxonomies

43 custom exception classes across 4 unrelated roots:

| Root | Defined at | Members |
|---|---|---|
| `QuantFundError` | `schemas/errors.py:6` | 8 (`PointInTimeError`, `LeakageError`, `OptimizationInfeasible`, `RiskGateRejected`, `KillSwitchActive`, `ConfigError`, `DataContractError`) |
| `ProofcoreError` | `proofcore/contracts.py:55` | 10 (`VaultError`, `VaultUnavailableError`, `ManifestError`, `ProofError`, `ProofVerificationError`, `ProofBundleError`, `SignatureUnavailableError`, `LeakageError`, `RealityFilterError`, `ProvenanceError`) |
| `AuditError` | `audit/errors.py:11` | 3 (`SignatureUnavailableError`, `ProofError`) |
| unaffiliated | — | `SourceError(RuntimeError)` `data/sources/base.py:16`, `IoError(Exception)` `data/concurrent_io.py:26`, `IoCancelled(BaseException)` `:30`, `_TerminalHTTP` ×2 (`data/adapters/stooq.py:171`, `data/adapters/yahoo_eod.py:48`), `_RequestBodyTooLarge` `api/app.py:38`, `CrashInjected`/`BrokerTimeout` `simtest/session.py:48,56`, `ReplayDivergence` `simtest/eventlog.py:22` |

The fork is severe enough that it required a **multiple-inheritance adapter**
to bridge:

```29:36:D:/dipcatcher/src/quant_fund/leakage/watchdog.py
class LeakageError(ContractsLeakageError, SchemasLeakageError):
    """Watchdog trip or error-severity scan finding (package-boundary adapter).

    Subclasses both ``contracts.LeakageError`` (PROOFCORE catch root) and
    ``quant_fund.schemas.errors.LeakageError`` so existing
    ``except PointInTimeError``/``except LeakageError`` clauses keep working
    (DESIGN.md §8.3).
    """
```

`pit/frame.py:27` does the same for `VaultUnavailableError`. Two classes named
`LeakageError`, two named `SignatureUnavailableError`, two named `ProofError` —
each pair from different roots, distinguished only by import path.

Also duplicated outright: `StaleValuationError(RuntimeError)` is defined
**twice** with different docstrings — `backtest/engine.py:41` and
`paper/loop.py:68`. They are distinct types, so `except` on one does not catch
the other, for the same domain concept.

And the taxonomy is bypassed in practice: of **5,046 `raise` statements**,
**4,117 (82%) are bare `ValueError`**, including inside modules that have a
purpose-built `ConfigError`/`DataContractError` in scope. Every validator in
`config/models.py` (30+ `model_validator`s) raises `ValueError` rather than
`ConfigError`.

*Positive note:* the 4 `except BaseException` sites are all correct —
cleanup-and-reraise (`data/sources/storage.py:63-65`, `data/lake.py:27-29`)
and cooperative-cancellation plumbing (`data/concurrent_io.py:216-222`, with
`IoCancelled(BaseException)` deliberately escaping `except Exception`). There
are **0 bare `except:`** clauses in 670 modules.

### P1-4 · Broad-except policing is comment-only

`except Exception` appears **77 times**. 16 of those sites carry
`# noqa: BLE001 — <intent>` comments asserting a deliberate fail-closed
boundary:

`api/app.py:909`, `cli/ops_cmds.py:431`, `hedge_lab/v2_slate.py:304`,
`models/robinhood_plus/compare.py:320,443`,
`models/robinhood_plus/torch_backend.py:281`,
`paper/ledger.py:773,785,794,822,844`, `paper/quantile_signals.py:577`,
`paper/sim_live.py:398`, `portfolio/optimizer.py:129,168`,
`research/sota_protocol.py:176`.

**`BLE001` is not in ruff's `select` list** (`pyproject.toml:129`), so the rule
never runs and all 16 suppressions are dead weight (flagged as `RUF100` when
the rule is enabled). The comments advertise enforcement that does not exist.
Because `RUF100` is also not selected, the vestigial `noqa`s never self-report.

Three sites swallow outright:
`models/caviar.py:106-107` (`except Exception: continue`),
`models/fracdiff.py:118` (`# ADF failure is a skip, not a crash`),
`research/fleet_eval.py:700-701` (`except Exception: pass` — silently leaves
`version == "unknown"` in the fleet evidence table).

### P2-1 · `models/covariance.py` — god module with an internally drifted DCC family

1,877 lines, 62 top-level functions, 6 distinct concerns in one file: PSD
repair (`:15-49`), shrinkage estimators (`:66-397`), factor covariance
(`:398-622`), the DCC recursion helper layer (`:773-1094`), two correlation
likelihoods (`:1096-1143`), and the estimator dispatch/registry
(`:657-772`). It is also the single largest `type: ignore` concentration
outside `research/catalog/`.

The DCC helpers were extracted — and then `dcc_gaussian` did not adopt them.
`dcc_gaussian` (`:1194-1307`) **inlines weaker copies** of five helpers:

| Helper (exists) | `dcc_gaussian` inlines instead |
|---|---|
| `_dcc_prepare_window` `:773` | `:1223-1226` |
| `_dcc_qbar` `:780` | `:1236-1240` |
| `_dcc_step_q` `:798` | `:1249`, `:1280` |
| `_dcc_r_from_q` `:790` | `:1250-1253`, `:1281-1284` |
| `_dcc_one_step_h` `:802` | `:1277-1288` |
| `_gaussian_corr_nll` `:1096` | `:1242-1262` |

The inlined likelihood **drops a guard**. The shared helper:

```1105:1108:D:/dipcatcher/src/quant_fund/models/covariance.py
    if not np.isfinite(quadratic):
        return 1e12
    return float(logdet + quadratic)
```

The inlined version at `:1257-1262`:

```1257:1262:D:/dipcatcher/src/quant_fund/models/covariance.py
            try:
                quadratic = float(z[i] @ np.linalg.solve(r, z[i]))
            except np.linalg.LinAlgError:
                return 1e12
            ll += logdet + quadratic
        return float(ll / t)
```

No `np.isfinite(quadratic)` check. A non-finite quadratic form propagates into
the accumulated log-likelihood instead of fail-closing to the `1e12` penalty —
so `dcc_gaussian` can return an optimizer result from a NaN objective while
`dcc_student_t` (`:1347`, calls `student_t_corr_nll`) and `agdcc` (`:1592`,
calls `_gaussian_corr_nll`) both fail closed. `agdcc` at `:1591-1592` is the
proof the helper layer is the intended path:

```1591:1592:D:/dipcatcher/src/quant_fund/models/covariance.py
            r = _dcc_r_from_q(q)
            term = _gaussian_corr_nll(z[i], r)
```

This is a correctness divergence between two catalog estimators that
`OptimizerConfig.covariance` treats as interchangeable named paths
(`config/models.py:317-330` documents that they "must not silently size as"
each other).

### P2-2 · `never_equate` / `never_equate_gap`: 58 copy-paste guards

1,387 + 1,400 lines, **643 identical lines**, 31 + 27 = 58 functions all named
`*_never_equate_honesty_errors(blob: object) -> list[str]` with one skeleton:

```56:73:D:/dipcatcher/src/quant_fund/research/catalog/never_equate.py
    if not isinstance(blob, dict):
        return []
    k_recon = "session_reconstructs_daily_rate"
    k_vol = "session_volume_conservation_rate"
    if k_recon not in blob or k_vol not in blob:
        return []

    errs: list[str] = []
    if k_recon == k_vol:
        errs.append("session_reconstructs_volume_conservation_keys_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    recon_hyp = by_key.get(k_recon)
    vol_hyp = by_key.get(k_vol)
    if recon_hyp != H23_HYPOTHESIS_ID:
        errs.append("session_reconstructs_daily_rate_not_bound_h23")
```

The only variation is the two key names, the two hypothesis IDs, and the error
strings. `primitives.py:68` already provides `_finite_pair(blob, key_a,
key_b)` for exactly this — and these 58 functions do not use it (callers are
`candle.py`, `northset.py`, `receipt.py`). The `by_key` dict is rebuilt from
`NORTHSET_H23_H28_SPECS` inside each function.

This is data expressed as code. A table of `(key_a, key_b, hyp_a, hyp_b,
label)` plus one generic checker would collapse ~2,700 lines to ~150, and —
more importantly — make the pair set auditable at a glance instead of requiring
58 function reads to confirm coverage.

### P2-3 · 5 real import cycles + 74 cross-module private imports

Tarjan SCC over the full import graph (module-level + relative, resolved
statically) found **5 non-trivial cycles**:

| Cycle | Mitigation | Risk |
|---|---|---|
| `quant_fund` ↔ `public` ↔ `research.agent` | `__init__.py:60-68` lazy `__getattr__` + `TYPE_CHECKING` re-exports | 3-cycle through the package root; import order sensitive |
| `backtest.engine` ↔ `backtest.fast_replay` | `engine.py:220` deferred; `fast_replay.py:37` **top-level** | tightest coupling — see below |
| `models.ranking` ↔ `models.asset_pricing` | `ranking.py:135` deferred import of `date_groups` | |
| `pipeline.forecast.history` ↔ `pipeline.forecast.state` | | |
| `research.fleet_eval` ↔ `research.receipt_v2` | both directions deferred (`receipt_v2.py:386,433`) | |

`fast_replay.py:37-44` imports **four private names** from `engine.py` at
module top level, then re-imports one of them again at `:1323`:

```37:44:D:/dipcatcher/src/quant_fund/backtest/fast_replay.py
from quant_fund.backtest.engine import (
    BacktestResult,
    StaleValuationError,
    _fast_replay_is_complete,
    _fast_replay_panel_supported,
    _target_weight_map,
    run_backtest,
)
```

**74 cross-module private-symbol imports** repo-wide, i.e. underscore-prefixed
names consumed outside their defining module. These are de-facto public API
wearing private names:

| Private symbol | Defined | Imported by |
|---|---|---|
| `_date_keys` | `metrics/cross_section.py` | **12 modules** |
| `_label_horizon` | `pipeline/train` | **7 modules** |
| `_finite` | `models/ranking.py:98` | **5 modules** (`asset_pricing`, `distribution`, `lgbm_q2`, `tail`, `volatility`) |
| `_build_result` | `backtest/engine.py:547` | `fast_replay.py:1323` |
| `_receipt_digest` | `research/verify.py` | `audit/trace.py:21` |
| `_align_ic`, `_date_key`, `_gates`, `_ic_card` | `hedge_lab/lightspeed_book.py` | `gated_race.py:20`, `v2_slate.py:40` |
| `_load_bars`, `_read_receipt`, `_seal` | `research/real_benchmark.py` | `paper/forward_shadow.py:32` |
| `_predict_ranker`, `_realized_garch_history` | `pipeline/train` | `pipeline/forecast/artifacts.py:178`, `realized.py:22` |

`_finite` is the clearest case: a 5-module dependency on
`models/ranking.py:98` that returns `Any` (untyped tuple) and is the *only*
reason `models/volatility.py:13` imports `ranking` at all — which is also what
creates the `ranking`↔`asset_pricing` cycle.

Total: **484 function-local imports**, concentrated in `__init__.py` (18),
`research/reality_sweep.py` (17), `cli/book_cmds.py` (17), `api/app.py` (13),
`cli/research_cmds.py` (12). The CLI/API ones are legitimate lazy-loading; the
`research/` and `models/` ones are cycle breakers.

### P2-4 · Untyped-blob pattern defeats typing where it matters most

219 `type: ignore` comments, **all code-scoped** (0 bare — genuinely good
discipline). But the distribution is lopsided:

| Code | Count |
|---|---|
| `arg-type` | **177** |
| `assignment` | 11 |
| `method-assign` | 7 (+3 `method-assign, assignment`) |
| `attr-defined` | 5 |
| `operator` | 4 |
| `no-redef` / `return-value` | 3 each |
| `misc` / `no-any-return` | 2 each |
| `import-not-found` / `no-untyped-def` | 1 each |

**142 of the 219** sit in `research/catalog/`: `candle.py` 38, `receipt.py` 35,
`session.py` 27, `sweep.py` 24, `ic_packs.py` 14, `primitives.py` 4. Root
cause is one repeated idiom — a `blob: object` / `dict[str, Any]` parameter
that must be re-narrowed on every access:

```38:41:D:/dipcatcher/src/quant_fund/research/catalog/primitives.py
        try:
            x = float(blob.get(key))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append(f"{key}_non_numeric")
```

The same `# type: ignore[arg-type]` appears at `primitives.py:48,58,80`,
`receipt.py:596`, and ~135 more times. Related: **383 functions return
`dict[str, Any]`** and there are 370 `: Any` annotations. Receipts, metrics
blobs, and hypothesis payloads — the objects the honesty contract is *about* —
are the least typed things in the codebase.

The 10 `method-assign` ignores mark runtime monkeypatching of a broker's
methods (`pretrade/shadow.py:92-94`, `:125-127`, `:132-134`,
`simtest/session.py:107`), including rebinding `SimulatedBroker.submit` at
class level inside a context manager. Functional and restored in `finally`, but
it is global mutation of a security-relevant class, invisible to the type
system and to any concurrent user of that class.

### P2-5 · God modules

21 files exceed 1,000 lines; 63 are 500–1,000.

| Lines | Module | Concerns mixed |
|---|---|---|
| 1,877 | `models/covariance.py` | PSD repair, shrinkage, factor, DCC family, likelihoods, dispatch registry (§P2-1) |
| 1,862 | `models/cs_papers.py` | cross-sectional paper estimators |
| 1,817 | `research/agent.py` | provenance, git worktree hashing, JSON coercion, FDR, data-snooping, 3 hypothesis builders, notebook assembly, `run_research` |
| 1,778 | `research/catalog/receipt.py` | receipt soft-verify |
| 1,553 | `backtest/event_sim/simulator.py` | event simulator |
| 1,520 | `research/catalog/candle.py` | candle/book honesty checks |
| 1,410 | `research/catalog/hypotheses.py` | H-table constants |
| 1,400 / 1,387 | `never_equate_gap.py` / `never_equate.py` | §P2-2 |
| 1,378 | `northset/sweep_research.py` | sweep research |
| 1,349 | `paper/forward_shadow.py` | forward shadow |
| 1,338 | `backtest/fast_replay.py` | vectorized replay + numba kernels |
| 1,331 | `northset/benches.py` | benches |
| 1,121 | `config/models.py` | 29 config models (cohesive, but see P3-1) |

`research/agent.py` is the worst offender per-function:
`_hypotheses_northset` spans **`:1000-1582` — 582 lines in a single function**.
`_hypotheses_conformal_bounds` is `:774-999` (225 lines),
`_hypotheses_rankers_and_rewards` is `:585-773` (188 lines). Together those
three functions are ~1,000 of the file's 1,817 lines.

`research/` holds 29 top-level modules of which **7 are benchmark variants**:
`benches_extra.py` (112), `benches_w810.py` (277), `garch_benchmark.py` (216),
`real_benchmark.py` (438), `research100_benchmark.py` (138), `vol_bench.py`
(665), plus `benches/` (2,658 across 5 files). Repo-wide there are **11
`*bench*` modules** (`mc_engine/benchmark.py`, `microstructure/bench.py`,
`models/robinhood_plus/bench.py`, `northset/benches.py`, `pretrade/bench.py`,
and the 6 above) with no shared harness base.

### P3-1 · Config: strong model discipline, weak environment story

`config/models.py` is the best-engineered file reviewed: `StrictConfigModel`
with `ConfigDict(extra="forbid")` (`:13-17`) so unknown keys are rejected at
every nesting level; 29 nested models; **every one** carrying a
`model_validator(mode="after")` with explicit finiteness and range checks
(`np.isfinite` guards on all floats, `(0,1]` bounds on participation rates,
uniqueness + monotonicity on quantile levels `:169-176`, SHA-256 hex format
validation on Kronos digests `:531-538`, and path-traversal rejection on
`ledger_subdir` `:895-902`). `RuntimeConfig.live_must_be_explicit` (`:86-96`)
fails closed twice — once on missing `allow_live`, once with
*"allow_live is unsupported: no live broker adapter in this repository"*,
directly enforcing honesty rule #3. `PortfolioConstraints.effective_gross_cap`
(`:296-299`) and `gross_redundant` (`:302-304`) document a real subtlety.
`config/loader.py` handles `inherit:` with cycle detection (`:46-47`) and
config-root escape prevention (`:41-44`).

Gaps:

- **`pydantic-settings>=2.6` is a declared core dependency with zero uses.**
  No `BaseSettings` subclass and no `pydantic_settings` import anywhere in
  `src/`. Environment configuration is instead **44 raw `os.environ`/`getenv`
  reads across 20 files**, with ad-hoc parsing per site:
  `api/research_api.py:81-96` reads five `RESEARCH_API_*` vars inline with
  `or`-chain fallbacks; `observe/flags.py:11` defines its own `_TRUE`
  frozenset bool parser; `api/app.py:137` reads `QUANT_API_KEY` while
  `cli/ops_cmds.py:23` reads the same var through a different loopback-host
  check; `api/research_api.py:969` re-implements that check. Secrets
  (`FRED_API_KEY` `data/sources/adapters.py:503,509`, `BEA_API_KEY` `:714`,
  `MOONSHOT_API_KEY`, `FX1_SIGNING_KEY` `pipeline/doctor.py:136-137`,
  `DIPCATCHER_SIGSTORE_ID_TOKEN` `audit/signing.py:158`) have no single
  declaration point.
- **Two config formats, two loaders.** `config/loader.py` is YAML-only with
  `inherit:`, but `configs/` holds 7 `.json` files (`net_tournament.json`,
  `cost_aware_tournament.json`, `forward_shadow.example.json`,
  `fx1_run.example.json`, `real_benchmark_us_wide.json`) read by separate
  ad-hoc paths — and `sota_protocol.yaml` has its own pydantic model at
  `research/sota_protocol.py:55` outside the `AppConfig` tree.
- **Config validation reaches into numeric internals.**
  `OptimizerConfig.implemented_optimizer_covariance` (`config/models.py:332-339`)
  performs a deferred import of `models.covariance.require_implemented_optimizer_covariance`
  inside a `field_validator`, so loading a config file imports the covariance
  module. It is also the reason `config/models.py` cannot be validated without
  the numeric stack.
- Mixed mutation of config for scenario runs: `backtest/engine.py:721-723`
  does `scenario = config.model_copy(deep=True)` then mutates
  `scenario.costs.impact_y` in place, relying on the deep copy. Correct, but
  the models are not frozen, so the safety depends entirely on remembering
  `deep=True`.

### P3-2 · Dataclass / pydantic discipline

Generally good. 201 `@dataclass` decorators: **131 `frozen=True`**, 66 bare
(mutable), 8 `slots=True`. 57 pydantic `BaseModel` subclasses, 14
`ConfigDict(` sites. Zero mutable-default-argument bugs — `ruff --select B006`
passes clean, and all container defaults correctly use
`Field(default_factory=…)` / `field(default_factory=…)`.

The 66 mutable dataclasses are mostly legitimate accumulators
(`backtest/engine.py:45 BacktestResult`, `:54 Book`,
`research/agent.py:119 HypothesisResult`, `:133 ResearchNotebook`), but
`Book` (`engine.py:54-62`) carries the cash-and-shares state of a backtest in
a mutable dataclass with an untyped `dict[str, float]`, mutated in-place across
a 300-line loop (`engine.py:410-527`).

### P3-3 · Dead code census

**Zero `TODO`/`FIXME`/`HACK`/`XXX` in all 670 modules** (1 in the whole repo,
at `src/fx1/eval/masking.py`). Unusually clean — but see P1-4: the intent
that `TODO` normally carries has migrated into `# noqa:` comments asserting
rules that are not enabled, which is worse because it looks enforced.

Dead code exists in other forms: the untracked `src/src/` (151 files, stale
pre-split snapshot containing the *only* two C901 violations, at complexity 78
and 196) and `tests/tests/` (457 files); the tracked `configs/configs/` and
`scripts/scripts/`; and 21 unused `noqa` directives.

---

## 3. Severity summary

| ID | Finding | Evidence | Severity |
|---|---|---|---|
| P0-1 | Build/gate identity instability; dropped real deps; committed duplicate config/script trees | `pyproject.toml` (tree vs HEAD); `configs/configs/base.yaml` missing `robinhood_plus`; `git log` `621023e`, `811dffc` | **P0** |
| P0-2 | Honesty gates fail **open** on import error | `research/catalog/receipt.py:588-590`; `candle.py:435-437`, `:1055-1057`; `fleet_eval.py:698-701` | **P0** |
| P1-1 | `_atomic_write_*` ×10; 8 not directory-fsynced; 3 divergent semantics | table in §P1-1; `_fsync_directory` only at `paper/ledger.py:66` | **P1** |
| P1-2 | 6/670 modules log; two non-convergent stacks; 26 `print()` | `utils/logging.py`; `observe/install.py:86-100`; `hedge_lab/v2_slate.py:154` | **P1** |
| P1-3 | 4 exception taxonomies; MI adapters to bridge; duplicate `StaleValuationError`; 82% bare `ValueError` | `schemas/errors.py`; `proofcore/contracts.py:55-104`; `leakage/watchdog.py:29`; `engine.py:41` vs `loop.py:68` | **P1** |
| P1-4 | Broad-except policing is comment-only; `BLE001` not enabled; 21 unused `noqa` | `pyproject.toml:129`; 10 `# noqa: BLE001` sites; `ruff --select RUF100` → 21 | **P1** |
| P2-1 | `covariance.py` 1,877-line god module; `dcc_gaussian` inlines a weaker NLL missing the `isfinite` guard | `:1257-1262` vs `:1105-1108`; helpers `:773-810`; `agdcc:1591` uses them | **P2** |
| P2-2 | 58 copy-paste `never_equate` guards; 643 identical lines; existing `_finite_pair` unused | `never_equate.py` (31 defs) / `never_equate_gap.py` (27 defs); `primitives.py:68` | **P2** |
| P2-3 | 5 import cycles; 74 cross-module private imports (`_date_keys` ×12, `_label_horizon` ×7, `_finite` ×5) | `fast_replay.py:37-44`, `:1323`; `ranking.py:135`; SCC analysis | **P2** |
| P2-4 | 142/219 `type: ignore` in `research/catalog`; 383 `dict[str, Any]` returns | §P2-4 tables; `primitives.py:38-41` | **P2** |
| P2-5 | 21 files >1,000 lines; `_hypotheses_northset` = 582 lines; 11 `*bench*` modules, no shared harness | `research/agent.py:1000-1582` | **P2** |
| P3-1 | `pydantic-settings` declared, unused; 44 raw `os.environ` reads; 2 config formats; validator imports numeric internals | `config/models.py:332-339`; `api/research_api.py:81-96` | **P3** |
| P3-2 | 66 mutable dataclasses; `Book` state mutated in place over 300 lines | `backtest/engine.py:54-62`, `:410-527` | **P3** |
| P3-3 | 151-file stale `src/src/` invisible to ruff; 457-file `tests/tests/` | §1.1, §P3-3 | **P3** |

---

## 4. Refactor backlog, ordered by leverage ÷ effort

Ranked so the top items are high-value and cheap. Items 1–4 are all under a
day each and remove P0/P1 risk.

| # | Action | Fixes | Effort | Leverage |
|---|---|---|---|---|
| 1 | **Freeze and verify `pyproject.toml` against `HEAD`.** Diff, restore, confirm `numba`/`exchange-calendars` present, `fail_under=80`, `C901`+`max-complexity=74`, `testpaths` excluding `tests/tests`, then `uv lock` and `uv sync --frozen` to prove the lockfile matches. | P0-1 | ~1 h | **Very high** — every other gate result depends on it |
| 2 | **Make the 4 honesty gates fail closed.** Narrow to `except ImportError` and return a non-empty sentinel (`["<check>_unverifiable"]`) instead of `[]`; add a test asserting a broken import produces an error, not a pass. | P0-2 | ~2 h | **Very high** — restores the fail-closed invariant the whole receipt system rests on |
| 3 | **Delete the stale trees.** `src/src/`, `tests/tests/`, `configs/configs/`, `scripts/scripts/`, `.backup-prerestore/`. The last two are tracked, so this needs a commit. Add a `tests/unit/test_quality_ratchet.py` assertion that no `X/X/` self-nested directory exists under `src`, `tests`, `configs`, `scripts`. | P0-1, P3-3 | ~1 h | **High** — removes 608 phantom files and the C901 outliers; prevents recurrence (`621023e` shows it already recurred once) |
| 4 | **Fix `backtest/_fast_kernel.py:28`** — change `type: ignore[misc]` → `[no-redef]` and annotate the shim params, matching `models/volatility.py:22`. Then flip `disallow_untyped_defs = true` globally: only 18 defs stand in the way and 8 are in this one file. | §1.2 | ~3 h | **High** — turns the type gate green and locks in the ratchet |
| 5 | **Extract `utils/atomic.py`** with one `atomic_write_text` / `atomic_write_bytes` / `atomic_write_parquet`, always directory-fsyncing, and an explicit `mode: Literal["overwrite","create_new"]` parameter. Migrate all 10 call sites; `seal_receipt` gets `create_new`. | P1-1 | ~4 h | **High** — one function, 10 sites, fixes durability on 8 of them and removes a semantic trap |
| 6 | **Enable `BLE001` + `RUF100` in ruff `select`,** then triage the 77 `except Exception` sites: keep the 16 documented boundaries with real suppressions, narrow the rest, and delete the 21 vestigial `noqa`s. | P1-4 | ~6 h | **High** — converts comment-only discipline into an enforced rule, and makes the suppression inventory self-maintaining |
| 7 | **Unify the exception taxonomy.** Keep `QuantFundError` as the single root; re-parent `ProofcoreError` and `AuditError` under it (or re-export aliases) so `leakage/watchdog.py:29` and `pit/frame.py:27` can drop their MI adapters; collapse the duplicate `StaleValuationError` into `schemas/errors.py`. Then convert the ~200 config/data-contract `ValueError`s to `ConfigError`/`DataContractError`. | P1-3 | ~1–2 d | **High** — removes two MI shims and makes `except QuantFundError` meaningful at boundaries |
| 8 | **Wire one observability stack.** Adopt `observe/`'s correlation-ID + redaction design *inside* `utils/logging.py` (structlog processors), delete `observe/logging.py`, and add `get_logger(module=…)` to the ~15 highest-value modules (`backtest/engine`, `research/agent`, `research/verify`, `paper/loop`, `paper/ledger`, `portfolio/optimizer`, `data/ingest`, `data/lake`). Replace the 13 library `print()`s with bound-logger calls. | P1-2 | ~2 d | **High** — 6→~20 logged modules, one stack, and P0-2-style silent failures become visible |
| 9 | **De-duplicate `dcc_gaussian`.** Replace its inlined window/`qbar`/step/`r_from_q`/one-step/NLL blocks (`:1223-1288`) with calls to `_dcc_prepare_window`, `_dcc_qbar`, `_dcc_step_q`, `_dcc_r_from_q`, `_dcc_one_step_h`, `_gaussian_corr_nll`. Add a regression test asserting a non-finite quadratic returns the `1e12` penalty, matching `dcc_student_t`/`agdcc`. | P2-1 (correctness) | ~4 h | **High** — removes an estimator divergence the config layer explicitly promises won't happen |
| 10 | **Table-drive the `never_equate` guards.** Define `NEVER_EQUATE_PAIRS: tuple[NeverEquateSpec, …]` and one generic checker; keep the 58 names as thin generated wrappers so the mypy strict allowlist and existing tests keep working. Use `_finite_pair`. | P2-2 | ~1 d | **High** — ~2,700 → ~200 lines, and pair coverage becomes auditable in one screen |
| 11 | **Promote the 8 shared private symbols to public modules:** `_date_keys` → `metrics/cross_section.py:date_keys` (12 importers), `_label_horizon` → `pipeline/train` public (7), `_finite` → `utils/numeric.py` (5 — this also breaks the `ranking`↔`asset_pricing` cycle), `_build_result` → `backtest/results.py` (breaks `engine`↔`fast_replay`), `_receipt_digest`, `_align_ic`/`_ic_card`, `_load_bars`/`_read_receipt`/`_seal`, `_predict_ranker`. Keep underscore aliases for one release. | P2-3 | ~1 d | **Medium-high** — breaks 2 of 5 cycles and stops private names acting as public API |
| 12 | **Split `models/covariance.py`** into `covariance/{repair,shrinkage,factor,dcc,likelihood,registry}.py` with a re-exporting `__init__`. Add the new module names to the mypy strict allowlist only after they are clean. | P2-1, P2-5 | ~1 d | **Medium** — 1,877 → ~300 lines/module; mechanical, low risk |
| 13 | **Split `research/agent.py`** — extract `_hypotheses_northset` (582 lines), `_hypotheses_conformal_bounds` (225), `_hypotheses_rankers_and_rewards` (188) into `research/hypotheses/{northset,conformal,rankers}.py`; move `_provenance`/`_git_revision`/`_git_worktree_sha256`/`_jsonable` to `utils/provenance.py`. | P2-5 | ~1 d | **Medium** |
| 14 | **Introduce `utils/env.py`** using the already-declared `pydantic-settings`: one `HarnessSettings(BaseSettings)` declaring all ~20 env vars with types, defaults, and secret handling. Migrate the 44 raw `os.environ` reads. Either use the dependency or drop it from `[project.dependencies]`. | P3-1 | ~1 d | **Medium** — single declaration point for secrets; removes an unused core dep |
| 15 | **Add a `Benchmark` protocol** and fold the 11 `*bench*` modules onto it; merge `research/benches_extra.py` (112 lines) and `benches_w810.py` (277) into `research/benches/`. | P2-5 | ~2 d | **Medium** |
| 16 | **Extend typing into `research/catalog/`.** Give receipt blobs a `TypedDict`/pydantic shape so the 142 `type: ignore[arg-type]` comments and 383 `dict[str, Any]` returns retire naturally. Start with `primitives.py` (the `_finite_scalar`/`_ic_pack_honesty_errors`/`_finite_pair` trio every catalog module funnels through). | P2-4 | ~3 d | **Medium** — highest `Any` concentration in the repo |
| 17 | **Lower `mccabe.max-complexity`** from 74 in steps (74 → 40 → 25), splitting `validate_ledger_schema` first. 106 functions violate a ceiling of 15; each ratchet step should land with the splits that make it pass. | §1.3 | ongoing | **Medium** — the only complexity control in the repo is currently set to grandfather one function |
| 18 | **Freeze or encapsulate `Book`** (`backtest/engine.py:54`) and make `BacktestResult` frozen; prefer `model_copy(deep=True)`-then-validate over in-place mutation of config in scenario loops (`engine.py:721-723`). | P3-2 | ~4 h | **Low-medium** |

**Sequencing note:** items 1–3 are prerequisites for trusting anything else —
do them first and re-run `make lint`, `make typecheck`, `make test`. Items
4–6 are the cheap gate-greening block. Items 9–11 carry real correctness and
coupling payoff per hour. Items 12–18 are structural and can be interleaved
with feature work; 16 and 17 are ongoing ratchets, not one-shot fixes.

---

## 5. What is already good (do not regress)

Recorded so the backlog does not accidentally unwind these:

- **0 bare `except:`** in 670 modules; all 4 `except BaseException` uses are
  correct cleanup-and-reraise or deliberate cancellation escape.
- **All 219 `type: ignore` comments are code-scoped.** Zero bare ignores.
- **0 TODO/FIXME/HACK/XXX** in the harness.
- **`config/models.py` validation depth** is exceptional: `extra="forbid"` at
  every nesting level, finiteness checks on every float, and a `live` mode
  that fails closed because no broker adapter ships.
- **The mypy ratchet design is sound**: an 81-module strict-tier override list
  (`pyproject.toml:186-270` at `HEAD`) spelling out
  the full `--strict` flag set, `warn_return_any = true` globally, a documented
  "promote only after clean, never loosen" rule, and a CI job running
  `mypy --strict --follow-imports=silent` on the public facade.
- **`quant_fund/__init__.py` lazy `__getattr__`** keeps `import quant_fund`
  light while preserving a typed `TYPE_CHECKING` surface — the right way to
  manage the root↔`public` cycle.
- **Honesty-contract architecture**: `FORBIDDEN_RESEARCH_METRIC_KEYS` mirrored
  to `fx1.honesty.FORBIDDEN_HEADLINE_TOKENS` with a drift test;
  `research_only`/`live_pnl_claim=False` stamped on every backtest metrics
  blob (`engine.py:639-640`, `:819-820`) including the frictionless and
  empty-frame branches.
- **Coverage floor 80** with a single source of truth (no inline
  `--cov-fail-under` in CI or Makefile, so it cannot drift).
- `ruff check src` and `ruff --select B006` both pass clean.
