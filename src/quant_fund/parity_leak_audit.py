"""Adversarial contract audit of ``quant_fund.parity`` + ``quant_fund.leakage``.

Two coupled safety layers get one probe suite:

* ``parity/`` — bar-by-bar backtest-vs-shadow attribution. ``attribute_pair``
  has a fixed cause precedence (data > timing > code_path > state_drift >
  rounding > costs > fills) and must never raise on a mismatch — the ledger
  is the verdict.
* ``leakage/`` — the syntactic LH001..LH014 linter, the runtime
  ``LeakageWatchdog`` (fail-closed on unverifiable reads), and the
  ``IOGuard`` monkey-interposition layer active inside proven decision
  windows.

Each probe is True iff the pinned contract holds. ``flag_*`` probes mark a
wart — behavior pinned for review, not necessarily a violation.

The bench function seals the probe results into a receipt-shaped claim for
``verify-receipt``.
"""

from __future__ import annotations

import math
import tempfile
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# leakage/proofcore are PROOFCORE packages — only cli/** may import them at
# module top level (test_proofcore_layering), so every leakage/proofcore
# dependency is imported lazily inside the functions that use it.
from quant_fund.parity.checker import CAUSE_PRECEDENCE, attribute_pair, check_parity
from quant_fund.parity.session import Bar, MarketSession, end_marks, session_fingerprint

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

_T = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)


def _bar(sid: str = "SYNTH", hour: int = 14, close: float = 100.0) -> Bar:
    return Bar(
        security_id=sid,
        event_time=datetime(2024, 1, 2, hour, 0, tzinfo=UTC),
        open=close - 0.1,
        high=close + 0.2,
        low=close - 0.3,
        close=close,
        volume=1000.0,
        source="synthetic",
        revision_id="0",
        available_time=datetime(2024, 1, 2, hour, 0, tzinfo=UTC),
    )


def _row(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "event_time": _T,
        "security_id": "SYNTH",
        "decision_time": _T,
        "source": "synthetic",
        "revision_id": "0",
        "available_time": _T,
        "open": 1.0,
        "high": 1.1,
        "low": 0.9,
        "close": 1.0,
        "volume": 1000.0,
        "exec_open": 1.0,
        "exec_close": 1.0,
        "exec_source": "synthetic",
        "exec_revision": "0",
        "code_path_digest": "d" * 8,
        "state_digest": "s" * 8,
        "restarted": False,
        "restart_generation": 0,
        "lot_size": 1.0,
        "raw_weight": 0.5,
        "rounded_weight": 0.5,
        "commission_bps": 1.0,
        "half_spread_bps": 2.0,
        "impact_y": 0.1,
        "bps_per_turnover": 0.5,
        "fill_signed_qty": 10.0,
        "fill_price": 1.0,
        "fill_fee": 0.01,
        "fill_spread": 0.02,
        "fill_impact": 0.03,
        "explicit_cost": 0.06,
    }
    base.update(over)
    return base


class _Run:
    def __init__(self, rows: list[dict[str, Any]], synthetic: bool = True) -> None:
        self.rows = rows
        self.synthetic = synthetic


def _scan(tmp: Path, name: str, source: str, rules: set[str] | None = None) -> list[Any]:
    from quant_fund.leakage.ast_scan import scan_file

    target = tmp / name
    target.write_text(source, encoding="utf-8")
    return scan_file(target, rules=rules)


def _rules_of(findings: list[Any]) -> set[str]:
    return {f.rule_id for f in findings}


def _record(params: dict[str, str] | None = None) -> Any:
    from quant_fund.proofcore.contracts import DataAccessRecord

    return DataAccessRecord(
        dataset="silver/bars",
        asof_utc=_T.isoformat(),
        params=params or {},
        rows=1,
        content_sha256="a" * 64,
    )


def parity_leak_audit() -> dict[str, bool]:
    """Run every probe; each key is True iff the pinned contract holds."""
    from quant_fund.leakage.ast_scan import _check_lh014, collect_py_files, scan_file
    from quant_fund.leakage.guard import GUARD_MODES, IOGuard, install_io_guard
    from quant_fund.leakage.watchdog import (
        MAX_KNOWN_AT_PARAM,
        LeakageError,
        LeakageWatchdog,
    )
    from quant_fund.proofcore.run_context import decision_window

    results: dict[str, bool] = {}
    dt = _T

    # -- parity/session.py -----------------------------------------------------
    try:
        Bar(
            security_id=" ",
            event_time=dt,
            open=1.0,
            high=1.1,
            low=0.9,
            close=1.0,
            volume=1.0,
            source="synthetic",
            revision_id="0",
            available_time=dt,
        )
        results["bar_blank_sid_rejected"] = False
    except ValueError:
        results["bar_blank_sid_rejected"] = True

    try:
        Bar(
            security_id="X",
            event_time=datetime(2024, 1, 2, 14, 0),  # noqa: DTZ001 — probing naive-dt rejection
            open=1.0,
            high=1.1,
            low=0.9,
            close=1.0,
            volume=1.0,
            source="synthetic",
            revision_id="0",
            available_time=dt,
        )
        results["bar_naive_dt_rejected"] = False
    except ValueError:
        results["bar_naive_dt_rejected"] = True

    try:
        Bar(
            security_id="X",
            event_time="2024-01-02",  # type: ignore[arg-type]
            open=1.0,
            high=1.1,
            low=0.9,
            close=1.0,
            volume=1.0,
            source="synthetic",
            revision_id="0",
            available_time=dt,
        )
        results["bar_nondatetime_rejected"] = False
    except TypeError:
        results["bar_nondatetime_rejected"] = True

    for field, bad in (("open", 0.0), ("close", -1.0), ("high", math.inf)):
        kwargs: dict[str, Any] = {"open": 1.0, "high": 1.1, "low": 0.9, "close": 1.0, "volume": 1.0}
        kwargs[field] = bad
        try:
            Bar(
                security_id="X",
                event_time=dt,
                source="synthetic",
                revision_id="0",
                available_time=dt,
                **kwargs,
            )
            results[f"bar_bad_{field}_rejected"] = False
        except ValueError:
            results[f"bar_bad_{field}_rejected"] = True

    try:
        Bar(
            security_id="X",
            event_time=dt,
            open=1.0,
            high=1.1,
            low=0.9,
            close=1.0,
            volume=-5.0,
            source="synthetic",
            revision_id="0",
            available_time=dt,
        )
        results["bar_negative_volume_rejected"] = False
    except ValueError:
        results["bar_negative_volume_rejected"] = True

    try:
        MarketSession([])
        results["session_empty_rejected"] = False
    except ValueError:
        results["session_empty_rejected"] = True

    import polars as pl

    try:
        MarketSession.from_frame(pl.DataFrame({"security_id": ["X"], "event_time": [dt]}))
        results["frame_missing_cols_rejected"] = False
    except ValueError:
        results["frame_missing_cols_rejected"] = True

    # iter_bar_groups fails closed on non-monotone event_time — keep sorted.
    sess = MarketSession([_bar("A", 14, 100.0), _bar("B", 14, 50.0), _bar("A", 15, 101.0)])
    try:
        MarketSession([_bar("A", 15, 101.0), _bar("A", 14, 100.0)])
        results["session_unsorted_rejected"] = False
    except ValueError:
        results["session_unsorted_rejected"] = True
    frame = sess.to_frame()
    rt = MarketSession.from_frame(frame)
    results["session_frame_roundtrip"] = len(rt.bars) == len(sess.bars)
    results["end_marks_last_close"] = end_marks(sess.bars) == {"A": 101.0, "B": 50.0}
    fp1 = session_fingerprint(sess.bars)
    results["fingerprint_deterministic"] = fp1 == session_fingerprint(sess.bars)
    results["fingerprint_tamper_sensitive"] = fp1 != session_fingerprint(
        [_bar("A", 14, 100.0), _bar("A", 15, 101.01), _bar("B", 14, 50.0)]
    )

    # -- parity/checker.py -----------------------------------------------------
    clean = check_parity(_Run([_row()]), _Run([_row()]))
    results["checker_clean_match"] = bool(clean["match"]) and clean["n_matched"] == 1

    results["both_none_raises"] = False
    try:
        attribute_pair(None, None)
    except ValueError:
        results["both_none_raises"] = True

    for bad_tol in (-1e-6, float("nan")):
        try:
            attribute_pair(_row(), _row(), tol=bad_tol)
            results[f"tol_{bad_tol}_rejected"] = False
        except ValueError:
            results[f"tol_{bad_tol}_rejected"] = True

    dup = [_row(), _row()]
    try:
        check_parity(_Run(dup), _Run([_row()]))
        results["dup_keys_rejected"] = False
    except ValueError:
        results["dup_keys_rejected"] = True

    # precedence: a row differing on every channel must attribute "data"
    wild = _row(close=9.9, decision_time=datetime(2024, 1, 3, tzinfo=UTC), fill_price=9.9)
    cause, _ = attribute_pair(_row(), wild)
    results["precedence_data_first"] = cause == "data"

    cause, detail = attribute_pair(_row(), _row(fill_price=9.9))
    results["precedence_fills_last"] = (cause, detail) == ("fills", "fill_price")

    cause, detail = attribute_pair(
        _row(), _row(state_digest="z" * 8, restart_generation=1, restarted=True)
    )
    results["restart_state_drift"] = (cause, detail) == ("state_drift", "after_restart")

    cause, detail = attribute_pair(_row(), _row(state_digest="z" * 8))
    results["state_drift_no_restart"] = (cause, detail) == ("state_drift", "state_digest")

    cause, _ = attribute_pair(_row(), _row(lot_size=2.0, rounded_weight=1.0))
    results["rounding_lot"] = cause == "rounding"

    cause, detail = attribute_pair(_row(), _row(rounded_weight=0.6))
    results["rounding_weight_only"] = (cause, detail) == ("rounding", "rounded_weight")

    cause, detail = attribute_pair(_row(), _row(raw_weight=0.6, rounded_weight=0.6))
    results["weight_mismatch_code_path"] = (cause, detail) == ("code_path", "output_mismatch")

    cause, _ = attribute_pair(_row(), _row(code_path_digest="e" * 8))
    results["code_path_digest"] = cause == "code_path"

    cause, _ = attribute_pair(_row(), _row(decision_time=datetime(2024, 1, 3, tzinfo=UTC)))
    results["timing_decision_time"] = cause == "timing"

    # missing-row attribution: shifted decision_time -> timing, else data
    shifted = _row(decision_time=datetime(2024, 1, 3, tzinfo=UTC), event_time=_T)
    cause, detail = attribute_pair(
        None,
        shifted,
        other_backtest=[
            _row(
                decision_time=datetime(2024, 1, 3, tzinfo=UTC),
                event_time=datetime(2024, 1, 4, tzinfo=UTC),
            )
        ],
    )
    results["missing_shifted_bar"] = (cause, detail) == ("timing", "shifted_bar")
    cause, detail = attribute_pair(None, _row(), other_backtest=[])
    results["missing_bar_data"] = (cause, detail) == ("data", "missing_bar")

    mixed = check_parity(
        _Run([_row()]),
        _Run([_row(), _row(event_time=datetime(2024, 1, 3, 14, 30, tzinfo=UTC))]),
    )
    results["missing_counted"] = (
        mixed["n_missing_backtest"] == 1 and mixed["n_divergent"] == 1 and not mixed["match"]
    )
    results["by_cause_sums"] = sum(mixed["by_cause"].values()) == mixed["n_divergent"]
    results["honesty_fields"] = (
        mixed["data_source"] == "SYNTHETIC"
        and mixed["research_only"] is True
        and mixed["live_pnl_claim"] is False
        and mixed["would_promote_live"] is False
    )
    results["precedence_tuple"] = CAUSE_PRECEDENCE == (
        "data",
        "timing",
        "code_path",
        "state_drift",
        "rounding",
        "costs",
        "fills",
    )

    # -- leakage/ast_scan.py ----------------------------------------------------
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)

        f = _scan(tmp, "a.py", "px = close.shift(-1)\n")
        results["lh001_negative_shift_fires"] = "LH001" in _rules_of(f)
        f = _scan(tmp, "a.py", "px = close.shift(1)\n")
        results["lh001_positive_shift_clean"] = "LH001" not in _rules_of(f)
        f = _scan(tmp, "a.py", "k = -1\npx = close.shift(k)\n")
        results["lh001_bound_name_fires"] = "LH001" in _rules_of(f)
        f = _scan(tmp, "a.py", 'px = getattr(close, "shift")(-1)\n')
        results["lh001_getattr_fires"] = "LH001" in _rules_of(f)
        f = _scan(tmp, "a.py", 'x = pl.col("close").shift(-1)\n')
        results["lh001_polars_col_fires"] = "LH001" in _rules_of(f)
        f = _scan(tmp, "a.py", "px = signal.shift(-1)\n")
        results["lh001_nonprice_clean"] = "LH001" not in _rules_of(f)

        f = _scan(tmp, "a.py", "x = close.rolling_mean(21, center=True)\n")
        results["lh002_center_true_fires"] = "LH002" in _rules_of(f)
        f = _scan(tmp, "a.py", "x = close.rolling_mean(21, center=1)\n")
        results["lh002_center_truthy_fires"] = "LH002" in _rules_of(f)
        f = _scan(tmp, "a.py", "x = close.rolling_mean(21)\n")
        results["lh002_default_clean"] = "LH002" not in _rules_of(f)

        f = _scan(tmp, "a.py", "scaler.fit(X)\n")
        results["lh003_naked_fit_fires"] = "LH003" in _rules_of(f)
        f = _scan(tmp, "a.py", "for fold in folds:\n    scaler.fit(fold)\n")
        results["lh003_fold_loop_clean"] = "LH003" not in _rules_of(f)
        f = _scan(tmp, "a.py", "from scipy.stats import norm\nnorm.fit(xs)\n")
        results["lh003_scipy_stats_clean"] = "LH003" not in _rules_of(f)

        f = _scan(tmp, "a.py", 'df.join_asof(other, on="event_time")\n')
        results["lh004_event_time_fires"] = "LH004" in _rules_of(f)
        f = _scan(tmp, "a.py", 'df.join_asof(other, on="known_at")\n')
        results["lh004_known_at_clean"] = "LH004" not in _rules_of(f)
        f = _scan(tmp, "a.py", 'key = "event_time"\ndf.join_asof(other, on=key)\n')
        results["lh004_bound_key_fires"] = "LH004" in _rules_of(f)

        uni = (
            "universe = ["
            + ",".join(f'"{c}"' for c in ("AAPL", "MSFT", "GOOG", "AMZN", "NVDA"))
            + "]\n"
        )
        f = _scan(tmp, "a.py", uni)
        results["lh005_frozen_universe_fires"] = "LH005" in _rules_of(f)
        tests_dir = tmp / "tests"
        tests_dir.mkdir()
        f = _scan(tests_dir, "a.py", uni)
        results["lh005_tests_skipped"] = "LH005" not in _rules_of(f)
        small = 'universe = ["AAPL", "MSFT", "GOOG", "AMZN"]\n'
        f = _scan(tmp, "a.py", small)
        results["lh005_under_5_clean"] = "LH005" not in _rules_of(f)

        f = _scan(tmp, "a.py", "delta_mid = mid.shift(-1) - mid\n")
        results["lh006_contemp_name_fires"] = "LH006" in _rules_of(f)
        f = _scan(tmp, "a.py", "fwd_mid = mid.shift(-1) - mid\n")
        results["lh006_fwd_name_clean"] = "LH006" not in _rules_of(f)

        f = _scan(tmp, "a.py", "psr = probabilistic_sharpe(sharpe_ratio(r))\n")
        results["lh007_raw_ratio_fires"] = "LH007" in _rules_of(f)
        f = _scan(tmp, "a.py", "psr = probabilistic_sharpe(sharpe_ratio(r, periods_per_year=1))\n")
        results["lh007_per_period_clean"] = "LH007" not in _rules_of(f)

        f = _scan(tmp, "a.py", 'msg = "Sharpe: 2.1 for the run"\n')
        results["lh008_headline_fires"] = "LH008" in _rules_of(f)
        f = _scan(tmp, "a.py", 'row = {"sharpe": 2.1}\n')
        results["lh008_dict_key_clean"] = "LH008" not in _rules_of(f)
        f = _scan(tmp, "a.py", 'msg = "Sharpe:" + " 2.1 for the run"\n')
        results["lh008_concat_fires"] = "LH008" in _rules_of(f)

        f = _scan(tmp, "a.py", "df = read_parquet(p)\n")
        results["lh009_direct_read_fires"] = "LH009" in _rules_of(f)
        f = _scan(tmp, "a.py", 'import duckdb\nduckdb.sql("select * from read_parquet(p)")\n')
        results["lh009_sql_parquet_fires"] = "LH009" in _rules_of(f)
        f = _scan(tmp, "a.py", 'eval("pl.read_parquet(p)")\n')
        results["lh009_eval_fires"] = "LH009" in _rules_of(f)
        f = _scan(tmp, "a.py", 'fn = getattr(pl, "read_" + "parquet")\ndf = fn(p)\n')
        results["lh009_getattr_fires"] = "LH009" in _rules_of(f)
        f = _scan(tmp, "a.py", "df = scan_something(p)\n")
        results["lh009_benign_clean"] = "LH009" not in _rules_of(f)

        f = _scan(
            tmp,
            "a.py",
            'def f():\n    df = df.with_columns(event_time=1)\n    df.fill_null(strategy="backward")\n',
        )
        results["lh010_backward_fill_fires"] = "LH010" in _rules_of(f)
        f = _scan(tmp, "a.py", 'def f():\n    df.fill_null(strategy="forward")\n')
        results["lh010_forward_clean"] = "LH010" not in _rules_of(f)

        pc_dir = tmp / "quant_fund" / "proofcore"
        pc_dir.mkdir(parents=True)
        f = _scan(pc_dir, "a.py", "import quant_fund.backtest\n")
        results["lh011_layer_violation_fires"] = "LH011" in _rules_of(f)
        f = _scan(pc_dir, "a.py", "import quant_fund.proofcore.contracts\n")
        results["lh011_intra_package_clean"] = "LH011" not in _rules_of(f)
        f = _scan(tmp, "a.py", "import quant_fund.backtest\n")
        results["lh011_outside_package_clean"] = "LH011" not in _rules_of(f)

        f = _scan(tmp, "a.py", "def broken(:\n")
        results["lh012_unparseable_fires"] = "LH012" in _rules_of(f)

        f = _scan(tmp, "a.py", "# Sharpe 2.1 was hit\nx = 1\n")
        results["lh013_comment_fires"] = "LH013" in _rules_of(f)
        f = _scan(tmp, "a.py", 'def f():\n    """Sharpe 2.1 achieved."""\n    pass\n')
        results["lh013_docstring_fires"] = "LH013" in _rules_of(f)
        f = _scan(tmp, "a.py", 'sr = 2\nmsg = f"Sharpe {sr}"\n')
        results["lh013_fstring_fires"] = "LH013" in _rules_of(f)

        leaky_src = "def helper():\n    px = close.shift(-1)\n\ndef caller():\n    helper()\n"
        target = tmp / "helpers.py"
        target.write_text(leaky_src)
        f = scan_file(target)
        results["lh014_helper_call_fires"] = "LH014" in _rules_of(f)

        f1 = scan_file(tmp / "helpers.py")
        results["scan_deterministic"] = [(x.rule_id, x.line) for x in f] == [
            (x.rule_id, x.line) for x in f1
        ]

        try:
            collect_py_files([tmp / "nonexistent_dir"])
            results["collect_missing_raises"] = False
        except ValueError:
            results["collect_missing_raises"] = True

        # _check_lh014 called directly with empty findings -> []
        import ast as _ast

        empty = _check_lh014(_ast.parse("x = 1\n"), "a.py", [])
        results["lh014_empty_clean"] = empty == []

    # -- leakage/watchdog.py ----------------------------------------------------
    strict = LeakageWatchdog(strict=True)
    lax = LeakageWatchdog(strict=False)
    results["watchdog_strict_default"] = strict.strict is True and lax.strict is False

    naive_dt = datetime(2024, 1, 2, 14, 30)  # noqa: DTZ001 — probing naive-dt rejection
    try:
        strict.observe(_record({MAX_KNOWN_AT_PARAM: _T.isoformat()}), naive_dt)
        results["watchdog_naive_decision_rejected"] = False
    except LeakageError:
        results["watchdog_naive_decision_rejected"] = True

    try:
        strict.observe(_record(), dt)
        results["watchdog_missing_watermark_strict_fails"] = False
    except LeakageError:
        results["watchdog_missing_watermark_strict_fails"] = True

    lax.observe(_record(), dt)
    results["watchdog_missing_watermark_lax_ok"] = lax.n_observed == 0

    try:
        strict.observe(_record({MAX_KNOWN_AT_PARAM: "not-a-date"}), dt)
        results["watchdog_malformed_watermark_fails"] = False
    except LeakageError:
        results["watchdog_malformed_watermark_fails"] = True

    try:
        strict.observe(_record({MAX_KNOWN_AT_PARAM: "2024-01-02T14:30:00"}), dt)
        results["watchdog_naive_watermark_fails"] = False
    except LeakageError:
        results["watchdog_naive_watermark_fails"] = True

    future = datetime(2024, 1, 3, tzinfo=UTC)
    try:
        strict.observe(_record({MAX_KNOWN_AT_PARAM: future.isoformat()}), dt)
        results["watchdog_future_leak_fails"] = False
    except LeakageError:
        # the leak is counted before the raise — pinned for the audit trail
        results["watchdog_future_leak_fails"] = strict.n_observed == 1

    ok_dt = datetime(2024, 1, 3, tzinfo=UTC)
    strict.observe(_record({MAX_KNOWN_AT_PARAM: dt.isoformat()}), ok_dt)
    strict.observe(_record({MAX_KNOWN_AT_PARAM: ok_dt.isoformat()}), ok_dt)
    results["watchdog_watermark_max"] = strict.watermark() == ok_dt
    results["watchdog_n_observed"] = strict.n_observed == 3

    # -- leakage/guard.py --------------------------------------------------------
    results["guard_modes"] = GUARD_MODES == ("redirect", "enforce", "audit")
    try:
        IOGuard(mode="bogus")  # type: ignore[arg-type]
        results["guard_bad_mode_rejected"] = False
    except ValueError:
        results["guard_bad_mode_rejected"] = True

    with tempfile.TemporaryDirectory() as td:
        probe = Path(td) / "data.parquet"
        probe.write_text("x")

        # enforce mode + active window + unknown path -> LeakageError
        with install_io_guard("enforce"):
            try:
                with decision_window(dt):
                    open(probe).read()
                results["guard_enforce_blocks"] = False
            except LeakageError:
                results["guard_enforce_blocks"] = True

        # same read outside a window passes through with zero interception
        with install_io_guard("enforce"):
            results["guard_no_window_passthrough"] = open(probe).read() == "x"

        # audit mode records without raising
        with install_io_guard("audit") as guard, decision_window(dt):
            open(probe).read()
        results["guard_audit_records"] = (
            len(guard.records) >= 1 and guard.records[0].action == "audited"
        )

        # redirect mode warns and proceeds for unmapped paths
        with install_io_guard("redirect") as guard, warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            with decision_window(dt):
                open(probe).read()
        results["guard_redirect_warns"] = guard.records[0].action == "warned"

        # allowlisted path proceeds under enforce
        with install_io_guard("enforce", allowlist=(td,)) as guard, decision_window(dt):
            open(probe).read()
        results["guard_allowlist_passes"] = guard.records[0].action == "allowlisted"

        # vault-mapped path redirects through vault.asof
        class _Vault:
            def __init__(self) -> None:
                self.calls: list[tuple[str, Any]] = []

            def asof(self, name: str, when: Any) -> str:
                self.calls.append((name, when))
                return "redirected-frame"

        vault = _Vault()
        with install_io_guard("enforce", vault_map={td: vault}) as guard, decision_window(dt):
            # the interposed read returns vault.asof(name, decision_time) directly
            result = open(probe)
        results["guard_vault_redirect"] = (
            result == "redirected-frame" and vault.calls[0][0] == "data"
        )
        results["guard_records_redirect"] = guard.records[0].action == "redirected"

        # re-entrancy: nested install yields the same guard; teardown restores
        import builtins

        orig_open = builtins.open
        with install_io_guard("audit") as outer, install_io_guard("enforce") as inner:
            results["guard_nested_same"] = inner is outer
        results["guard_teardown_restores"] = builtins.open is orig_open

    return results


def parity_leak_audit_bench() -> dict[str, Any]:
    """Sealed receipt over the audit probes."""
    from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
    from quant_fund.utils.reproducibility import git_revision

    results = parity_leak_audit()
    payload: dict[str, Any] = {
        "kind": "parity_leak_audit",
        "schema": "parity_leak_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {
            "results": results,
            "ok": all(results.values()),
            "n_probes": len(results),
            "n_passed": sum(1 for v in results.values() if v),
        },
        "interpretation": (
            "Parity attribution pins the 7-cause precedence lattice (data > timing > "
            "code_path > state_drift > rounding > costs > fills) and fail-closed edge "
            "cases. The leakage pack pins each LH rule's fire/clean boundary plus the "
            "watchdog's fail-closed watermark contract and the IOGuard's "
            "window-scoped interception. flag_* probes mark documented warts."
        ),
    }
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
