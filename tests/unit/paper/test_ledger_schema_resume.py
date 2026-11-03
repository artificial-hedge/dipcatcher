"""Ledger reconciliation and the paper resume refusals that must fail closed."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import numpy as np
import polars as pl
import pytest

from quant_fund.config.loader import load_config
from quant_fund.paper.ledger import (
    PaperLedger,
    latest_run_id,
    load_broker_state,
    paper_root,
    promotion_dry_run,
    validate_ledger_schema,
    validate_promotion_dry_run_receipt,
)
from quant_fund.paper.loop import (
    _bar_maps,
    _merge_divergence_summary,
    _paper_portfolio_conformal,
    _parse_iso,
    _prior_divergence_samples,
    _valid_price,
    _validate_bar_panel,
    _validate_weight_panel,
    run_paper_loop,
)

_META = {
    "run_id": "run",
    "schema_version": 2,
    "label": "PAPER_SIMULATED",
    "disclaimer": "research only",
    "live_pnl_claim": False,
}


def _cfg(tmp_path):
    cfg = load_config("configs/paper.yaml").model_copy(deep=True)
    cfg.data.root = tmp_path
    return cfg


def _write(tmp_path, name: str, files: dict[str, object]):
    root = tmp_path / name
    root.mkdir()
    for filename, content in files.items():
        path = root / filename
        if isinstance(content, pl.DataFrame):
            content.write_parquet(path)
        elif isinstance(content, bytes):
            path.write_bytes(content)
        elif isinstance(content, str):
            path.write_text(content)
        else:
            path.write_text(json.dumps(content))
    return root


def _state(**overrides: object) -> dict[str, object]:
    state: dict[str, object] = {
        "run_id": "run",
        "schema_version": 2,
        "step": 0,
        "champion": {
            "slot": "champion",
            "cash": 1.0,
            "shares": {},
            "allow_capital": True,
            "n_orders": 0,
            "n_fills": 0,
        },
    }
    state.update(overrides)
    return state


def _equity(n: int = 1) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "event_time": ["2024-01-02T00:00:00+00:00"] * n,
            "nav": [1.0] * n,
            "cash": [1.0] * n,
        }
    )


def _errors(tmp_path, name: str, files: dict[str, object]) -> list[str]:
    report = validate_ledger_schema(_write(tmp_path, name, files))
    assert report["live_pnl_claim"] is False
    return report["errors"]


def _has(errors: list[str], prefix: str) -> None:
    assert any(item == prefix or item.startswith(prefix + ":") for item in errors), errors


def test_run_id_and_ledger_root_reject_path_escape(tmp_path) -> None:
    for run_id in ("", ".", "..", "a/b", "a\\b", "  "):
        with pytest.raises(ValueError, match="path-safe"):
            PaperLedger(tmp_path, run_id)
    for subdir in ("", "/", "\\windows", "..", "a/../b", "C:/paper", "/tmp/paper"):
        with pytest.raises(ValueError, match="safe relative"):
            paper_root(tmp_path, subdir)
    nested = paper_root(tmp_path, "sessions/one")
    assert nested.name == "one"


def test_broker_state_and_latest_run_fail_closed(tmp_path) -> None:
    assert load_broker_state(tmp_path, "missing") is None
    assert latest_run_id(tmp_path) is None
    root = paper_root(tmp_path) / "run-a"
    root.mkdir()
    (root / "broker_state.json").write_text("{")
    assert load_broker_state(tmp_path, "run-a") is None
    (root / "broker_state.json").write_text("[]")
    assert load_broker_state(tmp_path, "run-a") is None
    (root / "broker_state.json").write_text('{"run_id": "run-a"}')
    assert load_broker_state(tmp_path, "run-a") == {"run_id": "run-a"}

    latest = paper_root(tmp_path) / "latest_run.json"
    latest.write_text("{")
    assert latest_run_id(tmp_path) is None
    latest.write_text("[]")
    assert latest_run_id(tmp_path) is None
    latest.write_text('{"run_id": ""}')
    assert latest_run_id(tmp_path) is None
    latest.write_text('{"run_id": ".."}')
    with pytest.raises(ValueError, match="path-safe"):
        latest_run_id(tmp_path)
    latest.write_text('{"run_id": "run-a"}')
    assert latest_run_id(tmp_path) == "run-a"


def test_promotion_dry_run_rejects_bad_inputs_and_seals_a_diagnostic() -> None:
    kwargs = {
        "mean_l1": 0.1,
        "max_l1": 0.2,
        "n_steps": 3,
        "champion_nav": 10.0,
        "shadow_gross": 1.0,
        "max_mean_l1": 0.5,
        "min_steps": 2,
        "data_source": "FILE",
    }
    with pytest.raises(ValueError, match="finite and non-negative"):
        promotion_dry_run(**{**kwargs, "mean_l1": float("nan")})
    with pytest.raises(ValueError, match="step counts"):
        promotion_dry_run(**{**kwargs, "n_steps": -1})
    with pytest.raises(ValueError, match="n_divergence_samples"):
        promotion_dry_run(**{**kwargs, "n_divergence_samples": True})
    with pytest.raises(ValueError, match="non-negative"):
        promotion_dry_run(**{**kwargs, "champion_nav": -1.0})
    with pytest.raises(ValueError, match="rolling_window"):
        promotion_dry_run(**{**kwargs, "rolling_window": 0})
    with pytest.raises(ValueError, match="path-safe"):
        promotion_dry_run(**{**kwargs, "run_id": ".."})

    receipt = promotion_dry_run(
        **{**kwargs, "data_source": "SYNTHETIC"},
        run_id="sealed",
        rolling_mean_l1=0.05,
        rolling_window=4,
        primary_challenger="shadow",
        challenger_metrics={"shadow": {"mean_l1": 0.1}},
        risk_accounting={"reject_count": 0},
        n_divergence_samples=3,
    )
    assert receipt["would_promote_live"] is False
    assert receipt["live_pnl_claim"] is False
    assert receipt["research_only"] is True
    assert receipt["data_source"] == "SYNTHETIC"
    assert validate_promotion_dry_run_receipt(receipt) == []


def test_promotion_receipt_validator_names_each_forgery() -> None:
    assert validate_promotion_dry_run_receipt([]) == ["promotion_receipt_not_object"]  # type: ignore[arg-type]
    missing = validate_promotion_dry_run_receipt({})
    assert "promotion_missing:would_promote_live" in missing
    forged = validate_promotion_dry_run_receipt(
        {
            "would_promote_paper": "yes",
            "would_promote_live": True,
            "live_pnl_claim": True,
            "research_only": False,
            "mean_l1": True,
            "max_l1": "high",
            "rolling_mean_l1": -1.0,
            "n_steps": True,
            "min_steps": -1,
            "rolling_window": 0,
            "n_divergence_samples": -2,
            "run_id": 3,
            "schema_version": 2,
            "receipt_sha256": "zz",
        }
    )
    assert "promotion_would_promote_paper_must_be_bool" in forged
    assert "promotion_would_promote_live_must_be_false" in forged
    assert "promotion_live_pnl_claim_must_be_false" in forged
    assert "promotion_research_only_must_be_true" in forged
    assert "promotion_mean_l1_must_be_finite_nonnegative" in forged
    assert "promotion_rolling_window_must_be_positive" in forged
    assert "promotion_run_id_invalid" in forged
    assert "promotion_receipt_sha256_missing" not in forged
    assert "promotion_receipt_sha256_invalid" in forged

    path = validate_promotion_dry_run_receipt(
        {
            "would_promote_paper": False,
            "would_promote_live": False,
            "live_pnl_claim": False,
            "research_only": True,
            "run_id": "..",
            "mean_l1": 0.4,
            "max_l1": 0.2,
            "rolling_mean_l1": 0.5,
            "max_mean_l1": 0.1,
            "receipt_sha256": "a" * 64,
        }
    )
    assert "promotion_run_id_invalid" in path
    assert "promotion_max_l1_below_mean_l1" in path
    assert "promotion_rolling_mean_l1_above_max_l1" in path
    assert "promotion_receipt_sha256_mismatch" in path

    nan_ok = promotion_dry_run(
        mean_l1=float("nan"),
        max_l1=float("nan"),
        n_steps=1,
        champion_nav=None,
        shadow_gross=None,
        max_mean_l1=0.2,
        min_steps=2,
        data_source="FILE",
        allow_missing_divergence=True,
    )
    assert validate_promotion_dry_run_receipt(nan_ok) == []


def test_schema_validator_reports_torn_books(tmp_path) -> None:
    _has(_errors(tmp_path, "no-meta", {}), "meta.json_missing")
    _has(_errors(tmp_path, "bad-meta", {"meta.json": "{"}), "meta.json_invalid_json")
    assert "meta.json_not_object" in _errors(tmp_path, "meta-list", {"meta.json": []})
    assert "meta_missing:run_id" in _errors(tmp_path, "meta-empty", {"meta.json": {}})
    assert "meta_run_id_mismatch" in _errors(
        tmp_path, "meta-id", {"meta.json": {**_META, "run_id": "other"}}
    )
    assert "meta_schema_version_not_int" in _errors(
        tmp_path, "meta-bool", {"meta.json": {**_META, "schema_version": True}}
    )
    assert "meta_live_pnl_claim_must_be_false_or_absent" in _errors(
        tmp_path, "meta-live", {"meta.json": {**_META, "live_pnl_claim": True}}
    )
    warned = validate_ledger_schema(
        _write(tmp_path, "meta-old", {"meta.json": {**_META, "schema_version": 1}}),
        expect_version=2,
    )
    assert any(item.startswith("schema_version_mismatch") for item in warned["warnings"])

    _has(
        _errors(tmp_path, "state-bad", {"meta.json": _META, "broker_state.json": "{"}),
        "broker_state_invalid_json",
    )
    assert "broker_state_not_object" in _errors(
        tmp_path, "state-list", {"meta.json": _META, "broker_state.json": []}
    )
    assert "broker_state_missing:champion" in _errors(
        tmp_path, "state-empty", {"meta.json": _META, "broker_state.json": {}}
    )
    assert "broker_state_step_invalid" in _errors(
        tmp_path, "state-step", {"meta.json": _META, "broker_state.json": _state(step=True)}
    )
    assert "broker_state_run_id_mismatch" in _errors(
        tmp_path,
        "state-id",
        {"meta.json": _META, "broker_state.json": _state(run_id="other")},
    )
    assert "broker_state_resume_fingerprint_invalid" in _errors(
        tmp_path,
        "state-fp",
        {"meta.json": _META, "broker_state.json": _state(resume_fingerprint="zz")},
    )
    assert "broker_state_champion_not_object" in _errors(
        tmp_path,
        "champ-list",
        {"meta.json": _META, "broker_state.json": _state(champion=[])},
    )
    assert "broker_state_champion_missing:cash" in _errors(
        tmp_path,
        "champ-missing",
        {"meta.json": _META, "broker_state.json": _state(champion={"slot": "champion"})},
    )
    assert "broker_state_champion_n_orders_invalid" in _errors(
        tmp_path,
        "champ-count",
        {
            "meta.json": _META,
            "broker_state.json": _state(
                champion={
                    "slot": "champion",
                    "cash": 1.0,
                    "shares": {},
                    "allow_capital": True,
                    "n_orders": True,
                    "n_fills": -1,
                }
            ),
        },
    )
    history_errors = _errors(
        tmp_path,
        "champ-history",
        {
            "meta.json": _META,
            "broker_state.json": _state(
                champion={
                    "slot": "champion",
                    "cash": 1.0,
                    "shares": {},
                    "allow_capital": True,
                    "n_orders": 3,
                    "n_fills": 0,
                    "history": ["nope", {"order": {}}],
                }
            ),
        },
    )
    assert "broker_state_champion_history_record_invalid:0" in history_errors
    assert "broker_state_champion_history_missing:1:fill" in history_errors
    assert "broker_state_champion_history_order_count_mismatch" in history_errors
    assert "broker_state_champion_history_not_list" in _errors(
        tmp_path,
        "champ-hist-type",
        {
            "meta.json": _META,
            "broker_state.json": _state(
                champion={
                    "slot": "champion",
                    "cash": 1.0,
                    "shares": {},
                    "allow_capital": True,
                    "history": {"order": 1},
                }
            ),
        },
    )
    filled = {
        "order": {},
        "reject_reason": None,
        "fill": {"qty": 1},
        "slot": "champion",
    }
    assert "broker_state_champion_history_fill_count_mismatch" in _errors(
        tmp_path,
        "champ-fills",
        {
            "meta.json": _META,
            "broker_state.json": _state(
                champion={
                    "slot": "champion",
                    "cash": 1.0,
                    "shares": {},
                    "allow_capital": True,
                    "n_orders": 1,
                    "n_fills": 0,
                    "history": [filled],
                }
            ),
        },
    )

    _has(
        _errors(tmp_path, "eq-bad", {"meta.json": _META, "equity.parquet": b"not-parquet"}),
        "equity_unreadable",
    )
    assert "equity_missing:nav" in _errors(
        tmp_path,
        "eq-nav",
        {"meta.json": _META, "equity.parquet": pl.DataFrame({"event_time": ["t"], "cash": [1.0]})},
    )
    assert "equity_missing:cash" in _errors(
        tmp_path,
        "eq-cash",
        {"meta.json": _META, "equity.parquet": pl.DataFrame({"asof": ["t"], "nav": [1.0]})},
    )
    assert "equity_missing:asof_or_event_time" in _errors(
        tmp_path,
        "eq-time",
        {"meta.json": _META, "equity.parquet": pl.DataFrame({"nav": [1.0], "cash": [1.0]})},
    )
    assert "shadow_equity_missing:gross" in _errors(
        tmp_path,
        "shadow-cols",
        {
            "meta.json": _META,
            "shadow_equity.parquet": pl.DataFrame(
                {"event_time": ["t"], "nav": [1.0], "net": [0.0], "cash": [1.0], "slot": ["s"]}
            ),
        },
    )
    _has(
        _errors(tmp_path, "shadow-bad", {"meta.json": _META, "shadow_equity.parquet": b"nope"}),
        "shadow_equity_unreadable",
    )
    _has(
        _errors(tmp_path, "pos-bad", {"meta.json": _META, "positions.parquet": b"nope"}),
        "positions_unreadable",
    )
    assert "orders_missing:status" in _errors(
        tmp_path,
        "orders-cols",
        {
            "meta.json": _META,
            "orders.parquet": pl.DataFrame(
                {"order_id": ["o"], "security_id": ["A"], "side": ["buy"]}
            ),
        },
    )
    _has(
        _errors(tmp_path, "orders-bad", {"meta.json": _META, "orders.parquet": b"nope"}),
        "orders_unreadable",
    )
    assert "cash_ledger_missing:fees" in _errors(
        tmp_path,
        "cash-cols",
        {
            "meta.json": _META,
            "cash_ledger.parquet": pl.DataFrame({"asof": ["t"], "cash_delta": [1.0]}),
        },
    )
    assert "cash_ledger_cash_delta_nonfinite" in _errors(
        tmp_path,
        "cash-nan",
        {
            "meta.json": _META,
            "cash_ledger.parquet": pl.DataFrame(
                {
                    "asof": ["t"],
                    "slot": ["champion"],
                    "order_id": ["o"],
                    "security_id": ["A"],
                    "cash_delta": [float("nan")],
                    "notional": [1.0],
                    "fees": [0.0],
                    "side": ["buy"],
                }
            ),
        },
    )
    _has(
        _errors(tmp_path, "cash-bad", {"meta.json": _META, "cash_ledger.parquet": b"nope"}),
        "cash_ledger_unreadable",
    )
    _has(
        _errors(tmp_path, "promo-bad", {"meta.json": _META, "promotion_dry_run.json": "{"}),
        "promotion_dry_run_invalid_json",
    )
    assert "promotion_run_id_mismatch" in _errors(
        tmp_path,
        "promo-id",
        {
            "meta.json": _META,
            "promotion_dry_run.json": {
                "run_id": "other",
                "would_promote_paper": False,
                "would_promote_live": False,
                "live_pnl_claim": False,
                "research_only": True,
            },
        },
    )
    _has(
        _errors(tmp_path, "an-bad", {"meta.json": _META, "analytics_export.json": "{"}),
        "analytics_export_invalid_json",
    )
    assert "analytics_export_run_id_mismatch" in _errors(
        tmp_path,
        "an-id",
        {"meta.json": _META, "analytics_export.json": {"run_id": "other"}},
    )
    digest_errors = _errors(
        tmp_path,
        "an-digest",
        {
            "meta.json": _META,
            "analytics_export.json": [
                {"analytics_export_sha256": "zz"},
                {"run_id": "run", "analytics_export_sha256": "a" * 64},
            ][1],
        },
    )
    assert "analytics_export_sha256_mismatch" in digest_errors
    assert "analytics_export_sha256_invalid" in _errors(
        tmp_path,
        "an-short",
        {"meta.json": _META, "analytics_export.json": {"analytics_export_sha256": "zz"}},
    )

    assert "broker_state_step_requires_equity" in _errors(
        tmp_path, "step-no-eq", {"meta.json": _META, "broker_state.json": _state(step=2)}
    )
    assert "broker_state_step_equity_empty" in _errors(
        tmp_path,
        "step-empty",
        {
            "meta.json": _META,
            "broker_state.json": _state(step=2),
            "equity.parquet": pl.DataFrame({"event_time": [], "nav": [], "cash": []}),
        },
    )
    assert any(
        item.startswith("broker_state_step_equity_count_mismatch")
        for item in _errors(
            tmp_path,
            "step-count",
            {"meta.json": _META, "broker_state.json": _state(step=2), "equity.parquet": _equity(1)},
        )
    )
    assert "broker_state_step_ahead_of_equity" in _errors(
        tmp_path,
        "step-ahead",
        {"meta.json": _META, "broker_state.json": _state(step=2), "equity.parquet": _equity(1)},
    )
    assert "broker_state_step_behind_of_equity" in _errors(
        tmp_path,
        "step-behind",
        {"meta.json": _META, "broker_state.json": _state(step=0), "equity.parquet": _equity(1)},
    )
    assert "broker_state_missing_for_equity" in _errors(
        tmp_path, "eq-no-state", {"meta.json": _META, "equity.parquet": _equity(1)}
    )

    clean = validate_ledger_schema(
        _write(
            tmp_path,
            "clean",
            {"meta.json": _META, "broker_state.json": _state(), "equity.parquet": _equity(0)},
        )
    )
    assert clean["ok"] is False or "broker_state.json_missing" not in clean["warnings"]
    assert clean["cursor"]["consistent"] is True


def test_reload_reads_a_published_frame_and_refuses_a_torn_one(tmp_path) -> None:
    ledger = PaperLedger(tmp_path, "reload")
    assert ledger.load_equity().height == 0
    assert ledger.load_shadow_equity().height == 0
    ledger.record_shadow_equity({"event_time": "2024-01-02", "nav": 1.0, "cash": 1.0})
    ledger.flush()
    again = PaperLedger(tmp_path, "reload", load_existing=True)
    assert again.load_shadow_equity().height == 1
    (again.root / "orders.parquet").write_bytes(b"torn")
    with pytest.raises(ValueError, match="ledger file is unreadable: orders"):
        PaperLedger(tmp_path, "reload", load_existing=True)


def test_divergence_helpers_and_panels_fail_closed() -> None:
    assert _prior_divergence_samples(None) == 0
    assert _prior_divergence_samples({"n_divergence_samples": True, "n_steps": -1}) == 0
    assert _prior_divergence_samples({"n_steps": 4}) == 4
    assert _valid_price(None) is None
    assert _valid_price("nope") is None
    assert _valid_price(-1) is None
    assert _valid_price("10.5") == 10.5
    assert _parse_iso(None) is None
    assert _parse_iso("not-a-date") is None
    parsed = _parse_iso("2024-01-02T00:00:00+00:00")
    assert parsed is not None and parsed.tzinfo is not None

    mean, maximum = _merge_divergence_summary(None, [float("nan")], total_steps=1)
    assert np.isnan(mean) and np.isnan(maximum)
    kept = _merge_divergence_summary({"mean_l1": 0.2, "max_l1": 0.4}, [], total_steps=3)
    assert kept == (0.2, 0.4)
    merged, merged_max = _merge_divergence_summary(
        {"n_divergence_samples": 2, "mean_l1": 0.2, "max_l1": 0.5},
        [0.1, 0.3],
        total_steps=4,
    )
    assert merged == pytest.approx(0.2)
    assert merged_max == pytest.approx(0.5)
    broken = _merge_divergence_summary({"n_steps": "nope", "mean_l1": "x"}, [0.1], total_steps=2)
    assert broken[0] == pytest.approx(0.1)

    day = pl.DataFrame(
        {
            "security_id": ["A", "B"],
            "open": [None, 11.0],
            "close": [10.0, None],
            "close_total_return": [None, 12.0],
            "adv": [None, 5.0],
            "vol_20": [None, 0.2],
        }
    )
    marks, closes, advs, vols = _bar_maps(day, use_open=False)
    assert marks["A"] == 10.0 and "B" not in marks
    assert closes["B"] == 12.0 and closes["A"] == 10.0
    assert advs["A"] == 1.0 and vols["A"] == 0.02
    open_marks, _, _, _ = _bar_maps(day, use_open=True)
    assert open_marks == {"B": 11.0}

    t0 = datetime(2024, 1, 2, tzinfo=UTC)
    dup = pl.DataFrame({"event_time": [t0, t0], "security_id": ["A", "A"]})
    with pytest.raises(ValueError, match="duplicate bars"):
        _validate_bar_panel(dup)
    _validate_bar_panel(pl.DataFrame({"close": [1.0]}))
    weights = pl.DataFrame({"event_time": [t0], "security_id": ["A"]})
    with pytest.raises(ValueError, match="missing required columns"):
        _validate_weight_panel(weights)
    duplicated = pl.DataFrame(
        {
            "event_time": [t0, t0],
            "security_id": ["A", "A"],
            "target_weight": [0.1, 0.2],
        }
    )
    with pytest.raises(ValueError, match="duplicate target weights"):
        _validate_weight_panel(duplicated)
    short = _paper_portfolio_conformal(np.zeros(5))
    assert short["status"] == "insufficient_sample"
    assert short["live_pnl_claim"] is False
    long = _paper_portfolio_conformal(np.linspace(-0.01, 0.01, 40))
    assert long["status"] == "ok"
    assert long["live_pnl_claim"] is False


def _bars() -> pl.DataFrame:
    t0 = datetime(2024, 1, 2, tzinfo=UTC)
    t1 = datetime(2024, 1, 3, tzinfo=UTC)
    return pl.DataFrame(
        {
            "security_id": ["A", "A"],
            "event_time": [t0, t1],
            "open": [10.0, 11.0],
            "close": [10.5, 11.5],
            "volume": [1_000.0, 1_000.0],
            "source": ["synthetic", "synthetic"],
        }
    )


def _weights() -> pl.DataFrame:
    t0 = datetime(2024, 1, 2, tzinfo=UTC)
    return pl.DataFrame({"event_time": [t0], "security_id": ["A"], "target_weight": [0.1]})


def _persist(tmp_path, run_id: str, state: dict[str, object]) -> None:
    root = tmp_path / "metadata" / "paper" / run_id
    root.mkdir(parents=True)
    (root / "broker_state.json").write_text(json.dumps(state))


def _champion() -> dict[str, object]:
    return {
        "slot": "champion",
        "cash": 1_000.0,
        "initial_cash": 1_000.0,
        "shares": {},
        "open_orders": [],
        "last_marks": {},
        "allow_capital": True,
    }


def test_resume_refuses_a_torn_checkpoint_before_another_order(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    bars = _bars()
    with pytest.raises(ValueError, match="requires resume_run_id"):
        run_paper_loop(bars, cfg, champion_weights=_weights(), resume=True)
    with pytest.raises(FileNotFoundError, match="no broker_state"):
        run_paper_loop(bars, cfg, champion_weights=_weights(), resume=True, resume_run_id="absent")

    _persist(tmp_path, "mismatch", {"run_id": "other", "step": 0, "champion": _champion()})
    with pytest.raises(ValueError, match="mismatched run_id"):
        run_paper_loop(
            bars, cfg, champion_weights=_weights(), resume=True, resume_run_id="mismatch"
        )

    for label, state, match in (
        ("bad-step", {"run_id": "bad-step", "step": True, "champion": _champion()}, "invalid step"),
        (
            "bad-cursor",
            {"run_id": "bad-cursor", "step": 0, "last_decision": 3, "champion": _champion()},
            "last_decision",
        ),
        (
            "bad-exec",
            {"run_id": "bad-exec", "step": 0, "last_exec": "nope", "champion": _champion()},
            "last_exec",
        ),
        (
            "bad-fp",
            {
                "run_id": "bad-fp",
                "step": 0,
                "resume_fingerprint": "zz",
                "champion": _champion(),
            },
            "resume_fingerprint",
        ),
        (
            "bad-ages",
            {"run_id": "bad-ages", "step": 0, "mark_ages": {"A": -1}, "champion": _champion()},
            "mark_ages",
        ),
    ):
        _persist(tmp_path, label, state)
        with pytest.raises(ValueError, match=match):
            run_paper_loop(bars, cfg, champion_weights=_weights(), resume=True, resume_run_id=label)

    _persist(
        tmp_path,
        "ages-null",
        {"run_id": "ages-null", "step": 0, "mark_ages": None, "champion": _champion()},
    )
    _persist(
        tmp_path,
        "no-equity",
        {"run_id": "no-equity", "step": 2, "champion": _champion()},
    )
    with pytest.raises(ValueError, match="equity.parquet is missing"):
        run_paper_loop(
            bars, cfg, champion_weights=_weights(), resume=True, resume_run_id="no-equity"
        )

    counted = tmp_path / "metadata" / "paper" / "count"
    counted.mkdir(parents=True)
    (counted / "broker_state.json").write_text(
        json.dumps({"run_id": "count", "step": 2, "champion": _champion()})
    )
    _equity(1).write_parquet(counted / "equity.parquet")
    with pytest.raises(ValueError, match="row count does not match"):
        run_paper_loop(bars, cfg, champion_weights=_weights(), resume=True, resume_run_id="count")

    torn = tmp_path / "metadata" / "paper" / "torn-eq"
    torn.mkdir(parents=True)
    (torn / "broker_state.json").write_text(
        json.dumps({"run_id": "torn-eq", "step": 0, "champion": _champion()})
    )
    (torn / "equity.parquet").write_bytes(b"torn")
    with pytest.raises(ValueError, match="unreadable"):
        run_paper_loop(bars, cfg, champion_weights=_weights(), resume=True, resume_run_id="torn-eq")

    bad_book = tmp_path / "metadata" / "paper" / "bad-book"
    bad_book.mkdir(parents=True)
    (bad_book / "broker_state.json").write_text(
        json.dumps({"run_id": "bad-book", "step": 0, "champion": {"slot": "champion"}})
    )
    with pytest.raises(ValueError, match="invalid champion"):
        run_paper_loop(
            bars, cfg, champion_weights=_weights(), resume=True, resume_run_id="bad-book"
        )

    bad_shadow = tmp_path / "metadata" / "paper" / "bad-shadow"
    bad_shadow.mkdir(parents=True)
    (bad_shadow / "broker_state.json").write_text(
        json.dumps(
            {
                "run_id": "bad-shadow",
                "step": 0,
                "champion": _champion(),
                "shadow": {"slot": "shadow"},
            }
        )
    )
    with pytest.raises(ValueError, match="invalid shadow"):
        run_paper_loop(
            bars,
            cfg,
            champion_weights=_weights(),
            shadow_weights=_weights(),
            resume=True,
            resume_run_id="bad-shadow",
        )

    bad_promo = tmp_path / "metadata" / "paper" / "bad-promo"
    bad_promo.mkdir(parents=True)
    (bad_promo / "broker_state.json").write_text(
        json.dumps({"run_id": "bad-promo", "step": 0, "champion": _champion()})
    )
    (bad_promo / "promotion_dry_run.json").write_text("[]")
    with pytest.raises(ValueError, match="invalid promotion receipt"):
        run_paper_loop(
            bars, cfg, champion_weights=_weights(), resume=True, resume_run_id="bad-promo"
        )
    (bad_promo / "promotion_dry_run.json").write_text("{")
    with pytest.raises(ValueError, match="unreadable"):
        run_paper_loop(
            bars, cfg, champion_weights=_weights(), resume=True, resume_run_id="bad-promo"
        )


def test_empty_window_still_refuses_a_missing_book_and_keeps_the_claim_false(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    with pytest.raises(ValueError, match="champion_weights or champion_fn"):
        run_paper_loop(_bars(), cfg, max_steps=0)
    seen: list[str] = []

    def prepare(broker) -> None:
        seen.append(broker.slot)

    result = run_paper_loop(
        _bars(),
        cfg,
        champion_weights=_weights(),
        challenger_weights={"scale": _weights()},
        max_steps=0,
        prefer_latest=False,
        prepare_broker=prepare,
        run_id="empty-window",
    )
    assert seen == ["champion", "shadow"]
    assert result.metrics["live_pnl_claim"] is False
    assert result.metrics["research_only"] is True

    halted = run_paper_loop(
        _bars(),
        cfg,
        champion_weights=_weights(),
        max_steps=1,
        prefer_latest=False,
        halt_after_steps=1,
        run_id="halt-one",
    )
    assert halted.metrics["live_pnl_claim"] is False
    assert halted.metrics["promotion_dry_run"]["would_promote_live"] is False
