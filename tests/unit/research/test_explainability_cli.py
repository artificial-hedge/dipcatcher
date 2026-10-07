"""CLI bridge for the explainability research pack.

``quant_fund.research.explainability`` shipped with 7 modules, 16 tests and no
operator surface at all. These tests drive the new ``dipcatcher explain``
sub-typer end-to-end and pin the two things that make the pack honest:
attribution is measured as *proper-score degradation* (the planted signal
feature must dominate, the planted noise features must not), and the receipt
sidecar is additive — it binds report bytes without ever resealing the receipt,
and it fails closed when a bound report changes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
from typer.testing import CliRunner

from quant_fund.research.allocation_cli import allocation_app
from quant_fund.research.explainability_cli import _data_label, explainability_app

runner = CliRunner()

_FEATURES = ["mom_20", "reversal_1", "vol_20", "signal_core", "noise_a", "noise_b"]
_SIGNAL = "signal_core"
_NOISE = ("noise_a", "noise_b")


def test_data_label_matches_the_canonical_cli_formatter() -> None:
    """The local copy may not drift from ``cli.support.format_data_label``.

    Duplicated because the ``library-no-cli-imports`` architecture guard
    forbids a research-layer module from importing the CLI, even lazily.
    """
    from quant_fund.cli.support import format_data_label

    for synthetic, source in ((True, "ignored"), (False, "features.parquet")):
        assert _data_label(synthetic=synthetic, data_source=source) == format_data_label(
            synthetic=synthetic, data_source=source
        )


def _parse_attribution(output: str) -> dict[str, float]:
    """Read the ``#N feature importance=...`` block into {feature: importance}."""
    found: dict[str, float] = {}
    for line in output.splitlines():
        parts = line.split()
        if (
            len(parts) >= 3
            and parts[0].startswith("#")
            and any(p.startswith("importance=") for p in parts)
        ):
            token = next(p for p in parts if p.startswith("importance="))
            found[parts[1]] = float(token.split("=", 1)[1])
    return found


def test_demo_runs_and_writes_all_three_formats(tmp_path: Path) -> None:
    out = tmp_path / "reports"
    result = runner.invoke(
        explainability_app, ["demo", "--n-rows", "200", "--seed", "2026", "--out-dir", str(out)]
    )
    assert result.exit_code == 0, result.output
    assert "DATA_LABEL=SYNTHETIC" in result.output
    assert "model=demo_least_squares" in result.output
    assert "scoring=pinball_tau0.5" in result.output
    for name in ("explainability.md", "explainability.html", "explainability.json"):
        assert (out / name).is_file(), name
        assert (out / f"{name}.sha256").is_file(), f"{name}.sha256"
    assert f"markdown={out / 'explainability.md'}" in result.output


def test_demo_attribution_recovers_the_planted_signal(tmp_path: Path) -> None:
    """The panel plants all signal in ``signal_core`` and none in the noise
    columns, so permutation attribution must rank it first by a wide margin."""
    result = runner.invoke(
        explainability_app,
        ["demo", "--n-rows", "400", "--seed", "2026", "--out-dir", str(tmp_path)],
    )
    assert result.exit_code == 0, result.output
    importance = _parse_attribution(result.output)
    assert set(importance) == set(_FEATURES), importance
    assert max(importance, key=lambda f: importance[f]) == _SIGNAL
    assert importance[_SIGNAL] > 10 * max(importance[n] for n in _NOISE), importance
    for noise in _NOISE:
        assert abs(importance[noise]) < 0.01 * importance[_SIGNAL], importance


def test_demo_attribution_is_proper_score_degradation(tmp_path: Path) -> None:
    """Attribution is score degradation, never accuracy — so the reported
    baseline is a proper loss and the honest direction is 'lower is better'."""
    result = runner.invoke(
        explainability_app, ["demo", "--n-rows", "200", "--out-dir", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    assert "baseline_score=" in result.output
    assert "(lower is better)" in result.output
    report = json.loads((tmp_path / "explainability.json").read_text())
    assert report["scoring"].startswith("pinball")
    assert report["synthetic"] is True
    # A proper-score report carries no headline performance ratio.
    assert not {"sharpe", "sortino", "calmar", "pnl", "nav"} & set(report)


def test_demo_scoring_can_be_switched_to_crps(tmp_path: Path) -> None:
    """Multi-tau CRPS needs a matrix forecast, so the demo head emits a
    quantile fan on exactly the requested grid."""
    result = runner.invoke(
        explainability_app,
        ["demo", "--n-rows", "200", "--scoring", "crps:0.1,0.5,0.9", "--out-dir", str(tmp_path)],
    )
    assert result.exit_code == 0, result.output
    assert "scoring=crps_quantiles:0.1,0.5,0.9" in result.output
    importance = _parse_attribution(result.output)
    # The planted signal must dominate under CRPS too, not just under pinball.
    assert max(importance, key=lambda f: importance[f]) == _SIGNAL


def test_demo_rejects_brier_for_a_continuous_target(tmp_path: Path) -> None:
    result = runner.invoke(
        explainability_app,
        ["demo", "--n-rows", "120", "--scoring", "brier", "--out-dir", str(tmp_path)],
    )
    assert result.exit_code != 0
    assert "binary" in result.output


def test_demo_rejects_a_malformed_crps_grid(tmp_path: Path) -> None:
    result = runner.invoke(
        explainability_app,
        ["demo", "--n-rows", "120", "--scoring", "crps:0.9,0.1", "--out-dir", str(tmp_path)],
    )
    assert result.exit_code != 0
    assert "strictly increasing" in result.output


def test_demo_quantile_fan_is_ordered_and_brackets_the_median() -> None:
    """The fan the CRPS path relies on must be a genuine monotone quantile grid."""
    from quant_fund.research.explainability_cli import _demo_panel, _demo_taus

    taus = _demo_taus("crps:0.1,0.5,0.9")
    assert taus == (0.1, 0.5, 0.9)
    x, _y, _names, model = _demo_panel(5, 200, 0.3, taus=taus)
    pred = model.predict(x)
    assert pred.shape == (x.shape[0], 3)
    assert np.all(np.diff(pred, axis=1) >= 0.0), "quantile fan must be non-decreasing"
    point = _demo_panel(5, 200, 0.3)[3].predict(x)
    assert np.allclose(pred[:, 1], point), "the tau=0.5 column must be the point forecast"


def test_demo_drift_block_is_stable_for_a_fixed_linear_head(tmp_path: Path) -> None:
    result = runner.invoke(
        explainability_app, ["demo", "--n-rows", "300", "--out-dir", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    assert "drift blocks=4" in result.output
    assert "flagged=false" in result.output


def test_no_drift_flag_skips_the_block(tmp_path: Path) -> None:
    result = runner.invoke(
        explainability_app, ["demo", "--n-rows", "200", "--no-drift", "--out-dir", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    assert "drift blocks=" not in result.output


def test_shap_flag_skips_cleanly_when_the_extra_is_absent(tmp_path: Path, monkeypatch) -> None:
    """``shap`` is an optional extra and is excluded on darwin/x86_64 — a
    missing import must be a typed skip, never an ImportError traceback."""
    monkeypatch.setitem(sys.modules, "shap", None)
    result = runner.invoke(
        explainability_app, ["demo", "--n-rows", "120", "--shap", "--out-dir", str(tmp_path)]
    )
    assert result.exit_code == 3, result.output
    assert "EXPLAINABILITY_SHAP_SKIP" in result.output
    assert "uv sync --extra explainability" in result.output
    assert "Traceback" not in result.output


def test_shap_flag_runs_when_the_extra_is_present(tmp_path: Path) -> None:
    try:
        import shap  # noqa: F401
    except ImportError:
        return  # extra absent on this platform; the skip path is covered above
    result = runner.invoke(
        explainability_app,
        ["demo", "--n-rows", "120", "--shap", "--out-dir", str(tmp_path)],
    )
    assert result.exit_code == 0, result.output
    assert "shap method=" in result.output
    importance = _parse_attribution(result.output)
    assert importance.get(_SIGNAL, 0.0) > 0.0


def test_report_on_a_user_model_and_matrix(tmp_path: Path) -> None:
    """The real-data path: a joblib head plus a parquet feature/label matrix."""
    import joblib

    from quant_fund.research.explainability_cli import _demo_panel, _LeastSquaresHead

    x, y, names, model = _demo_panel(11, 240, 0.3)
    model_path = tmp_path / "head.joblib"
    joblib.dump(_LeastSquaresHead(model.coef), model_path)

    frame = pl.DataFrame({name: x[:, i] for i, name in enumerate(names)})
    frame = frame.with_columns(pl.Series("label", y))
    features = tmp_path / "features.parquet"
    frame.write_parquet(features)

    out = tmp_path / "reports"
    result = runner.invoke(
        explainability_app,
        [
            "report",
            "--model-path",
            str(model_path),
            "--features",
            str(features),
            "--label-col",
            "label",
            "--out-dir",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert f"DATA_LABEL={features.name}" in result.output
    assert "DATA_LABEL=SYNTHETIC" not in result.output
    assert (out / "explainability.json").is_file()
    report = json.loads((out / "explainability.json").read_text())
    assert report["synthetic"] is False
    importance = _parse_attribution(result.output)
    assert max(importance, key=lambda f: importance[f]) == _SIGNAL


def test_report_requires_a_label_source(tmp_path: Path) -> None:
    import joblib

    from quant_fund.research.explainability_cli import _LeastSquaresHead

    model_path = tmp_path / "head.joblib"
    joblib.dump(_LeastSquaresHead(np.zeros(3)), model_path)
    features = tmp_path / "features.parquet"
    pl.DataFrame({"a": np.zeros(40), "b": np.zeros(40), "y": np.zeros(40)}).write_parquet(features)

    neither = runner.invoke(
        explainability_app,
        ["report", "--model-path", str(model_path), "--features", str(features)],
    )
    assert neither.exit_code != 0
    assert "--labels or --label-col" in neither.output

    both = runner.invoke(
        explainability_app,
        [
            "report",
            "--model-path",
            str(model_path),
            "--features",
            str(features),
            "--label-col",
            "y",
            "--labels",
            str(features),
        ],
    )
    assert both.exit_code != 0
    assert "not both" in both.output


def _sealed_receipt(directory: Path) -> Path:
    """A real sealed receipt, produced by the allocation lane (fast, offline)."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "allocation.json"
    result = runner.invoke(
        allocation_app,
        [
            "walk-forward",
            "--engine",
            "risk_parity",
            "--n-assets",
            "3",
            "--n-periods",
            "160",
            "--window",
            "50",
            "--step",
            "10",
            "--out",
            str(path),
        ],
    )
    assert result.exit_code == 0, result.output
    return path


def test_sidecar_round_trip_and_tamper_detection(tmp_path: Path) -> None:
    receipt = _sealed_receipt(tmp_path)
    before = receipt.read_bytes()

    reports = tmp_path / "reports"
    demo = runner.invoke(explainability_app, ["demo", "--n-rows", "150", "--out-dir", str(reports)])
    assert demo.exit_code == 0, demo.output

    attach = runner.invoke(explainability_app, ["attach-sidecar", str(receipt), str(reports)])
    assert attach.exit_code == 0, attach.output
    from quant_fund.research.explainability import explainability_sidecar_path

    sidecar = explainability_sidecar_path(receipt)
    assert f"sidecar={sidecar}" in attach.output
    assert sidecar.is_file()

    # Additive only: the sealed receipt bytes are untouched.
    assert receipt.read_bytes() == before

    clean = runner.invoke(explainability_app, ["verify-sidecar", str(receipt)])
    assert clean.exit_code == 0, clean.output
    body = json.loads(clean.output)
    assert body["valid"] is True
    assert body["errors"] == []
    assert body["reports_checked"] == 3

    # A changed report byte must fail the binding closed.
    with (reports / "explainability.json").open("a") as handle:
        handle.write("tampered")
    tampered = runner.invoke(explainability_app, ["verify-sidecar", str(receipt)])
    assert tampered.exit_code == 1
    assert "sidecar_report_hash_mismatch" in tampered.output


def test_verify_sidecar_fails_closed_when_absent(tmp_path: Path) -> None:
    receipt = _sealed_receipt(tmp_path)
    result = runner.invoke(explainability_app, ["verify-sidecar", str(receipt)])
    assert result.exit_code == 1
    assert "sidecar_missing" in result.output


def test_attach_rejects_reports_outside_the_receipt_root(tmp_path: Path) -> None:
    """The sidecar binds paths relative to the receipt root, so a report
    elsewhere is rejected rather than silently bound to an unverifiable path."""
    receipt = _sealed_receipt(tmp_path / "tree")
    outside = tmp_path / "outside"
    demo = runner.invoke(explainability_app, ["demo", "--n-rows", "120", "--out-dir", str(outside)])
    assert demo.exit_code == 0, demo.output

    result = runner.invoke(explainability_app, ["attach-sidecar", str(receipt), str(outside)])
    assert result.exit_code != 0
    assert "report_outside_receipt_root" in result.output
