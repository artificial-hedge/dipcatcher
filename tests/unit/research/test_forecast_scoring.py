"""User-supplied forecast scoring: proper scores, sealed receipts, fail-closed inputs.

SYNTHETIC fixtures here are correctness checks. They are not market evidence.
"""

from __future__ import annotations

import json
import math
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
from scipy.stats import chi2

from quant_fund.research.catalog import family_blob_forbidden_metrics_absent
from quant_fund.research.forecast_scoring import (
    FORECAST_SCORING_KIND,
    MAX_BYTES,
    MAX_ROWS,
    MIN_ROWS,
    N_BOOTSTRAP,
    REPLAY_GENERATED_AT,
    ForecastScoringError,
    forecast_scoring_v2_consistency_errors,
    inspect_dataset,
    run_selftest,
    score_forecasts,
)
from quant_fund.schemas.receipt_v2_leaf import verify_receipt_file


@pytest.fixture(autouse=True)
def _isolated_fx1_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("FX1_CONFIG_DIR", str(tmp_path / "fx1-config"))


def _ts(index: int) -> str:
    return f"2020-01-{(index // 24) + 1:02d}T{index % 24:02d}:00:00+00:00"


def _write(path: Path, header: str, rows: list[str]) -> None:
    path.write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")


def _vol_rows(n: int, realized: float = 1.0, forecast: float = 2.0) -> list[str]:
    return [f"{_ts(i)},{realized},{forecast}" for i in range(n)]


def _score_vol(path: Path, out: Path, **kwargs: object):
    return score_forecasts(path, task="volatility", out_dir=out, **kwargs)


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _pinball(y: float, q: float, tau: float) -> float:
    diff = y - q
    return max(tau * diff, (tau - 1.0) * diff)


def test_constants_match_the_contract() -> None:
    """Row, byte, and bootstrap caps are the contract values."""
    assert MIN_ROWS == 30
    assert MAX_ROWS == 2_000_000
    assert MAX_BYTES == 200_000_000
    assert N_BOOTSTRAP == 1000


def test_perfect_variance_forecast_has_zero_qlike(tmp_path: Path) -> None:
    """A forecast equal to the realized variance has QLIKE 0, MSE 0, and ratio 1."""
    path = tmp_path / "perfect.csv"
    _write(path, "timestamp,realized,forecast_var", _vol_rows(40, 1.0, 1.0))
    outcome = _score_vol(path, tmp_path / "out")
    assert outcome.scores["qlike"] == pytest.approx(0.0, abs=1e-15)
    assert outcome.scores["mse"] == pytest.approx(0.0, abs=1e-15)
    assert outcome.scores["calibration_ratio"] == pytest.approx(1.0)
    assert outcome.data_label == "file"
    assert outcome.rows == 40
    assert outcome.task == "volatility"
    verdict = verify_receipt_file(outcome.receipt_path)
    assert verdict["valid"] is True


def test_constant_variance_ratio_matches_the_qlike_formula(tmp_path: Path) -> None:
    """QLIKE of realized=1, forecast=2 is 1/2 - log(1/2) - 1, with MSE 1."""
    path = tmp_path / "ratio.csv"
    _write(path, "timestamp,realized,forecast_var", _vol_rows(40, 1.0, 2.0))
    outcome = _score_vol(path, tmp_path / "out")
    expected = 0.5 - math.log(0.5) - 1.0
    assert outcome.scores["qlike"] == pytest.approx(expected)
    assert outcome.scores["mse"] == pytest.approx(1.0)
    assert outcome.scores["calibration_ratio"] == pytest.approx(0.5)


def test_quantile_scores_match_a_hand_computed_pinball_and_crps(tmp_path: Path) -> None:
    """Alternating 0/1 outcomes against fixed quantiles have a known pinball and CRPS."""
    rows = []
    for i in range(30):
        y = 0 if i % 2 == 0 else 1
        rows.append(f"{_ts(i)},{y},-1,0.5,2")
    path = tmp_path / "q.csv"
    _write(path, "timestamp,realized,q05,q50,q95", rows)
    outcome = score_forecasts(path, task="quantile", out_dir=tmp_path / "out")
    ys = [0.0 if i % 2 == 0 else 1.0 for i in range(30)]
    levels = (("q05", 0.05, -1.0), ("q50", 0.5, 0.5), ("q95", 0.95, 2.0))
    per_tau = []
    for token, tau, q in levels:
        losses = [_pinball(y, q, tau) for y in ys]
        per_tau.append(sum(losses) / len(losses))
        assert outcome.scores[f"pinball_{token}"] == pytest.approx(per_tau[-1])
    assert outcome.scores["mean_pinball"] == pytest.approx(sum(per_tau) / len(per_tau))
    # CRPS ≈ 2 * integral pinball dτ with left Riemann weights from 0.
    weights = (0.05, 0.45, 0.45)
    crps_rows = []
    for y in ys:
        total = 0.0
        for (_token, tau, q), weight in zip(levels, weights, strict=True):
            total += 2.0 * _pinball(y, q, tau) * weight
        crps_rows.append(total)
    assert outcome.scores["crps"] == pytest.approx(sum(crps_rows) / len(crps_rows))
    assert outcome.scores["crps"] == pytest.approx(0.3)
    assert outcome.scores["coverage_q05"] == pytest.approx(0.0)
    assert outcome.scores["coverage_q50"] == pytest.approx(0.5)
    assert outcome.scores["coverage_q95"] == pytest.approx(1.0)
    assert "kupiec_pvalue_q50" not in outcome.scores


def test_kupiec_tail_matches_the_likelihood_ratio(tmp_path: Path) -> None:
    """Four lower-tail hits in 40 rows produce the Kupiec LR for alpha 0.05."""
    rows = []
    for i in range(40):
        y = 0 if i < 4 else 1
        rows.append(f"{_ts(i)},{y},0.5,2,3")
    path = tmp_path / "k.csv"
    _write(path, "timestamp,realized,q05,q50,q95", rows)
    outcome = score_forecasts(path, task="quantile", out_dir=tmp_path / "out")
    n, x, alpha = 40, 4, 0.05
    rate = x / n
    expected = -2.0 * (
        (n - x) * math.log(1.0 - alpha)
        + x * math.log(alpha)
        - (n - x) * math.log(1.0 - rate)
        - x * math.log(rate)
    )
    assert outcome.scores["kupiec_lr_q05"] == pytest.approx(expected)
    assert "kupiec_lr_q50" not in outcome.scores
    assert outcome.scores["kupiec_pvalue_q05"] == pytest.approx(float(chi2.sf(expected, 1)))


def test_probability_scores_match_brier_log_loss_and_ece(tmp_path: Path) -> None:
    """Constant 0.25 forecasts on a balanced 0/1 sample have Brier 0.3125 and ECE 0.25."""
    rows = [f"{_ts(i)},{0 if i < 20 else 1},0.25" for i in range(40)]
    path = tmp_path / "p.csv"
    _write(path, "timestamp,realized,forecast_prob", rows)
    outcome = score_forecasts(path, task="probability", out_dir=tmp_path / "out")
    assert outcome.scores["brier"] == pytest.approx(0.3125)
    expected_ll = -0.5 * (math.log(0.75) + math.log(0.25))
    assert outcome.scores["log_loss"] == pytest.approx(expected_ll)
    assert outcome.scores["ece"] == pytest.approx(0.25)
    body = _load(outcome.receipt_path)
    payload = body["payload"]
    assert isinstance(payload, dict)
    diagnostics = payload["diagnostics"]
    assert isinstance(diagnostics, dict)
    table = diagnostics["reliability"]
    assert isinstance(table, list) and len(table) == 10
    total = sum(int(row["count"]) for row in table if isinstance(row, dict))
    ece = 0.0
    for row in table:
        assert isinstance(row, dict)
        count = int(row["count"])
        if count == 0:
            continue
        ece += (count / total) * abs(float(row["mean_outcome"]) - float(row["mean_forecast"]))
    assert ece == pytest.approx(outcome.scores["ece"])
    assert payload["params"]["log_loss_eps"] == 1e-12
    assert payload["params"]["ece_bins"] == 10


def test_explicit_quantile_columns_match_autodetected_names(tmp_path: Path) -> None:
    """--quantile-cols renames columns without changing the pinball of the same numbers."""
    rows = [f"{_ts(i)},{0 if i % 2 == 0 else 1},-1,0.5,2" for i in range(30)]
    named = tmp_path / "named.csv"
    _write(named, "timestamp,realized,low,mid,high", rows)
    outcome = score_forecasts(
        named,
        task="quantile",
        out_dir=tmp_path / "out",
        quantile_cols="q05=low,q50=mid,q95=high",
    )
    assert outcome.scores["pinball_q05"] == pytest.approx(0.075)
    assert outcome.scores["pinball_q50"] == pytest.approx(0.25)


def test_q5_token_is_the_median_and_skips_kupiec(tmp_path: Path) -> None:
    """A one-digit q5 token is tau 0.5, which is not a tail."""
    rows = [f"{_ts(i)},{0 if i % 2 == 0 else 1},0.5" for i in range(30)]
    path = tmp_path / "med.csv"
    _write(path, "timestamp,realized,q5", rows)
    outcome = score_forecasts(path, task="quantile", out_dir=tmp_path / "out")
    assert "pinball_q5" in outcome.scores
    assert not any(name.startswith("kupiec_") for name in outcome.scores)


def test_equal_quantiles_are_not_crossings(tmp_path: Path) -> None:
    """Identical quantile values are monotone and score."""
    rows = [f"{_ts(i)},0,1,1,1" for i in range(30)]
    path = tmp_path / "tie.csv"
    _write(path, "timestamp,realized,q05,q50,q95", rows)
    outcome = score_forecasts(path, task="quantile", out_dir=tmp_path / "out")
    assert outcome.rows == 30


def test_receipt_is_a_sealed_v2_file_label_without_forbidden_keys(tmp_path: Path) -> None:
    """A file-backed run seals forecast_scoring.v1, research-only, with no headline ratio keys."""
    path = tmp_path / "v.csv"
    _write(path, "timestamp,realized,forecast_var,sharpe", [f"{_ts(i)},1,1.1,9" for i in range(30)])
    outcome = score_forecasts(
        path, task="volatility", out_dir=tmp_path / "out", forecast_col="forecast_var"
    )
    body = _load(outcome.receipt_path)
    assert body["schema"] == "receipt.v2"
    assert body["kind"] == FORECAST_SCORING_KIND
    assert body["data_label"] == "file"
    assert body["live_pnl_claim"] is False
    assert body["verdict"] == "pass"
    assert body["generated_at"] == REPLAY_GENERATED_AT
    assert body["receipt_sha256"] == outcome.receipt_sha256
    payload = body["payload"]
    assert isinstance(payload, dict)
    assert payload["research_only"] is True
    assert payload["live_pnl_claim"] is False
    assert payload["kind"] == FORECAST_SCORING_KIND
    assert payload["data_label"] == "file"
    assert payload["scores"]["qlike"] == pytest.approx(outcome.scores["qlike"])
    assert "dataset_sha256" not in payload
    assert "tape_manifest_sha256" not in body
    assert family_blob_forbidden_metrics_absent(body) is True
    assert forecast_scoring_v2_consistency_errors(body) == []
    assert outcome.receipt_path.name == f"forecast_scoring_{outcome.receipt_sha256[:16]}.json"


def test_same_bytes_in_two_directories_seal_the_same_receipt(tmp_path: Path) -> None:
    """The seal binds file bytes and arguments, not the path the file was copied to."""
    source = tmp_path / "a.csv"
    _write(source, "timestamp,realized,forecast_var", _vol_rows(35, 1.0, 1.5))
    copy = tmp_path / "nested" / "b.csv"
    copy.parent.mkdir()
    copy.write_bytes(source.read_bytes())
    first = _score_vol(source, tmp_path / "out1", seed=3)
    second = _score_vol(copy, tmp_path / "out2", seed=3)
    assert first.receipt_sha256 == second.receipt_sha256
    assert first.receipt_path.read_bytes() == second.receipt_path.read_bytes()
    third = _score_vol(source, tmp_path / "out1", seed=3)
    assert third.receipt_path == first.receipt_path
    assert list(first.receipt_path.parent.iterdir()) == [first.receipt_path]


def test_different_seeds_change_the_interval_and_not_the_point_score(tmp_path: Path) -> None:
    """The bootstrap seed moves the sealed interval. The QLIKE point does not move."""
    path = tmp_path / "v.csv"
    rows = [f"{_ts(i)},{1.0 + (i % 5) * 0.25},1.2" for i in range(40)]
    _write(path, "timestamp,realized,forecast_var", rows)
    first = _score_vol(path, tmp_path / "a", seed=0)
    second = _score_vol(path, tmp_path / "b", seed=1)
    assert first.scores["qlike"] == pytest.approx(second.scores["qlike"])
    assert first.receipt_sha256 != second.receipt_sha256
    left = _load(first.receipt_path)["payload"]["diagnostics"]["bootstrap"]
    right = _load(second.receipt_path)["payload"]["diagnostics"]["bootstrap"]
    assert left["point"] == pytest.approx(first.scores["qlike"])
    assert left["low"] <= left["high"]
    assert (left["low"], left["high"]) != (right["low"], right["high"])
    assert left["n_resamples"] == 1000


def test_time_blocks_partition_the_sample_and_name_the_worst_block(tmp_path: Path) -> None:
    """Up to ten equal-count blocks cover every row, and the weakest has the largest loss."""
    path = tmp_path / "v.csv"
    rows = [f"{_ts(i)},{1.0 + (i % 7) * 0.2},{1.0 + (i % 3) * 0.05}" for i in range(40)]
    _write(path, "timestamp,realized,forecast_var", rows)
    outcome = _score_vol(path, tmp_path / "out")
    diagnostics = _load(outcome.receipt_path)["payload"]["diagnostics"]
    blocks = diagnostics["blocks"]
    assert len(blocks) <= 10
    assert sum(block["n"] for block in blocks) == outcome.rows
    assert blocks[0]["start"] == 0
    assert blocks[-1]["end"] == outcome.rows
    weighted = sum(block["primary"] * block["n"] for block in blocks) / outcome.rows
    assert weighted == pytest.approx(outcome.scores["qlike"])
    weakest = diagnostics["weakest_block"]
    assert weakest["primary"] == max(block["primary"] for block in blocks)


def test_missing_issue_time_records_the_point_in_time_limitation(tmp_path: Path) -> None:
    """Without available_time the run scores and says point-in-time validity was not checked."""
    path = tmp_path / "v.csv"
    _write(path, "timestamp,realized,forecast_var", _vol_rows(30, 1.0, 1.0))
    outcome = _score_vol(path, tmp_path / "out")
    limitations = _load(outcome.receipt_path)["payload"]["limitations"]
    assert any("point-in-time validity is NOT verified" in note for note in limitations)
    pit = _load(outcome.receipt_path)["payload"]["diagnostics"]["point_in_time"]
    assert pit["verified"] is False


def test_earlier_available_time_is_accepted_and_marked_verified(tmp_path: Path) -> None:
    """Issue times strictly before the realization verify point-in-time order."""
    rows = [f"{_ts(i)},2019-12-31T{i % 24:02d}:00:00+00:00,1,1" for i in range(30)]
    path = tmp_path / "pit.csv"
    _write(path, "timestamp,available_time,realized,forecast_var", rows)
    outcome = _score_vol(path, tmp_path / "out", available_time_col="available_time")
    payload = _load(outcome.receipt_path)["payload"]
    assert payload["diagnostics"]["point_in_time"]["verified"] is True
    assert not any("NOT verified" in note for note in payload["limitations"])


def test_naive_dates_are_labeled_as_utc(tmp_path: Path) -> None:
    """Date-only timestamps score and the receipt says they were read as UTC."""
    rows = [f"2020-01-{i + 1:02d},1,1" for i in range(30)]
    path = tmp_path / "dates.csv"
    _write(path, "timestamp,realized,forecast_var", rows)
    outcome = _score_vol(path, tmp_path / "out")
    notes = _load(outcome.receipt_path)["payload"]["limitations"]
    assert any("Naive timestamps were interpreted as UTC" in note for note in notes)


def test_equal_instants_in_different_zones_are_not_increasing(tmp_path: Path) -> None:
    """Timestamps are compared after conversion to UTC, so equal instants are refused."""
    rows = ["2020-01-02T00:00:00-05:00,1,1", "2020-01-02T05:00:00+00:00,1,1"]
    rows.extend(f"2020-01-{4 + i // 24:02d}T{i % 24:02d}:00:00+00:00,1,1" for i in range(28))
    path = tmp_path / "tz.csv"
    _write(path, "timestamp,realized,forecast_var", rows)
    with pytest.raises(ForecastScoringError, match="strictly increasing"):
        _score_vol(path, tmp_path / "out")


def test_synthetic_column_and_preamble_stay_synthetic(tmp_path: Path) -> None:
    """A data_label column or a CSV preamble of SYNTHETIC is copied onto the receipt."""
    rows = [f"{_ts(i)},1,1,SYNTHETIC" for i in range(30)]
    columned = tmp_path / "col.csv"
    _write(columned, "timestamp,realized,forecast_var,data_label", rows)
    by_column = _score_vol(columned, tmp_path / "a")
    assert by_column.data_label == "SYNTHETIC"
    preamble = tmp_path / "pre.csv"
    preamble.write_text(
        "# DATA_LABEL=SYNTHETIC\n"
        + "timestamp,realized,forecast_var\n"
        + "\n".join(_vol_rows(30, 1, 1))
        + "\n",
        encoding="utf-8",
    )
    by_preamble = _score_vol(preamble, tmp_path / "b")
    assert by_preamble.data_label == "SYNTHETIC"
    notes = _load(by_preamble.receipt_path)["payload"]["limitations"]
    assert any("SYNTHETIC" in note for note in notes)


def test_real_and_mixed_data_labels_are_refused(tmp_path: Path) -> None:
    """Only SYNTHETIC or file are accepted in a data_label column."""
    rows = [f"{_ts(i)},1,1,REAL" for i in range(30)]
    path = tmp_path / "real.csv"
    _write(path, "timestamp,realized,forecast_var,data_label", rows)
    with pytest.raises(ForecastScoringError, match="data_label column"):
        _score_vol(path, tmp_path / "out")


def test_conflicting_synthetic_markers_are_refused(tmp_path: Path) -> None:
    """A file column and a SYNTHETIC preamble cannot disagree."""
    body = "\n".join(f"{_ts(i)},1,1,file" for i in range(30))
    path = tmp_path / "mix.csv"
    path.write_text(
        "# SYNTHETIC\ntimestamp,realized,forecast_var,data_label\n" + body + "\n", encoding="utf-8"
    )
    with pytest.raises(ForecastScoringError, match="preamble"):
        _score_vol(path, tmp_path / "out")


def test_csv_and_parquet_of_the_same_numbers_share_scores(tmp_path: Path) -> None:
    """Parquet and CSV transports of one series produce the same proper scores."""
    stamps = [datetime(2020, 1, 1, tzinfo=UTC) + timedelta(hours=i) for i in range(36)]
    realized = [1.0 + (i % 4) * 0.25 for i in range(36)]
    forecast = [1.5] * 36
    frame = pl.DataFrame({"timestamp": stamps, "realized": realized, "forecast_var": forecast})
    parquet = tmp_path / "v.parquet"
    csv_path = tmp_path / "v.csv"
    frame.write_parquet(parquet)
    frame.write_csv(csv_path)
    from_parquet = _score_vol(parquet, tmp_path / "pq")
    from_csv = _score_vol(csv_path, tmp_path / "csv")
    assert from_parquet.scores["qlike"] == pytest.approx(from_csv.scores["qlike"])
    assert from_parquet.scores["mse"] == pytest.approx(from_csv.scores["mse"])
    assert from_parquet.receipt_sha256 != from_csv.receipt_sha256


def test_byte_edit_score_edit_and_dataset_hash_edit_fail_verification(tmp_path: Path) -> None:
    """Any byte change, score edit, or dataset-hash edit makes verify-receipt say invalid."""
    path = tmp_path / "v.csv"
    _write(path, "timestamp,realized,forecast_var", _vol_rows(32, 1.0, 1.4))
    outcome = _score_vol(path, tmp_path / "out")
    original = outcome.receipt_path.read_bytes()

    flipped = bytearray(original)
    flipped[len(flipped) // 2] ^= 0x1
    byte_path = tmp_path / "flipped.json"
    byte_path.write_bytes(flipped)
    assert verify_receipt_file(byte_path)["valid"] is False

    edited = json.loads(original)
    edited["payload"]["scores"]["qlike"] = 123.0
    score_path = tmp_path / "score.json"
    score_path.write_text(json.dumps(edited), encoding="utf-8")
    assert verify_receipt_file(score_path)["valid"] is False

    rehashed = json.loads(original)
    digest = str(rehashed["dataset_hash"])
    replacement = ("0" if digest[0] != "0" else "1") + digest[1:]
    rehashed["dataset_hash"] = replacement
    hash_path = tmp_path / "hash.json"
    hash_path.write_text(json.dumps(rehashed), encoding="utf-8")
    assert verify_receipt_file(hash_path)["valid"] is False
    drifted = json.loads(original)
    drifted["payload"]["dataset"]["file_sha256"] = "ab" * 32
    assert "dataset_hash_mismatch" in forecast_scoring_v2_consistency_errors(drifted)


def test_refuses_to_overwrite_a_different_file_with_the_same_name(tmp_path: Path) -> None:
    """An existing receipt with different bytes and the same name is left untouched."""
    path = tmp_path / "v.csv"
    _write(path, "timestamp,realized,forecast_var", _vol_rows(30, 1.0, 1.0))
    out = tmp_path / "out"
    outcome = _score_vol(path, out)
    outcome.receipt_path.write_text("{not the receipt}\n", encoding="utf-8")
    with pytest.raises(ForecastScoringError, match="different content"):
        _score_vol(path, out)
    assert outcome.receipt_path.read_text(encoding="utf-8") == "{not the receipt}\n"


def test_symlink_receipt_path_is_refused(tmp_path: Path) -> None:
    """A symlink sitting on the receipt filename is not replaced."""
    path = tmp_path / "v.csv"
    _write(path, "timestamp,realized,forecast_var", _vol_rows(30, 1.0, 1.2))
    out = tmp_path / "out"
    outcome = _score_vol(path, out)
    target = tmp_path / "elsewhere.json"
    target.write_text("other\n", encoding="utf-8")
    outcome.receipt_path.unlink()
    outcome.receipt_path.symlink_to(target)
    with pytest.raises(ForecastScoringError, match="symlink"):
        _score_vol(path, out)
    assert target.read_text(encoding="utf-8") == "other\n"


def test_symlink_output_directory_keeps_the_receipt_under_that_path(tmp_path: Path) -> None:
    """An operator-chosen symlink out dir is honored; the returned path stays inside it."""
    path = tmp_path / "v.csv"
    _write(path, "timestamp,realized,forecast_var", _vol_rows(30, 1.0, 1.2))
    real = tmp_path / "real"
    link = tmp_path / "link"
    real.mkdir()
    link.symlink_to(real, target_is_directory=True)
    outcome = _score_vol(path, link)
    assert outcome.receipt_path.parent == link
    assert outcome.receipt_path.is_file()
    assert (real / outcome.receipt_path.name).is_file()


def test_inspect_profiles_without_a_minimum_row_count(tmp_path: Path) -> None:
    """Inspect returns dataset_profile.v1 for a short file and does not score it."""
    path = tmp_path / "short.csv"
    _write(path, "timestamp,realized,forecast_var,q05,forecast_prob", [f"{_ts(0)},1,1,0.1,0.2"])
    profile = inspect_dataset(path)
    assert profile["schema"] == "dataset_profile.v1"
    assert profile["n_rows"] == 1
    assert profile["sha256"]
    assert profile["size_bytes"] == path.stat().st_size
    roles = profile["suggested_roles"]
    assert isinstance(roles, dict)
    assert roles["time"]["column"] == "timestamp"
    assert roles["time"]["confidence"] == "exact_name"
    assert roles["quantiles"][0]["column"] == "q05"
    assert roles["quantiles"][0]["tau"] == pytest.approx(0.05)
    assert "heuristics" in str(profile["confidence_note"])
    time_info = profile["time"]
    assert isinstance(time_info, dict)
    assert time_info["parsed"] is True
    assert time_info["strictly_increasing"] is False
    assert not list(tmp_path.glob("*.json"))


def test_inspect_flags_duplicate_timestamps(tmp_path: Path) -> None:
    """Inspect reports non-monotonic time instead of refusing the file."""
    path = tmp_path / "dups.csv"
    rows = [f"{_ts(0)},1", f"{_ts(0)},1", f"{_ts(2)},1"]
    _write(path, "timestamp,realized", rows)
    profile = inspect_dataset(path)
    assert profile["time"]["strictly_increasing"] is False
    assert profile["time"]["duplicate_count"] == 1


@pytest.mark.parametrize(
    ("name", "builder", "match"),
    [
        ("missing", "missing", "not found"),
        ("directory", "directory", "not a file"),
        ("empty", "empty", "empty"),
        ("extension", "extension", "unsupported extension"),
        ("short", "short", "at least 30"),
        ("nan-token", "nan-token", "finite"),
        ("inf", "inf", "finite"),
        ("null", "null", "null"),
        ("zero-variance", "zero-variance", "positive"),
        ("negative-variance", "negative-variance", "positive"),
        ("zero-realized", "zero-realized", "positive"),
        ("crossing", "crossing", "cross"),
        ("lookahead-equal", "lookahead-equal", "lookahead"),
        ("lookahead-later", "lookahead-later", "lookahead"),
        ("decreasing", "decreasing", "strictly increasing"),
        ("duplicate-time", "duplicate-time", "strictly increasing"),
        ("duplicate-header", "duplicate-header", "duplicate column"),
        ("missing-column", "missing-column", "missing column"),
        ("prob-range", "prob-range", "\\[0, 1\\]"),
        ("prob-outcome", "prob-outcome", "0 or 1"),
        ("bad-task", "bad-task", "task must be one of"),
        ("bad-version", "bad-version", "receipt-version"),
        ("bad-seed", "bad-seed", "seed"),
        ("quantile-on-vol", "quantile-on-vol", "quantile-cols"),
        ("forecast-on-quantile", "forecast-on-quantile", "forecast-col"),
        ("no-quantiles", "no-quantiles", "q05"),
        ("corrupt-parquet", "corrupt-parquet", "could not read"),
    ],
)
def test_fail_closed_inputs(tmp_path: Path, name: str, builder: str, match: str) -> None:
    """Each contract violation raises ForecastScoringError and writes no receipt."""
    out = tmp_path / "out"
    path = _failure_file(tmp_path, builder)
    with pytest.raises(ForecastScoringError, match=match):
        _invoke_failure(path, out, builder)
    assert not out.exists() or not any(out.glob("forecast_scoring_*.json")), name


def test_oversize_file_is_refused_before_parsing(tmp_path: Path) -> None:
    """A sparse file over 200 MB is refused from its size, not parsed as a table."""
    path = tmp_path / "big.csv"
    path.write_bytes(b"this is not a table")
    os.truncate(path, MAX_BYTES + 1)
    with pytest.raises(ForecastScoringError, match="200000000"):
        _score_vol(path, tmp_path / "out")


def test_row_cap_uses_the_configured_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """More rows than the cap are refused. The test lowers the cap so it stays small."""
    monkeypatch.setattr("quant_fund.research.forecast_scoring.MAX_ROWS", 30)
    path = tmp_path / "wide.csv"
    _write(path, "timestamp,realized,forecast_var", _vol_rows(31, 1.0, 1.0))
    with pytest.raises(ForecastScoringError, match="exceeds 30"):
        _score_vol(path, tmp_path / "out")


def test_selftest_passes_for_every_task_and_rejects_a_tamper() -> None:
    """Each task builds a SYNTHETIC fixture, verifies the receipt, and rejects a flipped byte."""
    for task in ("volatility", "quantile", "probability"):
        result = run_selftest(task)
        assert result.passed is True
        assert result.verified is True
        assert result.task == task
        assert result.receipt_sha256 is not None and len(result.receipt_sha256) == 64
        assert result.duration_s < 2.0
        assert "tampered copy rejected" in result.detail
    unknown = run_selftest("momentum")
    assert unknown.passed is False
    assert unknown.verified is False
    assert unknown.receipt_sha256 is None


def test_endpoint_probabilities_are_allowed(tmp_path: Path) -> None:
    """Forecast probabilities of exactly 0 and 1 are inside [0, 1] and score."""
    rows = []
    for i in range(30):
        probability = 0.0 if i % 2 == 0 else 1.0
        outcome = 0 if i % 2 == 0 else 1
        rows.append(f"{_ts(i)},{outcome},{probability}")
    path = tmp_path / "ends.csv"
    _write(path, "timestamp,realized,forecast_prob", rows)
    outcome = score_forecasts(path, task="probability", out_dir=tmp_path / "out")
    assert math.isfinite(outcome.scores["log_loss"])
    assert outcome.scores["brier"] == pytest.approx(0.0)


def _failure_file(tmp_path: Path, builder: str) -> Path:
    path = tmp_path / ("bad.parquet" if builder == "corrupt-parquet" else "bad.csv")
    if builder == "missing":
        return tmp_path / "missing.csv"
    if builder == "directory":
        path.mkdir()
        return path
    if builder == "empty":
        path.write_bytes(b"")
        return path
    if builder == "extension":
        text = tmp_path / "bad.txt"
        text.write_text("timestamp,realized,forecast_var\n", encoding="utf-8")
        return text
    if builder == "corrupt-parquet":
        path.write_bytes(b"not parquet")
        return path
    if builder == "duplicate-header":
        rows = "\n".join(f"{_ts(i)},1,1" for i in range(30))
        path.write_text("timestamp,realized,realized\n" + rows + "\n", encoding="utf-8")
        return path
    header, rows = _failure_table(builder)
    if header:
        _write(path, header, rows)
    return path


def _failure_table(builder: str) -> tuple[str, list[str]]:
    n = 30
    if builder == "short":
        return "timestamp,realized,forecast_var", _vol_rows(10, 1, 1)
    if builder == "nan-token":
        rows = _vol_rows(29, 1, 1) + [f"{_ts(29)},nan,1"]
        return "timestamp,realized,forecast_var", rows
    if builder == "inf":
        rows = _vol_rows(29, 1, 1) + [f"{_ts(29)},inf,1"]
        return "timestamp,realized,forecast_var", rows
    if builder == "null":
        rows = _vol_rows(29, 1, 1) + [f"{_ts(29)},,1"]
        return "timestamp,realized,forecast_var", rows
    if builder == "zero-variance":
        return "timestamp,realized,forecast_var", _vol_rows(n, 1, 0)
    if builder == "negative-variance":
        return "timestamp,realized,forecast_var", _vol_rows(n, 1, -1)
    if builder == "zero-realized":
        return "timestamp,realized,forecast_var", _vol_rows(n, 0, 1)
    if builder == "crossing":
        return "timestamp,realized,q05,q50", [f"{_ts(i)},0,1,0" for i in range(n)]
    if builder == "lookahead-equal":
        return (
            "timestamp,available_time,realized,forecast_var",
            [f"{_ts(i)},{_ts(i)},1,1" for i in range(n)],
        )
    if builder == "lookahead-later":
        return (
            "timestamp,available_time,realized,forecast_var",
            [f"{_ts(i)},{_ts(i + 1)},1,1" for i in range(n)],
        )
    if builder == "decreasing":
        return "timestamp,realized,forecast_var", [f"{_ts(n - 1 - i)},1,1" for i in range(n)]
    if builder == "duplicate-time":
        rows = [f"{_ts(0)},1,1", f"{_ts(0)},1,1"]
        rows.extend(f"{_ts(i)},1,1" for i in range(2, n))
        return "timestamp,realized,forecast_var", rows
    if builder == "missing-column":
        return "timestamp,realized", [f"{_ts(i)},1" for i in range(n)]
    if builder == "prob-range":
        return "timestamp,realized,forecast_prob", [f"{_ts(i)},1,1.5" for i in range(n)]
    if builder == "prob-outcome":
        return "timestamp,realized,forecast_prob", [f"{_ts(i)},2,0.4" for i in range(n)]
    if builder == "no-quantiles":
        return "timestamp,realized", [f"{_ts(i)},0" for i in range(n)]
    if builder in {
        "bad-task",
        "bad-version",
        "bad-seed",
        "quantile-on-vol",
        "forecast-on-quantile",
    }:
        return "timestamp,realized,forecast_var,q05,q50", [f"{_ts(i)},0,1,0,1" for i in range(n)]
    return "", []


def _invoke_failure(path: Path, out: Path, builder: str) -> None:
    if builder == "bad-task":
        score_forecasts(path, task="VOLATILITY", out_dir=out)
        return
    if builder == "bad-version":
        score_forecasts(path, task="volatility", out_dir=out, receipt_version=1)
        return
    if builder == "bad-seed":
        score_forecasts(path, task="volatility", out_dir=out, seed=-1)
        return
    if builder == "quantile-on-vol":
        score_forecasts(path, task="volatility", out_dir=out, quantile_cols="q05=q05")
        return
    if builder == "forecast-on-quantile":
        score_forecasts(path, task="quantile", out_dir=out, forecast_col="forecast_var")
        return
    if builder in {"lookahead-equal", "lookahead-later"}:
        score_forecasts(path, task="volatility", out_dir=out, available_time_col="available_time")
        return
    if builder in {"crossing", "no-quantiles"}:
        score_forecasts(path, task="quantile", out_dir=out)
        return
    if builder in {"prob-range", "prob-outcome"}:
        score_forecasts(path, task="probability", out_dir=out)
        return
    if builder == "corrupt-parquet":
        inspect_dataset(path)
        return
    score_forecasts(path, task="volatility", out_dir=out)
