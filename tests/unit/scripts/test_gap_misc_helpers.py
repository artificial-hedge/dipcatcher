"""Tests for assorted standalone ``scripts/`` helpers with no prior coverage.

Covers four scripts whose logic is pure and worth pinning:

- ``scripts/mutation_score.py`` — target selection and the mutmut score
  formula.
- ``scripts/splice_timesfm_fix.py`` — shard hashing, npz load/validate, and
  the pair splice contract.
- ``scripts/screen_basis_reversion.py`` — file hashing.
- ``scripts/c5_ml_carry.py`` — the ML funding feature builder and the
  prediction -> weight mapping.
- ``scripts/fx1_tiny_lm_train.py`` — seed-corpus text extraction.

Everything runs against ``tmp_path`` fixtures. No script is executed, no
mutmut run is launched, and no repository artifact is written.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

_SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


def _load(name: str):
    import importlib

    return importlib.import_module(name)


mutation_score = _load("mutation_score")
splice = _load("splice_timesfm_fix")
screen = _load("screen_basis_reversion")
c5 = _load("c5_ml_carry")
tiny_lm = _load("fx1_tiny_lm_train")


# --------------------------------------------------------------------------- #
# mutation_score._select_targets
# --------------------------------------------------------------------------- #


def test_select_targets_without_args_returns_all() -> None:
    assert mutation_score._select_targets([]) == mutation_score.TARGETS


def test_select_targets_filters_by_stem() -> None:
    chosen = mutation_score._select_targets(["costs.py"])
    assert len(chosen) == 1
    assert Path(str(chosen[0]["path"])).stem == "costs"


def test_select_targets_accepts_stem_without_suffix() -> None:
    assert mutation_score._select_targets(["costs"]) == mutation_score._select_targets(["costs.py"])


def test_select_targets_unknown_stem_exits() -> None:
    with pytest.raises(SystemExit, match="no TARGETS entries"):
        mutation_score._select_targets(["definitely_not_a_target"])


def test_select_targets_preserves_catalog_order() -> None:
    stems = [Path(str(t["path"])).stem for t in mutation_score.TARGETS]
    chosen = [Path(str(t["path"])).stem for t in mutation_score._select_targets(stems)]
    assert chosen == stems


def test_select_targets_entries_have_path_and_tests() -> None:
    for target in mutation_score.TARGETS:
        assert "path" in target
        assert "tests" in target
        assert isinstance(target["tests"], list)
        assert target["tests"]


# --------------------------------------------------------------------------- #
# mutation_score._score
# --------------------------------------------------------------------------- #


def test_score_is_killed_over_tested() -> None:
    # (killed + timeout) / (total - skipped) -> 50 / 100 = 50%
    assert mutation_score._score({"total": 100, "skipped": 0, "killed": 50}) == 50.0


def test_score_counts_timeouts_as_kills() -> None:
    stats = {"total": 100, "skipped": 0, "killed": 40, "timeout": 10}
    assert mutation_score._score(stats) == 50.0


def test_score_excludes_skipped_from_denominator() -> None:
    stats = {"total": 100, "skipped": 50, "killed": 50}
    assert mutation_score._score(stats) == 100.0


def test_score_returns_none_when_nothing_tested() -> None:
    assert mutation_score._score({"total": 10, "skipped": 10, "killed": 0}) is None


def test_score_returns_none_on_empty_stats() -> None:
    assert mutation_score._score({}) is None


def test_score_zero_kills_is_zero_not_none() -> None:
    assert mutation_score._score({"total": 10, "skipped": 0, "killed": 0}) == 0.0


# --------------------------------------------------------------------------- #
# mutation_score._setup_cfg
# --------------------------------------------------------------------------- #


def test_setup_cfg_lists_source_and_tests() -> None:
    cfg = mutation_score._setup_cfg("src/foo.py", ["tests/unit/test_foo.py"])
    assert "[mutmut]" in cfg
    assert "src/foo.py" in cfg
    assert "tests/unit/test_foo.py" in cfg


def test_setup_cfg_indents_each_test_path() -> None:
    cfg = mutation_score._setup_cfg("src/foo.py", ["tests/a.py", "tests/b.py"])
    assert "    tests/a.py" in cfg
    assert "    tests/b.py" in cfg


def test_setup_cfg_disables_git_change_detection() -> None:
    cfg = mutation_score._setup_cfg("src/foo.py", [])
    assert "use_git_change_detection = false" in cfg


# --------------------------------------------------------------------------- #
# _sha256 helpers
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("mod", [splice, screen], ids=lambda m: m.__name__)
def test_sha256_matches_hashlib(mod, tmp_path: Path) -> None:
    blob = b"research evidence bytes\n" * 1000
    target = tmp_path / "artifact.bin"
    target.write_bytes(blob)
    assert mod._sha256(target) == hashlib.sha256(blob).hexdigest()


@pytest.mark.parametrize("mod", [splice, screen], ids=lambda m: m.__name__)
def test_sha256_of_empty_file(mod, tmp_path: Path) -> None:
    target = tmp_path / "empty.bin"
    target.write_bytes(b"")
    assert mod._sha256(target) == hashlib.sha256(b"").hexdigest()


@pytest.mark.parametrize("mod", [splice, screen], ids=lambda m: m.__name__)
def test_sha256_changes_with_content(mod, tmp_path: Path) -> None:
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    a.write_bytes(b"a")
    b.write_bytes(b"b")
    assert mod._sha256(a) != mod._sha256(b)


@pytest.mark.parametrize("mod", [splice, screen], ids=lambda m: m.__name__)
def test_sha256_is_stable_across_calls(mod, tmp_path: Path) -> None:
    target = tmp_path / "artifact.bin"
    target.write_bytes(b"stable")
    assert mod._sha256(target) == mod._sha256(target)


# --------------------------------------------------------------------------- #
# splice_timesfm_fix.load / save / splice_pair
# --------------------------------------------------------------------------- #


def _valid_shard(
    path: Path,
    *,
    names: list[str] | None = None,
    taus: list[float] | None = None,
    meta_extra: dict | None = None,
    config_extra: dict | None = None,
    timesfm_offset: float = 0.0,
) -> Path:
    names = names if names is not None else ["baseline", "timesfm"]
    taus = taus if taus is not None else [0.1, 0.5, 0.9]
    n = len(names)
    config = {
        "origins_per_asset": 3,
        "samples_per_origin": 2,
        "lookback": 64,
        "window": 32,
        "garch_window": 32,
        "seed": 0,
        "taus": taus,
    }
    if config_extra:
        config.update(config_extra)
    meta = {
        "config": config,
        "asset_names": ["SPY", "QQQ"],
        "bars_sha256": "0" * 64,
        "artifact_sha256": {"timesfm": "a" * 64},
    }
    if meta_extra:
        meta.update(meta_extra)
    crps = np.zeros((2, n), dtype=np.float64)
    if timesfm_offset and "timesfm" in names:
        crps[:, names.index("timesfm")] = timesfm_offset
    pin = np.zeros((2, n, len(taus)), dtype=np.float64)
    if timesfm_offset and "timesfm" in names:
        pin[:, names.index("timesfm"), :] = timesfm_offset
    np.savez_compressed(
        path,
        crps_matrix=crps,
        pinball_cube=pin,
        asset_ids=np.array(["SPY", "QQQ"]),
        model_names=np.asarray(names),
        meta_json=np.array(json.dumps(meta)),
    )
    return path


def test_load_returns_all_five_arrays(tmp_path: Path) -> None:
    path = _valid_shard(tmp_path / "shard.npz")
    data = splice.load(path)
    assert set(data) == {"crps", "pin", "aids", "names", "meta"}
    assert data["names"] == ["baseline", "timesfm"]
    assert data["crps"].shape == (2, 2)
    assert data["pin"].shape == (2, 2, 3)


def test_load_rejects_mismatched_crps_shape(tmp_path: Path) -> None:
    path = tmp_path / "bad.npz"
    np.savez_compressed(
        path,
        crps_matrix=np.zeros((2, 5), dtype=np.float64),
        pinball_cube=np.zeros((2, 5, 3), dtype=np.float64),
        asset_ids=np.array(["SPY"]),
        model_names=np.asarray(["baseline", "timesfm"]),
        meta_json=np.array(
            json.dumps({"config": {"taus": [0.1, 0.5, 0.9]}, "asset_names": ["SPY"]})
        ),
    )
    with pytest.raises(ValueError, match="invalid CRPS matrix shape"):
        splice.load(path)


def test_load_rejects_duplicate_model_names(tmp_path: Path) -> None:
    path = _valid_shard(tmp_path / "dup.npz", names=["timesfm", "timesfm"])
    with pytest.raises(ValueError, match="nonempty and unique"):
        splice.load(path)


def test_load_rejects_missing_target_column(tmp_path: Path) -> None:
    path = _valid_shard(tmp_path / "notarget.npz", names=["baseline", "other"])
    with pytest.raises(ValueError, match="no timesfm column"):
        splice.load(path)


def test_load_rejects_missing_metadata(tmp_path: Path) -> None:
    path = tmp_path / "nometa.npz"
    np.savez_compressed(
        path,
        crps_matrix=np.zeros((2, 2), dtype=np.float64),
        pinball_cube=np.zeros((2, 2, 3), dtype=np.float64),
        asset_ids=np.array(["SPY", "QQQ"]),
        model_names=np.asarray(["baseline", "timesfm"]),
        meta_json=np.array(json.dumps({"asset_names": ["SPY"]})),
    )
    with pytest.raises(ValueError, match="missing evaluation metadata/config"):
        splice.load(path)


def test_load_rejects_missing_asset_names(tmp_path: Path) -> None:
    path = tmp_path / "noassets.npz"
    meta = {"config": {"taus": [0.1, 0.5, 0.9]}}
    np.savez_compressed(
        path,
        crps_matrix=np.zeros((2, 2), dtype=np.float64),
        pinball_cube=np.zeros((2, 2, 3), dtype=np.float64),
        asset_ids=np.array(["SPY", "QQQ"]),
        model_names=np.asarray(["baseline", "timesfm"]),
        meta_json=np.array(json.dumps(meta)),
    )
    with pytest.raises(ValueError, match="missing asset names"):
        splice.load(path)


def test_load_rejects_pinball_shape_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "badpin.npz"
    meta = {
        "config": {"taus": [0.1, 0.5, 0.9]},
        "asset_names": ["SPY", "QQQ"],
    }
    np.savez_compressed(
        path,
        crps_matrix=np.zeros((2, 2), dtype=np.float64),
        pinball_cube=np.zeros((2, 2, 2), dtype=np.float64),
        asset_ids=np.array(["SPY", "QQQ"]),
        model_names=np.asarray(["baseline", "timesfm"]),
        meta_json=np.array(json.dumps(meta)),
    )
    with pytest.raises(ValueError, match="pinball shape does not match"):
        splice.load(path)


def test_save_round_trips_through_load(tmp_path: Path) -> None:
    source = _valid_shard(tmp_path / "source.npz")
    data = splice.load(source)
    out = tmp_path / "written.npz"
    splice.save(out, data)
    reloaded = splice.load(out)
    assert reloaded["names"] == data["names"]
    assert np.array_equal(reloaded["crps"], data["crps"])
    assert reloaded["meta"]["asset_names"] == data["meta"]["asset_names"]


def test_splice_pair_returns_the_original_shard(tmp_path: Path) -> None:
    original = _valid_shard(tmp_path / "original.npz")
    corrected = _valid_shard(tmp_path / "corrected.npz", timesfm_offset=0.5)
    out = splice.splice_pair(original, corrected)
    assert out["names"] == ["baseline", "timesfm"]
    # The corrected TimesFM column is grafted onto the original shard.
    assert out["crps"][:, out["names"].index("timesfm")] == pytest.approx(0.5)


def test_splice_pair_writes_nothing(tmp_path: Path) -> None:
    original = _valid_shard(tmp_path / "original.npz")
    corrected = _valid_shard(tmp_path / "corrected.npz", timesfm_offset=0.5)
    before = sorted(p.name for p in tmp_path.iterdir())
    splice.splice_pair(original, corrected)
    assert sorted(p.name for p in tmp_path.iterdir()) == before


def test_splice_pair_rejects_inconsistent_protocol(tmp_path: Path) -> None:
    original = _valid_shard(tmp_path / "original.npz")
    corrected = _valid_shard(
        tmp_path / "corrected.npz",
        config_extra={"seed": 1},
        timesfm_offset=0.5,
    )
    with pytest.raises(ValueError, match="protocol mismatch"):
        splice.splice_pair(original, corrected)


def test_splice_pair_rejects_bars_provenance_mismatch(tmp_path: Path) -> None:
    original = _valid_shard(tmp_path / "original.npz")
    corrected = _valid_shard(
        tmp_path / "corrected.npz",
        meta_extra={"bars_sha256": "1" * 64},
        timesfm_offset=0.5,
    )
    with pytest.raises(ValueError, match="bars provenance mismatch"):
        splice.splice_pair(original, corrected)


def test_splice_pair_rejects_an_identical_corrected_column(tmp_path: Path) -> None:
    original = _valid_shard(tmp_path / "original.npz")
    corrected = _valid_shard(tmp_path / "corrected.npz")
    with pytest.raises(ValueError, match="nothing to fix"):
        splice.splice_pair(original, corrected)


def test_splice_pair_records_the_transformation(tmp_path: Path) -> None:
    original = _valid_shard(tmp_path / "original.npz")
    corrected = _valid_shard(tmp_path / "corrected.npz", timesfm_offset=0.5)
    out = splice.splice_pair(original, corrected)
    transformations = out["meta"]["transformations"]
    assert transformations[-1]["operation"] == "replace_model_forecasts"
    assert transformations[-1]["model"] == "timesfm"
    # The splice records the source-file hashes of both inputs.
    inputs = transformations[-1]["inputs"]
    assert inputs["original"]["sha256"] == splice._sha256(original)
    assert inputs["corrected"]["sha256"] == splice._sha256(corrected)


def test_splice_pair_marks_the_scoring_contract(tmp_path: Path) -> None:
    original = _valid_shard(tmp_path / "original.npz")
    corrected = _valid_shard(tmp_path / "corrected.npz", timesfm_offset=0.5)
    out = splice.splice_pair(original, corrected)
    assert out["meta"]["scoring_contract"] == "native_shapes_timesfm_point_first.v2+spliced"
    assert "timesfm_corrected" in out["meta"]


def test_target_constant_is_timesfm() -> None:
    assert splice.TARGET == "timesfm"


# --------------------------------------------------------------------------- #
# c5_ml_carry._features / _ml_weights
# --------------------------------------------------------------------------- #


def _funding(n: int = 60, sid: str = "BTC") -> pl.DataFrame:
    base = datetime(2024, 6, 1, tzinfo=UTC)
    return pl.DataFrame(
        [
            {
                "event_time": base + timedelta(hours=i),
                "security_id": sid,
                "value": 0.001 * (i % 4),
            }
            for i in range(n)
        ]
    )


def _bars(n: int = 60, sid: str = "BTC") -> pl.DataFrame:
    base = datetime(2024, 6, 1, tzinfo=UTC)
    return pl.DataFrame(
        [
            {
                "event_time": base + timedelta(hours=i),
                "security_id": sid,
                "close": 100.0 + i * 0.1,
            }
            for i in range(n)
        ]
    )


def test_c5_features_produce_expected_columns() -> None:
    out = c5._features(_bars(), _funding())
    for col in ("f_lag1", "f_ma3", "f_ma9", "f_sd9", "target", "ret24", "ret168", "vol168"):
        assert col in out.columns


def test_c5_features_drop_first_row_per_sid() -> None:
    out = c5._features(_bars(), _funding(60))
    assert out.height == 59
    assert out["f_lag1"].null_count() == 0


def test_c5_features_lag_is_previous_value() -> None:
    out = c5._features(_bars(10), _funding(10))
    assert out["f_lag1"].to_list()[0] == pytest.approx(0.0)
    assert out["f_lag1"].to_list()[1] == pytest.approx(0.001)


def test_c5_features_target_is_next_value() -> None:
    out = c5._features(_bars(10), _funding(10))
    assert out["target"].to_list()[0] == pytest.approx(0.002)


def test_c5_features_constant_columns_are_declared() -> None:
    assert set(c5.FEATURES).issubset(set(c5._features(_bars(), _funding()).columns))


def _c5_bars(n: int = 60, sids: tuple[str, ...] = ("BTC", "ETH")) -> pl.DataFrame:
    base = datetime(2024, 6, 1, tzinfo=UTC)
    return pl.DataFrame(
        [
            {
                "event_time": base + timedelta(hours=i),
                "security_id": sid,
                "close": 100.0 + i * 0.1 + j,
            }
            for j, sid in enumerate(sids)
            for i in range(n)
        ]
    )


def _c5_preds(n: int = 60, sids: tuple[str, ...] = ("BTC", "ETH")) -> pl.DataFrame:
    base = datetime(2024, 6, 1, tzinfo=UTC)
    return pl.DataFrame(
        [
            {
                "event_time": base + timedelta(hours=i),
                "security_id": sid,
                "pred": 0.001 * (j + 1),
            }
            for j, sid in enumerate(sids)
            for i in range(n)
        ]
    )


def test_c5_ml_weights_emits_weight_rows() -> None:
    bars = _c5_bars(60)
    preds = _c5_preds(60)
    out = c5._ml_weights(bars, preds, max_name=1.0, gross_scale=1.0, vol_window=24)
    assert out.height > 0
    assert out.columns == ["event_time", "security_id", "target_weight"]


def test_c5_ml_weights_are_cross_sectionally_demeaned() -> None:
    """Raw weights are de-meaned across the cross-section before capping.

    Two names with different predictions therefore get opposite-signed
    weights rather than both carrying the full raw signal.
    """
    bars = _c5_bars(60)
    preds = _c5_preds(60)
    out = c5._ml_weights(bars, preds, max_name=1.0, gross_scale=1.0, vol_window=24)
    last = out.filter(pl.col("event_time") == out["event_time"].max())
    assert last.height == 2
    assert last["target_weight"].to_list()[0] == pytest.approx(-last["target_weight"].to_list()[1])


def test_c5_ml_weights_respect_the_gross_cap() -> None:
    bars = _c5_bars(60)
    preds = _c5_preds(60)
    out = c5._ml_weights(bars, preds, max_name=1.0, gross_scale=0.5, vol_window=24)
    gross = out.group_by("event_time").agg(pl.col("target_weight").abs().sum().alias("g"))
    assert gross["g"].max() <= 0.5 + 1e-9


def test_c5_ml_weights_does_not_mutate_inputs() -> None:
    bars = _c5_bars(60)
    preds = _c5_preds(60)
    before_bars = bars.clone()
    before_preds = preds.clone()
    c5._ml_weights(bars, preds, max_name=1.0, gross_scale=1.0, vol_window=24)
    assert bars.equals(before_bars)
    assert preds.equals(before_preds)


# --------------------------------------------------------------------------- #
# fx1_tiny_lm_train._corpus_text
# --------------------------------------------------------------------------- #


def test_corpus_text_joins_message_content(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    # The trainer refuses a corpus below MIN_CORPUS_BYTES, so pad the content.
    corpus.write_text(
        "\n".join(
            [
                json.dumps({"messages": [{"role": "user", "content": "hello " + "a" * 400}]}),
                json.dumps({"messages": [{"role": "assistant", "content": "world " + "b" * 400}]}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkey = pytest.MonkeyPatch()
    monkey.setattr(tiny_lm, "_CORPUS_FILE", "corpus.jsonl")
    try:
        text = tiny_lm._corpus_text(tmp_path)
    finally:
        monkey.undo()
    assert text.splitlines() == ["hello " + "a" * 400, "world " + "b" * 400]


def test_corpus_text_skips_blank_lines(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        "\n" + json.dumps({"messages": [{"role": "user", "content": "x" * 600}]}) + "\n\n",
        encoding="utf-8",
    )
    monkey = pytest.MonkeyPatch()
    monkey.setattr(tiny_lm, "_CORPUS_FILE", "corpus.jsonl")
    try:
        text = tiny_lm._corpus_text(tmp_path)
    finally:
        monkey.undo()
    assert text == "x" * 600


def test_corpus_text_rejects_malformed_json(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("{not json\n", encoding="utf-8")
    monkey = pytest.MonkeyPatch()
    monkey.setattr(tiny_lm, "_CORPUS_FILE", "corpus.jsonl")
    try:
        with pytest.raises(ValueError, match="not JSON"):
            tiny_lm._corpus_text(tmp_path)
    finally:
        monkey.undo()


def test_corpus_text_rejects_row_without_messages(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(json.dumps({"nope": 1}) + "\n", encoding="utf-8")
    monkey = pytest.MonkeyPatch()
    monkey.setattr(tiny_lm, "_CORPUS_FILE", "corpus.jsonl")
    try:
        with pytest.raises(ValueError, match="no messages list"):
            tiny_lm._corpus_text(tmp_path)
    finally:
        monkey.undo()


def test_corpus_text_rejects_a_corpus_that_is_too_small(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        json.dumps({"messages": [{"role": "user", "content": "x"}]}) + "\n",
        encoding="utf-8",
    )
    monkey = pytest.MonkeyPatch()
    monkey.setattr(tiny_lm, "_CORPUS_FILE", "corpus.jsonl")
    try:
        with pytest.raises(ValueError, match="too small to train"):
            tiny_lm._corpus_text(tmp_path)
    finally:
        monkey.undo()


def test_corpus_file_constant_points_at_the_seed_corpus() -> None:
    assert tiny_lm._CORPUS_FILE == "fx1_seed_corpus.jsonl"
