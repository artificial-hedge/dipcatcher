"""fleet_race: sequential elimination over ordered eval chunks."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.research.fleet_race import fleet_race


class _ConstHead:
    """Emits a fixed quantile grid shifted by `bias` — biased heads lose."""

    def __init__(self, bias: float, n_taus: int = 7) -> None:
        self.bias = bias
        self.n_taus = n_taus

    def fit(self, x, y, **kwargs):
        return self

    def predict(self, x):
        n = np.asarray(x).shape[0]
        grid = np.linspace(-0.01, 0.01, self.n_taus)
        return np.tile(self.bias + grid, (n, 1))


class _BrokenHead:
    def fit(self, x, y, **kwargs):
        raise RuntimeError("fit exploded")


def _factories(biases: dict[str, float]) -> dict:
    return {name: (lambda b=bias: _ConstHead(b)) for name, bias in biases.items()}


def test_good_head_promotes_bad_heads_eliminated() -> None:
    factories = _factories({"oracle": 0.0, "lagged": 0.05, "wild": -0.08})
    frame, receipt = fleet_race(
        factories,
        shards={"iid_gaussian": SHARD_G},
        n_train=64,
        n_eval=32,
        n_chunks=16,
        seed=0,
    )
    assert receipt["kind"] == "fleet_race.v1"
    assert receipt["data_label"] == "SYNTHETIC"
    ok = frame.filter(frame["status"] == "ok")
    assert ok["shard_winner"].sum() == 1
    winner_row = ok.filter(ok["shard_winner"]).row(0, named=True)
    assert winner_row["model"] == "oracle"
    assert winner_row["verdict"] in {"anytime", "final_mean"}
    eliminated = ok.filter(ok["eliminated_at"].is_not_null())
    assert eliminated.height >= 1


def test_error_head_recorded_not_crash() -> None:
    factories = {"broken": lambda: _BrokenHead(), "fine": lambda: _ConstHead(0.0)}
    frame, _ = fleet_race(
        factories,
        shards={"iid_gaussian": SHARD_G},
        n_train=64,
        n_eval=32,
        n_chunks=8,
    )
    err = frame.filter(frame["status"] == "error")
    assert err["model"].to_list() == ["broken"]
    assert "fit exploded" in err["error"].to_list()[0]


def test_determinism_and_causality() -> None:
    factories = _factories({"a": 0.0, "b": 0.03})
    f1, r1 = fleet_race(
        factories,
        shards={"iid_gaussian": SHARD_G},
        n_train=64,
        n_eval=32,
        n_chunks=8,
        seed=1,
    )
    f2, r2 = fleet_race(
        factories,
        shards={"iid_gaussian": SHARD_G},
        n_train=64,
        n_eval=32,
        n_chunks=8,
        seed=1,
    )
    assert f1.equals(f2)
    assert r1["inputs_sha256"] == r2["inputs_sha256"]


def test_fails_closed_on_bad_args() -> None:
    factories = _factories({"a": 0.0})
    with pytest.raises(ValueError, match="n_chunks"):
        fleet_race(
            factories,
            shards={"iid_gaussian": SHARD_G},
            n_train=64,
            n_eval=33,
            n_chunks=8,
        )
    with pytest.raises(ValueError, match="alpha"):
        fleet_race(
            factories,
            shards={"iid_gaussian": SHARD_G},
            n_train=64,
            n_eval=32,
            n_chunks=8,
            alpha=2.0,
        )
    with pytest.raises(ValueError, match="nonempty"):
        fleet_race({}, shards={"iid_gaussian": SHARD_G}, n_train=64, n_eval=32, n_chunks=8)


def _make_shard(name: str):
    from quant_fund.research.fleet_eval import SyntheticShard

    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1.0, size=(96, 4))
    y = rng.normal(0.0, 0.01, size=96)
    return SyntheticShard(name, x, y, {"data_label": "SYNTHETIC", "n": 96})


SHARD_G = lambda n, seed: _make_shard("iid_gaussian")  # noqa: E731


SHARD_H = lambda n, seed: _make_shard("heavy_tail")  # noqa: E731


def test_global_evidence_product_pooling() -> None:
    """Cross-shard e-value pooling: promote/demote products coherent."""
    factories = _factories({"oracle": 0.0, "lagged": 0.05, "wild": -0.08})
    shards = {"iid_gaussian": SHARD_G, "heavy_tail": SHARD_H}
    _frame, receipt = fleet_race(
        factories,
        shards=shards,
        n_train=64,
        n_eval=32,
        n_chunks=8,
        seed=0,
    )
    ge = receipt["global_evidence"]
    assert isinstance(ge, dict)
    assert set(ge) == {"oracle", "lagged", "wild"}
    for entry in ge.values():
        assert entry["n_shards"] == len(shards)
        assert entry["promote_evalue_product"] > 0
        assert entry["demote_evalue_product"] > 0
        assert entry["global_promotion"] == (
            entry["promote_evalue_product"] >= 20.0  # 1/0.05
        )
    # oracle is never worse than any head -> its demote product should be 1
    assert ge["oracle"]["demote_evalue_product"] <= 1.0 + 1e-9


def test_short_shard_fails_closed() -> None:
    """A shard shorter than n_train + n_eval raises before any racing."""
    with pytest.raises(ValueError, match="produced .* rows"):
        fleet_race(
            _factories({"oracle": 0.0}),
            shards={"iid_gaussian": SHARD_G},
            n_train=64,
            n_eval=64,  # needs 128 rows; SHARD_G yields 96
            n_chunks=16,
            seed=0,
        )


def test_race_data_label_derived_and_mixed_refused() -> None:
    """Receipt stamps the shards' own label; mixed corpora fail closed."""
    from quant_fund.research.fleet_eval import SyntheticShard

    def real_shard(n: int, seed: int) -> SyntheticShard:
        rng = np.random.default_rng(seed)
        return SyntheticShard(
            "r", np.zeros((n, 1)), rng.normal(0.0, 0.01, n), {"data_label": "yahoo_eod"}
        )

    _frame, receipt = fleet_race(
        _factories({"a": 0.0}),
        shards={"r": real_shard},
        n_train=64,
        n_eval=32,
        n_chunks=8,
    )
    assert receipt["data_label"] == "yahoo_eod"
    assert receipt["shard_meta"]["r"]["data_label"] == "yahoo_eod"

    import pytest

    with pytest.raises(ValueError, match="mixed data_label"):
        fleet_race(
            _factories({"a": 0.0}),
            shards={"s": SHARD_G, "r": real_shard},
            n_train=64,
            n_eval=32,
            n_chunks=8,
        )


def test_race_receipt_v2_round_trip(tmp_path) -> None:
    """receipt_version=2 seals the fleet_race.v1 body in the envelope."""
    import json
    from pathlib import Path

    from quant_fund.research.fleet_race import write_race_receipt
    from quant_fund.research.receipt_v2 import verify_receipt_file

    assert isinstance(tmp_path, Path)
    factories = _factories({"oracle": 0.0, "lagged": 0.05})
    _, receipt = fleet_race(
        factories,
        shards={"iid_gaussian": SHARD_G},
        n_train=64,
        n_eval=32,
        n_chunks=16,
        seed=0,
    )
    path = write_race_receipt(receipt, tmp_path, receipt_version=2)
    payload = json.loads(path.read_text())
    assert payload["schema"] == "receipt.v2"
    assert payload["payload"]["kind"] == "fleet_race.v1"
    assert payload["payload"]["inputs_sha256"] == receipt["inputs_sha256"]
    assert verify_receipt_file(path)["valid"] is True


class _CrossingHead:
    """Emits a quantile grid whose columns cross — must be refused, not scored."""

    def fit(self, x, y, **kwargs):
        return self

    def predict(self, x):
        n = np.asarray(x).shape[0]
        grid = np.linspace(-0.01, 0.01, 7)
        frame = np.tile(grid, (n, 1))
        frame[:, 2] = frame[:, 6]  # q25 < q50 would invert here
        return frame


class _FlipHead:
    """SYNTHETIC: optimal on chunks 1-5, badly biased after — promotes, then
    is eliminated by the mediocre incumbent it briefly beat."""

    def __init__(self, n_eval: int) -> None:
        self.n_eval = n_eval

    def fit(self, x, y, **kwargs):
        return self

    def predict(self, x):
        n = np.asarray(x).shape[0]
        bias = np.empty(n, dtype=float)
        chunk = n // 20  # chunk length when n_chunks=20
        bias[:chunk] = 0.01  # slightly off — loses the chunk-0 argmin to drifter
        bias[chunk : chunk * 6] = 0.0  # chunks 1-5: optimal — promotes
        bias[chunk * 6 :] = 0.4  # chunks 6+: badly biased — eliminated
        grid = np.linspace(-0.01, 0.01, 7)
        return bias[:, None] + np.tile(grid, (n, 1))


class _DrifterHead:
    """SYNTHETIC: perfect on chunk 0 (wins incumbent), mediocre after — loses
    chunks 1-5 to flipper, then beats it again on the late chunks."""

    def fit(self, x, y, **kwargs):
        return self

    def predict(self, x):
        n = np.asarray(x).shape[0]
        chunk = n // 20
        bias = np.full(n, 0.05, dtype=float)
        bias[:chunk] = 0.0
        grid = np.linspace(-0.01, 0.01, 7)
        return bias[:, None] + np.tile(grid, (n, 1))


def test_crossing_quantile_frame_recorded_as_error() -> None:
    factories = {"crossed": lambda: _CrossingHead(), "fine": lambda: _ConstHead(0.0)}
    frame, receipt = fleet_race(
        factories,
        shards={"iid_gaussian": SHARD_G},
        n_train=64,
        n_eval=32,
        n_chunks=8,
        seed=0,
    )
    bad = frame.filter(frame["model"] == "crossed").row(0, named=True)
    assert bad["status"] == "error"
    assert "invalid quantile frame" in bad["error"]
    assert receipt["shard_winners"] == {"iid_gaussian": "fine"}


def test_promoted_then_eliminated_head_cannot_win() -> None:
    """SYNTHETIC: contradictory evidence excludes a head from the winner pool."""
    n_eval = 80  # SHARD_G is a fixed 96-row fixture -> n_train + n_eval = 96
    factories = {
        "drifter": lambda: _DrifterHead(),
        "flipper": lambda: _FlipHead(n_eval),
    }
    frame, receipt = fleet_race(
        factories,
        shards={"iid_gaussian": SHARD_G},
        n_train=16,
        n_eval=n_eval,
        n_chunks=20,
        alpha=0.5,
        seed=0,
    )
    flip = frame.filter(frame["model"] == "flipper").row(0, named=True)
    assert flip["promoted_at"] is not None
    assert flip["eliminated_at"] is not None
    assert not flip["shard_winner"]
    assert receipt["shard_winners"] == {"iid_gaussian": "drifter"}
    winner = frame.filter(frame["shard_winner"]).row(0, named=True)
    assert winner["verdict"] == "final_mean"


def test_race_receipt_v2_fails_when_no_head_survives(tmp_path) -> None:
    """SYNTHETIC: a v2 race with no shard winner seals verdict=fail."""
    import json

    from quant_fund.research.fleet_race import write_race_receipt

    factories = {"broken": lambda: _BrokenHead()}
    _, receipt = fleet_race(
        factories,
        shards={"iid_gaussian": SHARD_G},
        n_train=64,
        n_eval=32,
        n_chunks=8,
        seed=0,
    )
    assert receipt["shard_winners"] == {}
    path = write_race_receipt(receipt, tmp_path, receipt_version=2)
    payload = json.loads(path.read_text())
    assert payload["verdict"] == "fail"
