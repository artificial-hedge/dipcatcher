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
        n_chunks=8,
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
