"""Pins for the tournament → verdict runner."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
import scipy.stats as st  # type: ignore[import-untyped]

from quant_fund.research.fleet_eval import SyntheticShard
from quant_fund.research.verdict_run import run_verdict, verdict_streams

TAUS = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)


class _GaussianFactory:
    def __init__(self, scale: float) -> None:
        self._scale = scale

    def __call__(self) -> object:
        scale = self._scale

        class _Model:
            fleet_lagged_predict = False

            def fit(self, x: np.ndarray, y: np.ndarray) -> None:
                pass

            def predict(self, x: np.ndarray) -> np.ndarray:
                return np.tile(st.norm.ppf(TAUS, scale=scale), (x.shape[0], 1))

        return _Model()


def _gauss_shard(n: int, seed: int) -> SyntheticShard:
    rng = np.random.default_rng(seed)
    return SyntheticShard(
        name="g",
        x=np.zeros((n, 1)),
        y=rng.normal(0.0, 1.0, n),
        config={"data_label": "SYNTHETIC"},
    )


class _BrokenFactory:
    def __call__(self) -> object:
        class _Model:
            fleet_lagged_predict = False

            def fit(self, x: np.ndarray, y: np.ndarray) -> None:
                raise RuntimeError("cannot fit")

        return _Model()


def test_streams_paired_and_aligned() -> None:
    factories = {"tight": _GaussianFactory(0.9), "wide": _GaussianFactory(1.4)}
    scores, pits, status = verdict_streams(
        factories, {"g": _gauss_shard}, n_train=64, n_eval=128, seed=0
    )
    assert set(scores) == {"tight", "wide"}
    assert scores["tight"].shape == scores["wide"].shape == (128,)
    assert pits["tight"].shape == (128,)
    assert status.height == 2
    assert status["status"].to_list() == ["ok", "ok"]
    # tight should beat wide on a N(0,1) shard (its 0.9-scale quantiles
    # sit closer to realized mass than the over-dispersed 1.4 grid)
    assert scores["tight"].mean() < scores["wide"].mean()


def test_broken_head_recorded_not_dropped_silently() -> None:
    factories = {"ok": _GaussianFactory(1.0), "broken": _BrokenFactory()}
    scores, _, status = verdict_streams(
        factories, {"g": _gauss_shard}, n_train=64, n_eval=64, seed=0
    )
    assert set(scores) == {"ok"}
    broken = status.filter(pl.col("head") == "broken")
    assert broken["status"].to_list() == ["error"]


def test_run_verdict_seals_and_excludes() -> None:
    factories = {"cal": _GaussianFactory(1.0), "wide": _GaussianFactory(2.0)}
    verdict, status = run_verdict(
        factories,
        {"g": _gauss_shard},
        n_train=64,
        n_eval=200,
        seed=0,
        n_boot=200,
    )
    assert verdict["data_label"] == "SYNTHETIC"
    assert verdict["research_only"] is True
    assert verdict["live_pnl_claim"] is False
    assert verdict["verdict"] in {
        "confirmed",
        "supported_with_caveats",
        "not_supported",
        "inconclusive",
    }
    assert verdict["run"]["schema"] == "honest_verdict_run.v1"
    assert verdict["run"]["params"]["heads"] == ["cal", "wide"]


def test_run_verdict_fails_closed_no_heads() -> None:
    with pytest.raises(ValueError):
        run_verdict(
            {"broken": _BrokenFactory()},
            {"g": _gauss_shard},
            n_train=64,
            n_eval=64,
            seed=0,
            n_boot=50,
        )


def test_run_verdict_propagates_shard_label() -> None:
    from quant_fund.research.fleet_eval import SyntheticShard

    def real_shard(n: int, seed: int) -> SyntheticShard:
        rng = np.random.default_rng(seed)
        return SyntheticShard(
            name="r",
            x=np.zeros((n, 1)),
            y=rng.normal(0.0, 1.0, n),
            config={"data_label": "yahoo_eod"},
        )

    verdict, _ = run_verdict(
        {"a": _GaussianFactory(1.0)},
        {"r": real_shard},
        n_train=64,
        n_eval=64,
        n_boot=50,
    )
    assert verdict["data_label"] == "yahoo_eod"
    assert verdict["run"]["params"]["data_labels"] == {"r": "yahoo_eod"}


def test_run_verdict_refuses_mixed_labels() -> None:
    import pytest

    from quant_fund.research.fleet_eval import SyntheticShard

    def real_shard(n: int, seed: int) -> SyntheticShard:
        rng = np.random.default_rng(seed)
        return SyntheticShard(
            name="r",
            x=np.zeros((n, 1)),
            y=rng.normal(0.0, 1.0, n),
            config={"data_label": "yahoo_eod"},
        )

    with pytest.raises(ValueError, match="mixed data_label"):
        run_verdict(
            {"a": _GaussianFactory(1.0)},
            {"s": _gauss_shard, "r": real_shard},
            n_train=64,
            n_eval=64,
            n_boot=50,
        )
