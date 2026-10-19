"""sundial fleet head: dep gate, contract shape, sample-quantile reduction."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from quant_fund.models.sundial import (
    SundialDistribution,
    _samples_to_quantiles,
)


class _FakeSundial:
    """Emits (1, num_samples, pred_len=1) draws centered at loc."""

    def __init__(self, loc: float = 0.01) -> None:
        self.loc = loc
        self.calls = 0

    def generate(self, seqs: Any, *, max_new_tokens: int, num_samples: int) -> np.ndarray:
        self.calls += 1
        samples = self.loc + np.linspace(-0.02, 0.02, num_samples)
        return samples.reshape(1, num_samples, max_new_tokens)


def _fit_stubbed(**kwargs: Any) -> tuple[SundialDistribution, _FakeSundial]:
    pred = _FakeSundial()
    rng = np.random.default_rng(0)
    y = rng.normal(0.0, 0.01, size=300)
    model = SundialDistribution(predictor=pred, **kwargs).fit(np.zeros((300, 2)), y)
    return model, pred


def test_fit_without_dep_fails_closed() -> None:
    model = SundialDistribution()
    with pytest.raises(RuntimeError, match="thuml/Sundial"):
        model.fit(np.zeros((300, 2)), np.zeros(300))


def test_predict_shape_and_monotone_grid() -> None:
    model, _ = _fit_stubbed(lookback=32, num_samples=16)
    q = model.predict(np.zeros((7, 2)))
    assert q.shape == (7, 3)
    assert np.isfinite(q).all()
    assert (np.diff(q, axis=1) > 0).all()


def test_predict_from_history_causal_windows() -> None:
    model, _ = _fit_stubbed(lookback=16, num_samples=8)
    history = np.random.default_rng(1).normal(0.0, 0.01, size=50)
    q = model.predict_from_history(history)
    assert q.shape == (50 - 16 + 1, 3)
    assert np.isfinite(q).all()


def test_rejects_bad_inputs() -> None:
    model, _ = _fit_stubbed(lookback=16)
    with pytest.raises(ValueError, match=">= lookback"):
        model.predict_from_history(np.zeros(4))
    with pytest.raises(RuntimeError, match="not been fitted"):
        SundialDistribution(predictor=_FakeSundial()).predict(np.zeros((2, 2)))


def test_samples_to_quantiles_empirical() -> None:
    taus = np.array([0.1, 0.5, 0.9])
    # (num_samples, pred_len) shaped draws.
    draws = np.linspace(0.0, 1.0, 101).reshape(101, 1)
    q = _samples_to_quantiles(draws, taus)
    np.testing.assert_allclose(q, [0.1, 0.5, 0.9], atol=0.01)
    # Bad shapes fail closed.
    with pytest.raises(ValueError, match="unexpected shape"):
        _samples_to_quantiles(np.zeros((2, 2, 2)), taus)
    with pytest.raises(ValueError, match="non-finite|singleton"):
        _samples_to_quantiles(np.array([[np.nan, 1.0]]), taus)


def test_metadata_flags() -> None:
    model, _ = _fit_stubbed()
    extra = model.metadata().extra
    assert extra["framework"] == "thuml/Sundial"
    assert "uv.lock-incompatible" in extra["weights_note"]
    assert "trust_remote_code" in extra["weights_note"]
    assert extra["pretrained"] is True


def test_fleet_registry_includes_sundial() -> None:
    from quant_fund.research.fleet_eval import FLEET_HEAD_REGISTRY

    assert "sundial" in FLEET_HEAD_REGISTRY


# Hub commit resolved via the Hugging Face API on 2026-10-19.
_PINNED = "3212e42564493f520593e5414af4367fc4b49226"


class _RecordingAutoModel:
    """Stands in for AutoModelForCausalLM; records the hub-load kwargs."""

    calls: list[tuple[str, dict[str, Any]]] = []

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs: Any) -> str:
        cls.calls.append((model_id, kwargs))
        return "loaded"


def test_default_hub_load_is_pinned_and_recorded() -> None:
    model = SundialDistribution()
    assert model.revision == _PINNED
    assert model.metadata().extra["model_revision"] == _PINNED


def test_unpinned_model_id_or_moving_branch_refused() -> None:
    with pytest.raises(ValueError, match="no pinned Hub revision"):
        SundialDistribution(model_id="someone/unpinned-fork")
    with pytest.raises(ValueError, match="40-hex"):
        SundialDistribution(revision="main")
    explicit = SundialDistribution(model_id="org/other", revision="a" * 40)
    assert explicit.metadata().extra["model_revision"] == "a" * 40


def test_trust_remote_code_load_carries_pinned_revision() -> None:
    from quant_fund.models.sundial import _pinned_from_pretrained

    _RecordingAutoModel.calls = []
    out = _pinned_from_pretrained(_RecordingAutoModel, "thuml/sundial-base-128m", _PINNED)
    assert out == "loaded"
    assert _RecordingAutoModel.calls == [
        (
            "thuml/sundial-base-128m",
            {"revision": _PINNED, "code_revision": _PINNED, "trust_remote_code": True},
        )
    ]
    with pytest.raises(ValueError, match="40-hex"):
        _pinned_from_pretrained(_RecordingAutoModel, "thuml/sundial-base-128m", "main")
    assert len(_RecordingAutoModel.calls) == 1


def test_fleet_receipt_records_sundial_revision() -> None:
    from quant_fund.research.fleet_eval import fleet_head_factories, run_distribution_fleet

    taus = (0.1, 0.5, 0.9)
    factories = {
        # Registry head: n_train below lookback fails fit before any hub load.
        **fleet_head_factories(taus, 0, ["sundial"]),
        "sundial_stub": lambda: SundialDistribution(taus, predictor=_FakeSundial(), lookback=16),
    }
    _, receipt = run_distribution_fleet(
        factories, ["iid_gaussian"], n_train=64, n_eval=8, seed=0, taus=taus
    )
    status = {row["model"]: row["status"] for row in receipt["results"]}
    assert status == {"sundial": "error", "sundial_stub": "ok"}
    for name in ("sundial", "sundial_stub"):
        assert receipt["model_versions"][name]["revision"] == _PINNED
