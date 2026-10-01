"""SYNTHETIC correctness fixtures, not financial predictive evidence."""

from __future__ import annotations

import importlib.util
import math
import sys
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pytest
from scipy.stats import norm

from quant_fund.models.multimodal_market import (
    GaussianForecast,
    MarketExample,
    MarketSample,
    MultimodalConfig,
    MultimodalMarketForecaster,
    NumericRecord,
    TextRecord,
    VisualRecord,
    visible_modalities,
)

START = datetime(2025, 1, 1, tzinfo=UTC)
HAS_TORCH = importlib.util.find_spec("torch") is not None
requires_torch = pytest.mark.skipif(not HAS_TORCH, reason="optional torch nn extra unavailable")


def sample(
    x: float = 0.0,
    text: str = "rising demand",
    *,
    day: int = 0,
    entity: str = "SYNTHETIC_A",
) -> MarketSample:
    cutoff = START + timedelta(days=day, hours=12)
    event = cutoff - timedelta(hours=1)
    return MarketSample(
        entity,
        cutoff,
        (NumericRecord(entity, f"n{day}", event, event, (x,)),),
        (TextRecord(entity, f"t{day}", event, event, text),),
    )


def example(s: MarketSample, y: float) -> MarketExample:
    event = s.cutoff + timedelta(hours=1)
    return MarketExample(s, y, event, event + timedelta(minutes=1))


@pytest.fixture(scope="module")
def fitted() -> MultimodalMarketForecaster:
    if not HAS_TORCH:
        pytest.skip("optional torch nn extra unavailable")
    import torch

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        rows = []
        # Repeated independent numeric/text combinations identify both effects.
        for i in range(32):
            x = (-1.0, -0.35, 0.35, 1.0)[i % 4]
            good = (i // 4) % 2 == 0
            text = "rising demand" if good else "falling demand"
            y = 0.8 * x + (0.7 if good else -0.7) + 0.04 * math.sin(i)
            rows.append(example(sample(x, text, day=i), y))
        model = MultimodalMarketForecaster(
            MultimodalConfig(
                numeric_dim=1,
                hidden_dim=8,
                attention_heads=2,
                max_numeric_records=3,
                max_text_tokens=8,
                epochs=130,
                learning_rate=0.015,
            )
        )
        model.fit(rows, training_cutoff=START + timedelta(days=33))
        yield model
    finally:
        torch.set_num_threads(previous_threads)


def test_gaussian_scores_are_proper_closed_form() -> None:
    f = GaussianForecast(np.array([0.0, 1.0]), np.array([1.0, 2.0]), np.ones((2, 3)), "joint")
    y = np.array([0.0, -1.0])
    np.testing.assert_allclose(f.nll(y), -norm.logpdf(y, loc=f.mean, scale=f.sigma))
    z = (y - f.mean) / f.sigma
    expected = f.sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / math.sqrt(math.pi))
    np.testing.assert_allclose(f.crps(y), expected)
    assert not f.mean.flags.writeable
    with pytest.raises(ValueError, match="targets"):
        f.nll([0.0])


def test_exact_publication_delay_boundary() -> None:
    s = sample()
    c = MultimodalConfig(1, text_delay=timedelta(hours=1))
    assert len(visible_modalities(s, c).text) == 1
    delayed = replace(s, cutoff=s.cutoff - timedelta(microseconds=1))
    assert visible_modalities(delayed, c).text == ()
    assert len(visible_modalities(delayed, c).numeric) == 1


def test_each_modality_uses_its_own_publication_clock() -> None:
    s = sample()
    visual = VisualRecord(
        s.entity_id,
        "v",
        s.cutoff - timedelta(hours=2),
        s.cutoff - timedelta(minutes=10),
        (0.2, 0.3),
        "SYNTHETIC_ENCODER",
    )
    s = replace(s, visual=(visual,))
    c = MultimodalConfig(
        1,
        visual_dim=2,
        visual_encoder_id="SYNTHETIC_ENCODER",
        numeric_delay=timedelta(hours=2),
        visual_delay=timedelta(minutes=11),
    )
    visible = visible_modalities(s, c)
    assert visible.numeric == () and visible.visual == () and len(visible.text) == 1


def test_dst_fold_cannot_make_future_publication_visible() -> None:
    zone = ZoneInfo("America/New_York")
    cutoff = datetime(2025, 11, 2, 1, 30, tzinfo=zone, fold=0)
    future = datetime(2025, 11, 2, 1, 10, tzinfo=zone, fold=1)
    s = MarketSample(
        "SYNTHETIC_A",
        cutoff,
        text=(TextRecord("SYNTHETIC_A", "future", future, future, "future news"),),
    )
    assert visible_modalities(s, MultimodalConfig(1)).text == ()
    with pytest.raises(ValueError, match="strictly after"):
        MarketExample(replace(s, cutoff=future), 0.1, cutoff, future)


def test_entity_alignment_clocks_duplicates_and_dimensions_fail_closed() -> None:
    s = sample()
    with pytest.raises(ValueError, match="entity alignment"):
        replace(s, entity_id="SYNTHETIC_OTHER")
    with pytest.raises(ValueError, match="duplicate"):
        replace(s, numeric=s.numeric * 2)
    with pytest.raises(ValueError, match="dimension"):
        visible_modalities(s, MultimodalConfig(2))
    with pytest.raises(ValueError, match="timezone-aware"):
        replace(s, cutoff=s.cutoff.replace(tzinfo=None))
    with pytest.raises(ValueError, match="precede"):
        replace(s.numeric[0], published_at=s.numeric[0].event_time - timedelta(seconds=1))
    with pytest.raises(ValueError, match="finite"):
        replace(s.numeric[0], values=(float("nan"),))
    with pytest.raises(ValueError, match="immutable"):
        replace(s, text=list(s.text))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="encoder identity"):
        visible_modalities(
            replace(
                s,
                visual=(
                    VisualRecord(
                        s.entity_id,
                        "v",
                        s.cutoff,
                        s.cutoff,
                        (1.0,),
                        "WRONG",
                    ),
                ),
            ),
            MultimodalConfig(1, visual_dim=1, visual_encoder_id="EXPECTED"),
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"numeric_dim": 0},
        {"numeric_dim": True},
        {"numeric_dim": 1, "hidden_dim": 3},
        {"numeric_dim": 1, "text_delay": timedelta(seconds=-1)},
        {"numeric_dim": 1, "learning_rate": float("nan")},
        {"numeric_dim": 1, "visual_dim": 1},
        {"numeric_dim": 1, "epochs": False},
    ],
)
def test_config_rejects_invalid_values(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        MultimodalConfig(**kwargs)


def test_fit_validates_targets_before_loading_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)
    m = MultimodalMarketForecaster(MultimodalConfig(1))
    e = example(sample(), 0.2)
    with pytest.raises(ValueError, match="unpublished"):
        m.fit([e], training_cutoff=e.sample.cutoff + timedelta(minutes=5))
    with pytest.raises(ValueError, match="strictly after"):
        replace(e, target_event_time=e.sample.cutoff)
    with pytest.raises(ValueError, match="precede"):
        replace(e, target_available_time=e.target_event_time - timedelta(seconds=1))
    with pytest.raises(ValueError, match="no visible"):
        m.fit(
            [example(replace(e.sample, numeric=(), text=()), 0.2)],
            training_cutoff=START + timedelta(days=1),
        )


def test_module_and_scores_do_not_require_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    import quant_fund.models.multimodal_market as module

    monkeypatch.setitem(sys.modules, "torch", None)
    spec = importlib.util.spec_from_file_location("_multimodal_no_torch", module.__file__)
    assert spec is not None and spec.loader is not None
    probe = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "_multimodal_no_torch", probe)
    spec.loader.exec_module(probe)
    f = probe.GaussianForecast(np.zeros(1), np.ones(1), np.ones((1, 3)), "joint")
    assert np.isfinite(f.crps([0.0])).all()
    m = MultimodalMarketForecaster(MultimodalConfig(1, epochs=1))
    with pytest.raises(ImportError, match="optional nn extra"):
        m.fit([example(sample(), 0.2)], training_cutoff=START + timedelta(days=1))


@requires_torch
def test_fit_predict_and_both_modalities_influence_forecast(
    fitted: MultimodalMarketForecaster,
) -> None:
    s = sample(0.0, "rising demand", day=40)
    f = fitted.predict([s])
    text_changed = fitted.predict([replace(s, text=(replace(s.text[0], text="falling demand"),))])
    numeric_changed = fitted.predict([replace(s, numeric=(replace(s.numeric[0], values=(1.0,)),))])
    assert f.mean[0] - text_changed.mean[0] > 0.4
    assert numeric_changed.mean[0] - f.mean[0] > 0.3
    assert np.isfinite(f.mean).all() and np.all(f.sigma > 0)
    assert fitted.fit_losses[-1] < fitted.fit_losses[0]
    assert all(not p.requires_grad for p in fitted._network.parameters())
    assert fitted.metadata()["pretrained_encoder"] is False
    assert fitted.metadata()["market_evidence"] is False


@requires_torch
def test_future_text_and_numeric_suffix_invariance(fitted: MultimodalMarketForecaster) -> None:
    s = sample(0.3, day=40)
    future = s.cutoff + timedelta(seconds=1)
    extended = replace(
        s,
        text=s.text
        + (
            TextRecord(
                s.entity_id, "future-text", future, future, "UNSEEN SECRET FUTURE news token"
            ),
        ),
        numeric=s.numeric
        + (NumericRecord(s.entity_id, "future-numeric", future, future, (999.0,)),),
    )
    before = fitted.predict([s])
    after = fitted.predict([extended])
    np.testing.assert_array_equal(before.mean, after.mean)
    np.testing.assert_array_equal(before.sigma, after.sigma)
    assert "secret" not in fitted.vocabulary


@requires_torch
def test_published_future_news_is_excluded(fitted: MultimodalMarketForecaster) -> None:
    s = sample(day=40)
    late = replace(s.text[0], published_at=s.cutoff + timedelta(seconds=1))
    f = fitted.predict([replace(s, text=(late,))])
    absent = fitted.predict([replace(s, text=())])
    np.testing.assert_array_equal(f.mean, absent.mean)
    np.testing.assert_array_equal(f.presence, [[1.0, 0.0, 0.0]])


@requires_torch
def test_inference_freezes_encoder_vocabulary_and_scaling(
    fitted: MultimodalMarketForecaster,
) -> None:
    state, preprocessing = fitted.state_sha256(), fitted.preprocessing_sha256
    model_hash = fitted.model_sha256()
    mean, scale = fitted.numeric_scaling
    fitted.predict([sample(100.0, "entirely unseen vocabulary", day=41)])
    assert fitted.state_sha256() == state
    assert fitted.preprocessing_sha256 == preprocessing
    assert fitted.model_sha256() == model_hash
    assert fitted.metadata()["model_sha256"] == model_hash
    assert fitted.metadata()["configuration"]["numeric_dim"] == 1
    mean[:] = 123.0
    np.testing.assert_array_equal(fitted.numeric_scaling[0], np.zeros(1))
    with pytest.raises(TypeError):
        fitted.vocabulary["leak"] = 99  # type: ignore[index]
    assert np.all(scale > 0)
    with pytest.raises(RuntimeError, match="frozen"):
        fitted.fit([example(sample(), 0.2)], training_cutoff=START + timedelta(days=1))


@requires_torch
def test_ablation_masks_inputs_and_missing_modalities(fitted: MultimodalMarketForecaster) -> None:
    s = sample(0.2, day=40)
    opposite = replace(s, text=(replace(s.text[0], text="falling demand"),))
    n1, n2 = (fitted.predict([x], ablation="numeric_only") for x in (s, opposite))
    np.testing.assert_array_equal(n1.mean, n2.mean)
    np.testing.assert_array_equal(n1.presence, [[1.0, 0.0, 0.0]])
    numeric_changed = replace(s, numeric=(replace(s.numeric[0], values=(-99.0,)),))
    t1, t2 = (fitted.predict([x], ablation="text_only") for x in (s, numeric_changed))
    np.testing.assert_array_equal(t1.mean, t2.mean)
    np.testing.assert_array_equal(t1.presence, [[0.0, 1.0, 0.0]])
    for missing in (replace(s, numeric=()), replace(s, text=())):
        assert np.isfinite(fitted.predict([missing]).sigma).all()
    with pytest.raises(ValueError, match="no visible"):
        fitted.predict([replace(s, numeric=())], ablation="numeric_only")
    with pytest.raises(ValueError, match="no visible"):
        fitted.predict([replace(s, numeric=(), text=())])


@requires_torch
def test_score_requires_chronological_holdout(fitted: MultimodalMarketForecaster) -> None:
    scores = fitted.score([example(sample(0.0, day=40), 0.7)])
    assert set(scores) == {"gaussian_nll", "crps"}
    assert all(math.isfinite(v) for v in scores.values())
    with pytest.raises(ValueError, match="strictly after training"):
        fitted.score([example(sample(day=0), 0.2)])
    bad_horizon = example(sample(day=40), 0.2)
    with pytest.raises(ValueError, match="forecast horizon"):
        fitted.score(
            [
                replace(
                    bad_horizon,
                    target_event_time=bad_horizon.target_event_time + timedelta(seconds=1),
                    target_available_time=bad_horizon.target_available_time + timedelta(seconds=1),
                )
            ]
        )
    with pytest.raises(ValueError, match="duplicate"):
        fitted.predict([sample(day=40), sample(day=40)])
    with pytest.raises(ValueError, match="typed"):
        fitted.predict(["wrong"])  # type: ignore[list-item]
    with pytest.raises(AttributeError):
        fitted.training_cutoff = START  # type: ignore[misc]
    with pytest.raises(AttributeError):
        fitted.config = MultimodalConfig(1)  # type: ignore[misc]


@requires_torch
def test_training_preprocessing_ignores_future_suffix_and_delivery_delays() -> None:
    import torch

    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        s = sample(2.0)
        delayed = replace(
            s.text[0],
            record_id="delayed",
            published_at=s.cutoff - timedelta(minutes=1),
            text="delayed words must never enter vocabulary",
        )
        future = NumericRecord(
            s.entity_id,
            "future",
            s.cutoff + timedelta(days=1),
            s.cutoff + timedelta(days=1),
            (10000.0,),
        )
        extended = replace(s, numeric=s.numeric + (future,), text=s.text + (delayed,))
        c = MultimodalConfig(
            1,
            epochs=2,
            text_delay=timedelta(minutes=2),
            hidden_dim=4,
            max_numeric_records=2,
            max_text_tokens=5,
        )
        a = MultimodalMarketForecaster(c).fit(
            [example(s, 1.0)], training_cutoff=START + timedelta(days=1)
        )
        b = MultimodalMarketForecaster(c).fit(
            [example(extended, 1.0)], training_cutoff=START + timedelta(days=1)
        )
        assert a.training_sha256 == b.training_sha256
        assert a.preprocessing_sha256 == b.preprocessing_sha256
        assert a.state_sha256() == b.state_sha256()
        assert a.model_sha256() == b.model_sha256()
        later_clock = MultimodalMarketForecaster(c).fit(
            [example(s, 1.0)],
            training_cutoff=START + timedelta(days=2),
        )
        assert later_clock.state_sha256() == a.state_sha256()
        assert later_clock.preprocessing_sha256 == a.preprocessing_sha256
        assert later_clock.model_sha256() != a.model_sha256()
        assert "delayed" not in b.vocabulary
        np.testing.assert_array_equal(b.numeric_scaling[0], [2.0])
        np.testing.assert_array_equal(b.numeric_scaling[1], [1.0])
    finally:
        torch.set_num_threads(previous)


@requires_torch
def test_trainable_visual_path_and_separately_trained_ablation() -> None:
    import torch

    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        s = sample()
        v = VisualRecord(s.entity_id, "v", s.cutoff, s.cutoff, (0.2,), "SYNTHETIC_ENCODER")
        s = replace(s, visual=(v,))
        c = MultimodalConfig(
            1,
            visual_dim=1,
            visual_encoder_id="SYNTHETIC_ENCODER",
            epochs=3,
            hidden_dim=4,
            max_numeric_records=2,
            max_text_tokens=5,
        )
        joint = MultimodalMarketForecaster(c).fit(
            [example(s, 0.3)], training_cutoff=START + timedelta(days=1)
        )
        a = joint.predict([s])
        b = joint.predict([replace(s, visual=(replace(v, values=(2.0,)),))])
        assert not np.array_equal(a.mean, b.mean)
        np.testing.assert_array_equal(a.presence, [[1.0, 1.0, 1.0]])
        baseline = MultimodalMarketForecaster(c).fit(
            [example(s, 0.3)], training_cutoff=START + timedelta(days=1), ablation="text_only"
        )
        assert baseline.predict([s]).ablation == "text_only"
        with pytest.raises(ValueError, match="untrained modalities"):
            baseline.predict([s], ablation="joint")
    finally:
        torch.set_num_threads(previous)
