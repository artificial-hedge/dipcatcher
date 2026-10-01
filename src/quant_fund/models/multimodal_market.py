"""Publication-aware numeric/text cross-attention Gaussian forecasting.

This small learned text encoder uses Transformer attention from Vaswani et al.,
https://arxiv.org/abs/1706.03762. It is not pretrained BERT, a FinGPT checkpoint,
or a reproduction of a published multimodal financial performance result.
Torch is an optional, lazily loaded CPU training dependency. Publication clocks,
entity alignment and train-only preprocessing are enforced before tensors are
built. Supplied clocks/data rights must still be independently verified.

Missing modalities have explicit learned null tokens and presence indicators;
an entirely empty visible sample is rejected. Optional visual observations are
identified external encoder vectors, not images or learned text representations.
``numeric_only`` and ``text_only`` mask other inputs to the same fitted network;
they are counterfactual ablations, not independently retrained baseline models.
All predictive/evaluation claims require independent chronological market data.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from quant_fund.metrics.scoring import crps_gaussian

Array = NDArray[np.float64]
Ablation = Literal["joint", "numeric_only", "text_only"]
_TOKENIZER = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_SPECIAL_TOKENS = ("<PAD>", "<UNK>", "<DOCUMENT>", "<MISSING>")
_IMPLEMENTATION_VERSION = "multimodal_market.v1"


def _clock(value: datetime, name: str) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be a timezone-aware datetime")


def _utc(value: datetime) -> datetime:
    return value.astimezone(UTC)


def _name(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def _count(value: int, name: str, minimum: int = 1) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _values(values: tuple[float, ...], name: str) -> None:
    if not isinstance(values, tuple) or not values:
        raise ValueError(f"{name} must be a nonempty immutable tuple")
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in values):
        raise ValueError(f"{name} must contain finite numbers")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain finite numbers")


def _record_clocks(entity_id: str, record_id: str, event: datetime, published: datetime) -> None:
    _name(entity_id, "entity_id")
    _name(record_id, "record_id")
    _clock(event, "event_time")
    _clock(published, "published_at")
    if _utc(event) > _utc(published):
        raise ValueError("published_at cannot precede event_time")


@dataclass(frozen=True, slots=True)
class NumericRecord:
    entity_id: str
    record_id: str
    event_time: datetime
    published_at: datetime
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        _record_clocks(self.entity_id, self.record_id, self.event_time, self.published_at)
        _values(self.values, "numeric values")


@dataclass(frozen=True, slots=True)
class TextRecord:
    entity_id: str
    record_id: str
    event_time: datetime
    published_at: datetime
    text: str

    def __post_init__(self) -> None:
        _record_clocks(self.entity_id, self.record_id, self.event_time, self.published_at)
        _name(self.text, "original text")
        if not _TOKENIZER.findall(self.text):
            raise ValueError("text must contain tokens")


@dataclass(frozen=True, slots=True)
class VisualRecord:
    entity_id: str
    record_id: str
    event_time: datetime
    published_at: datetime
    values: tuple[float, ...]
    encoder_id: str

    def __post_init__(self) -> None:
        _record_clocks(self.entity_id, self.record_id, self.event_time, self.published_at)
        _values(self.values, "visual values")
        _name(self.encoder_id, "visual encoder_id")


@dataclass(frozen=True, slots=True)
class MarketSample:
    entity_id: str
    cutoff: datetime
    numeric: tuple[NumericRecord, ...] = ()
    text: tuple[TextRecord, ...] = ()
    visual: tuple[VisualRecord, ...] = ()

    def __post_init__(self) -> None:
        _name(self.entity_id, "entity_id")
        _clock(self.cutoff, "cutoff")
        for name, records, kind in (
            ("numeric", self.numeric, NumericRecord),
            ("text", self.text, TextRecord),
            ("visual", self.visual, VisualRecord),
        ):
            if not isinstance(records, tuple) or any(not isinstance(r, kind) for r in records):
                raise ValueError(f"{name} must be an immutable tuple of typed records")
            if any(r.entity_id != self.entity_id for r in records):
                raise ValueError(f"{name} entity alignment mismatch")
            if len({r.record_id for r in records}) != len(records):
                raise ValueError(f"duplicate {name} record_id")


@dataclass(frozen=True, slots=True)
class MarketExample:
    sample: MarketSample
    target: float
    target_event_time: datetime
    target_available_time: datetime

    def __post_init__(self) -> None:
        if not isinstance(self.sample, MarketSample):
            raise ValueError("sample must be a MarketSample")
        if isinstance(self.target, bool) or not isinstance(self.target, (int, float)):
            raise ValueError("target must be a finite number")
        if not math.isfinite(self.target):
            raise ValueError("target must be a finite number")
        _clock(self.target_event_time, "target_event_time")
        _clock(self.target_available_time, "target_available_time")
        if _utc(self.target_event_time) <= _utc(self.sample.cutoff):
            raise ValueError("forecast target must occur strictly after sample cutoff")
        if _utc(self.target_available_time) < _utc(self.target_event_time):
            raise ValueError("target_available_time cannot precede target event")


@dataclass(frozen=True, slots=True)
class MultimodalConfig:
    numeric_dim: int
    visual_dim: int = 0
    visual_encoder_id: str | None = None
    hidden_dim: int = 16
    attention_heads: int = 2
    max_numeric_records: int = 16
    max_visual_records: int = 8
    max_text_tokens: int = 64
    max_vocabulary: int = 4096
    numeric_delay: timedelta = timedelta(0)
    text_delay: timedelta = timedelta(0)
    visual_delay: timedelta = timedelta(0)
    learning_rate: float = 0.01
    epochs: int = 100
    min_sigma: float = 0.05  # Gaussian scale floor in standardized target units.
    seed: int = 42

    def __post_init__(self) -> None:
        for name in (
            "numeric_dim",
            "hidden_dim",
            "attention_heads",
            "max_numeric_records",
            "max_visual_records",
            "max_text_tokens",
            "max_vocabulary",
            "epochs",
        ):
            _count(getattr(self, name), name)
        _count(self.visual_dim, "visual_dim", 0)
        _count(self.seed, "seed", 0)
        if self.hidden_dim % self.attention_heads:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        if self.visual_dim:
            if self.visual_encoder_id is None:
                raise ValueError("visual_encoder_id is required when visual_dim > 0")
            _name(self.visual_encoder_id, "visual_encoder_id")
        elif self.visual_encoder_id is not None:
            raise ValueError("visual_encoder_id requires visual_dim > 0")
        for name in ("numeric_delay", "text_delay", "visual_delay"):
            delay = getattr(self, name)
            if not isinstance(delay, timedelta) or delay < timedelta(0):
                raise ValueError(f"{name} must be a nonnegative timedelta")
        for name in ("learning_rate", "min_sigma"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be positive and finite")


@dataclass(frozen=True, slots=True)
class VisibleModalities:
    numeric: tuple[NumericRecord, ...]
    text: tuple[TextRecord, ...]
    visual: tuple[VisualRecord, ...]


def visible_modalities(sample: MarketSample, config: MultimodalConfig) -> VisibleModalities:
    """Validate alignment/dimensions and mask publication delays before encoding.

    Numeric/visual histories retain the latest configured number of records.
    Both event_time and published_at + the modality's delivery delay must be
    <= cutoff. Future records never affect tokenization or normalization.
    """
    if not isinstance(sample, MarketSample):
        raise ValueError("sample must be a MarketSample")
    if any(len(r.values) != config.numeric_dim for r in sample.numeric):
        raise ValueError("numeric dimension mismatch")
    if any(len(r.values) != config.visual_dim for r in sample.visual):
        raise ValueError("visual dimension mismatch")
    if any(r.encoder_id != config.visual_encoder_id for r in sample.visual):
        raise ValueError("visual encoder identity mismatch")

    def visible(records: tuple[Any, ...], delay: timedelta, limit: int) -> tuple[Any, ...]:
        selected = [
            r
            for r in records
            if _utc(r.event_time) <= _utc(sample.cutoff)
            and _utc(r.published_at) + delay <= _utc(sample.cutoff)
        ]
        selected.sort(key=lambda r: (_utc(r.published_at), _utc(r.event_time), r.record_id))
        return tuple(selected[-limit:])

    return VisibleModalities(
        visible(sample.numeric, config.numeric_delay, config.max_numeric_records),
        visible(sample.text, config.text_delay, len(sample.text) or 1),
        visible(sample.visual, config.visual_delay, config.max_visual_records),
    )


def _tokens(records: tuple[TextRecord, ...], limit: int) -> tuple[str, ...]:
    stream: list[str] = []
    for record in records:
        stream.extend(_TOKENIZER.findall(record.text.casefold()))
        stream.append("<DOCUMENT>")
    return tuple(stream[-limit:])


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ImportError("multimodal market training needs the optional nn extra (torch)") from exc
    return torch


def _network(torch: Any, config: MultimodalConfig, vocabulary_size: int) -> Any:
    nn = torch.nn

    class CrossAttentionGaussian(nn.Module):  # type: ignore[misc, name-defined]
        def __init__(self) -> None:
            super().__init__()
            d = config.hidden_dim
            self.numeric_projection = nn.Linear(config.numeric_dim, d)
            self.numeric_positions = nn.Embedding(config.max_numeric_records, d)
            self.numeric_null = nn.Parameter(torch.zeros(d))
            self.text_embedding = nn.Embedding(vocabulary_size, d, padding_idx=0)
            self.text_positions = nn.Embedding(config.max_text_tokens, d)
            self.text_attention = nn.MultiheadAttention(d, config.attention_heads, batch_first=True)
            self.text_norm = nn.LayerNorm(d)
            self.numeric_attention = nn.MultiheadAttention(
                d, config.attention_heads, batch_first=True
            )
            self.numeric_norm = nn.LayerNorm(d)
            self.cross_attention = nn.MultiheadAttention(
                d, config.attention_heads, batch_first=True
            )
            self.cross_norm = nn.LayerNorm(d)
            if config.visual_dim:
                self.visual_projection = nn.Linear(config.visual_dim, d)
                self.visual_positions = nn.Embedding(config.max_visual_records, d)
                self.visual_null = nn.Parameter(torch.zeros(d))
            self.head = nn.Sequential(nn.Linear(4 * d + 3, d), nn.GELU(), nn.Linear(d, 2))

        def forward(self, batch: dict[str, Any], ablation: Ablation) -> tuple[Any, Any]:
            flags = batch["presence"].clone()
            numeric = self.numeric_projection(batch["numeric"])
            numeric = numeric + self.numeric_positions(torch.arange(numeric.shape[1]))
            numeric = torch.where(flags[:, 0, None, None] > 0, numeric, self.numeric_null)
            numeric_mask = batch["numeric_mask"]
            if ablation == "text_only":
                flags[:, 0] = 0
                flags[:, 2] = 0
                numeric = self.numeric_null.expand_as(numeric)
                numeric_mask = torch.ones_like(numeric_mask)
                numeric_mask[:, 0] = False
            attended, _ = self.numeric_attention(
                numeric, numeric, numeric, key_padding_mask=numeric_mask
            )
            numeric = self.numeric_norm(numeric + attended)
            numeric_weights = (~numeric_mask).unsqueeze(-1)
            numeric_mean = (numeric * numeric_weights).sum(1) / numeric_weights.sum(1)

            text_mask = batch["text_mask"]
            text = self.text_embedding(batch["text"])
            text = text + self.text_positions(torch.arange(text.shape[1]))
            attended, _ = self.text_attention(text, text, text, key_padding_mask=text_mask)
            text = self.text_norm(text + attended)
            text_weights = (~text_mask).unsqueeze(-1)
            text_mean = (text * text_weights).sum(1) / text_weights.sum(1)
            memory, memory_mask = text, text_mask
            visual_mean = torch.zeros_like(text_mean)
            if config.visual_dim and ablation != "text_only":
                visual = self.visual_projection(batch["visual"])
                visual = visual + self.visual_positions(torch.arange(visual.shape[1]))
                visual = torch.where(flags[:, 2, None, None] > 0, visual, self.visual_null)
                visual_weights = (~batch["visual_mask"]).unsqueeze(-1)
                visual_mean = (visual * visual_weights).sum(1) / visual_weights.sum(1)
                memory = torch.cat((text, visual), dim=1)
                memory_mask = torch.cat((text_mask, batch["visual_mask"]), dim=1)
            cross, _ = self.cross_attention(numeric, memory, memory, key_padding_mask=memory_mask)
            cross = self.cross_norm(numeric + cross)
            cross_mean = (cross * numeric_weights).sum(1) / numeric_weights.sum(1)
            if ablation == "numeric_only":
                flags[:, 1:] = 0
                cross_mean = torch.zeros_like(cross_mean)
                text_mean = torch.zeros_like(text_mean)
                visual_mean = torch.zeros_like(visual_mean)
            features = torch.cat((numeric_mean, cross_mean, text_mean, visual_mean, flags), dim=1)
            raw = self.head(features)
            return raw[:, 0], torch.nn.functional.softplus(raw[:, 1]) + config.min_sigma

    return CrossAttentionGaussian().cpu()


@dataclass(frozen=True, slots=True)
class GaussianForecast:
    mean: Array
    sigma: Array
    presence: Array
    ablation: Ablation

    def __post_init__(self) -> None:
        mean, sigma, presence = (
            np.array(x, dtype=float, copy=True)
            for x in (
                self.mean,
                self.sigma,
                self.presence,
            )
        )
        if mean.ndim != 1 or sigma.shape != mean.shape or presence.shape != (len(mean), 3):
            raise ValueError("forecast dimensions must match")
        if not np.isfinite(mean).all() or not np.isfinite(sigma).all() or np.any(sigma <= 0):
            raise ValueError("Gaussian forecast must be finite with positive scale")
        if not np.isin(presence, [0.0, 1.0]).all():
            raise ValueError("presence must be binary")
        for name, value in (("mean", mean), ("sigma", sigma), ("presence", presence)):
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    def _targets(self, targets: Sequence[float] | Array) -> Array:
        y = np.asarray(targets, dtype=float)
        if y.shape != self.mean.shape or not np.isfinite(y).all():
            raise ValueError("targets must be finite and match forecast rows")
        return y

    def nll(self, targets: Sequence[float] | Array) -> Array:
        y = self._targets(targets)
        return np.asarray(
            0.5 * math.log(2 * math.pi)
            + np.log(self.sigma)
            + 0.5 * ((y - self.mean) / self.sigma) ** 2,
            dtype=float,
        )

    def crps(self, targets: Sequence[float] | Array) -> Array:
        return crps_gaussian(self._targets(targets), self.mean, self.sigma)


class MultimodalMarketForecaster:
    """Train a small CPU cross-attention model; freeze it for inference.

    ``fit`` accepts only a declared training partition and an explicit training
    cutoff. All targets must already be published at that cutoff. The caller
    still owns purging/embargo and ensuring records were genuinely available.
    ``score`` requires decision cutoffs strictly after the fitted training cutoff.
    ``predict`` also supports retrospective training diagnostics without calling
    those predictions out-of-sample.
    """

    def __init__(self, config: MultimodalConfig) -> None:
        if not isinstance(config, MultimodalConfig):
            raise ValueError("config must be a MultimodalConfig")
        self._config = config
        self._network: Any = None
        self._vocabulary: tuple[str, ...] = ()
        self._numeric_mean = np.zeros(config.numeric_dim)
        self._numeric_scale = np.ones(config.numeric_dim)
        self._visual_mean = np.zeros(config.visual_dim)
        self._visual_scale = np.ones(config.visual_dim)
        self._target_mean = 0.0
        self._target_scale = 1.0
        self._training_cutoff: datetime | None = None
        self._fit_losses: tuple[float, ...] = ()
        self._training_sha256: str | None = None
        self._fit_ablation: Ablation = "joint"
        self._target_horizon: timedelta | None = None
        self._implementation_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    @property
    def config(self) -> MultimodalConfig:
        return self._config

    @property
    def training_cutoff(self) -> datetime | None:
        return self._training_cutoff

    @property
    def fit_losses(self) -> tuple[float, ...]:
        return self._fit_losses

    @property
    def training_sha256(self) -> str | None:
        return self._training_sha256

    @property
    def fit_ablation(self) -> Ablation:
        return self._fit_ablation

    @property
    def target_horizon(self) -> timedelta | None:
        return self._target_horizon

    @property
    def vocabulary(self) -> Mapping[str, int]:
        return MappingProxyType({word: i for i, word in enumerate(self._vocabulary)})

    @property
    def numeric_scaling(self) -> tuple[Array, Array]:
        return self._numeric_mean.copy(), self._numeric_scale.copy()

    @property
    def preprocessing_sha256(self) -> str:
        payload = {
            "vocabulary": self._vocabulary,
            "numeric_mean": self._numeric_mean.tolist(),
            "numeric_scale": self._numeric_scale.tolist(),
            "visual_mean": self._visual_mean.tolist(),
            "visual_scale": self._visual_scale.tolist(),
            "target_mean": self._target_mean,
            "target_scale": self._target_scale,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def state_sha256(self) -> str:
        """Hash tensors only; use model_sha256 for the complete fitted identity."""
        if self._network is None:
            raise RuntimeError("model is not fitted")
        digest = hashlib.sha256()
        for name, value in sorted(self._network.state_dict().items()):
            array = value.detach().cpu().numpy()
            digest.update(name.encode())
            digest.update(str(array.shape).encode())
            digest.update(str(array.dtype).encode())
            digest.update(array.tobytes())
        return digest.hexdigest()

    def _configuration(self) -> dict[str, Any]:
        configuration = asdict(self.config)
        for field in ("numeric_delay", "text_delay", "visual_delay"):
            configuration[field] = getattr(self.config, field).total_seconds()
        return configuration

    def model_sha256(self) -> str:
        """Bind code bytes, config, frozen preprocessing/weights and training clocks.

        This is an identity/integrity hash, not proof that supplied source clocks
        or labels are genuine. Publication and data entitlement audits remain
        separate evidence requirements.
        """
        if self.training_cutoff is None:
            raise RuntimeError("model is not fitted")
        payload = {
            "implementation_version": _IMPLEMENTATION_VERSION,
            "implementation_sha256": self._implementation_sha256,
            "configuration": self._configuration(),
            "preprocessing_sha256": self.preprocessing_sha256,
            "state_sha256": self.state_sha256(),
            "training_sha256": self.training_sha256,
            "training_cutoff": self.training_cutoff.isoformat(),
            "target_horizon_seconds": self.target_horizon.total_seconds()
            if self.target_horizon is not None
            else None,
            "fit_ablation": self.fit_ablation,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def _select(
        self, samples: Sequence[MarketSample], ablation: Ablation
    ) -> list[VisibleModalities]:
        if ablation not in ("joint", "numeric_only", "text_only"):
            raise ValueError("unknown ablation")
        if not samples:
            raise ValueError("samples must be nonempty")
        if any(not isinstance(s, MarketSample) for s in samples):
            raise ValueError("samples must contain typed MarketSamples")
        if len({(s.entity_id, _utc(s.cutoff)) for s in samples}) != len(samples):
            raise ValueError("duplicate entity/cutoff sample")
        rows = [visible_modalities(s, self.config) for s in samples]
        if ablation == "numeric_only":
            rows = [VisibleModalities(r.numeric, (), ()) for r in rows]
        elif ablation == "text_only":
            rows = [VisibleModalities((), r.text, ()) for r in rows]
        if any(not (r.numeric or r.text or r.visual) for r in rows):
            raise ValueError("sample has no visible modality under requested ablation")
        return rows

    def _batch(self, rows: list[VisibleModalities], torch: Any) -> dict[str, Any]:
        c, n = self.config, len(rows)
        numeric = np.zeros((n, c.max_numeric_records, c.numeric_dim), dtype=np.float32)
        visual = np.zeros((n, c.max_visual_records, c.visual_dim), dtype=np.float32)
        tokens = np.zeros((n, c.max_text_tokens), dtype=np.int64)
        numeric_mask = np.ones((n, c.max_numeric_records), dtype=bool)
        visual_mask = np.ones((n, c.max_visual_records), dtype=bool)
        text_mask = np.ones((n, c.max_text_tokens), dtype=bool)
        presence = np.zeros((n, 3), dtype=np.float32)
        vocabulary = self.vocabulary
        for i, row in enumerate(rows):
            presence[i] = (bool(row.numeric), bool(row.text), bool(row.visual))
            if row.numeric:
                values = np.array([r.values for r in row.numeric])
                numeric[i, : len(values)] = (values - self._numeric_mean) / self._numeric_scale
                numeric_mask[i, : len(values)] = False
            else:
                numeric_mask[i, 0] = False
            words = _tokens(row.text, c.max_text_tokens)
            ids = [vocabulary.get(word, 1) for word in words] if words else [3]
            tokens[i, : len(ids)] = ids
            text_mask[i, : len(ids)] = False
            if row.visual:
                values = np.array([r.values for r in row.visual])
                visual[i, : len(values)] = (values - self._visual_mean) / self._visual_scale
                visual_mask[i, : len(values)] = False
            else:
                visual_mask[i, 0] = False
        return {
            name: torch.from_numpy(value)
            for name, value in (
                ("numeric", numeric),
                ("visual", visual),
                ("text", tokens),
                ("numeric_mask", numeric_mask),
                ("visual_mask", visual_mask),
                ("text_mask", text_mask),
                ("presence", presence),
            )
        }

    def fit(
        self,
        examples: Sequence[MarketExample],
        *,
        training_cutoff: datetime,
        ablation: Ablation = "joint",
    ) -> MultimodalMarketForecaster:
        _clock(training_cutoff, "training_cutoff")
        if not examples or any(not isinstance(e, MarketExample) for e in examples):
            raise ValueError("examples must be nonempty typed MarketExamples")
        if any(_utc(e.sample.cutoff) >= _utc(training_cutoff) for e in examples):
            raise ValueError("training decisions must precede training_cutoff")
        if any(_utc(e.target_available_time) > _utc(training_cutoff) for e in examples):
            raise ValueError("training target is unpublished at training_cutoff")
        horizons = {_utc(e.target_event_time) - _utc(e.sample.cutoff) for e in examples}
        if len(horizons) != 1:
            raise ValueError("training targets must use one forecast horizon")
        rows = self._select([e.sample for e in examples], ablation)
        if self._network is not None:
            raise RuntimeError("fitted model is frozen; construct a fresh instance to refit")
        counts: Counter[str] = Counter()
        for row in rows:
            counts.update(
                word
                for word in _tokens(row.text, self.config.max_text_tokens)
                if word not in _SPECIAL_TOKENS
            )
        words = sorted(counts, key=lambda w: (-counts[w], w))[: self.config.max_vocabulary]
        self._vocabulary = _SPECIAL_TOKENS + tuple(words)
        for attr, values in (
            ("numeric", [r.values for row in rows for r in row.numeric]),
            ("visual", [r.values for row in rows for r in row.visual]),
        ):
            if values:
                matrix = np.array(values, dtype=float)
                setattr(self, f"_{attr}_mean", np.mean(matrix, axis=0))
                scale = np.std(matrix, axis=0)
                setattr(self, f"_{attr}_scale", np.where(scale < 1e-8, 1.0, scale))
        target = np.array([e.target for e in examples], dtype=float)
        self._target_mean = float(np.mean(target))
        target_scale = float(np.std(target))
        self._target_scale = target_scale if target_scale >= 1e-8 else 1.0
        torch = _torch()
        batch = self._batch(rows, torch)
        standardized_target = torch.tensor(
            (target - self._target_mean) / self._target_scale,
            dtype=torch.float32,
        )
        losses: list[float] = []
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.config.seed)
            network = _network(torch, self.config, len(self._vocabulary))
            optimizer = torch.optim.Adam(network.parameters(), lr=self.config.learning_rate)
            network.train()
            for _ in range(self.config.epochs):
                optimizer.zero_grad()
                mean, sigma = network(batch, ablation)
                loss = (
                    torch.log(sigma)
                    + 0.5 * ((standardized_target - mean) / sigma) ** 2
                    + 0.5 * math.log(2 * math.pi)
                ).mean()
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("nonfinite training loss; fit rejected")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), 10.0)
                optimizer.step()
                losses.append(float(loss.detach()) + math.log(self._target_scale))
        network.eval()
        for parameter in network.parameters():
            parameter.requires_grad_(False)
        self._network = network
        self._fit_losses = tuple(losses)
        self._training_cutoff = _utc(training_cutoff)
        self._fit_ablation = ablation
        self._target_horizon = next(iter(horizons))
        payload = {
            "training_cutoff": training_cutoff.isoformat(),
            "ablation": ablation,
            "config": asdict(self.config),
            "preprocessing_sha256": self.preprocessing_sha256,
            "rows": [
                {
                    "entity_id": e.sample.entity_id,
                    "cutoff": e.sample.cutoff.isoformat(),
                    "visible": asdict(row),
                    "target": e.target,
                    "target_event_time": e.target_event_time.isoformat(),
                    "target_available_time": e.target_available_time.isoformat(),
                }
                for e, row in zip(examples, rows, strict=True)
            ],
        }
        self._training_sha256 = hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode(),
        ).hexdigest()
        return self

    def predict(
        self,
        samples: Sequence[MarketSample],
        *,
        ablation: Ablation | None = None,
    ) -> GaussianForecast:
        if self._network is None:
            raise RuntimeError("model is not fitted")
        ablation = self.fit_ablation if ablation is None else ablation
        if self.fit_ablation != "joint" and ablation != self.fit_ablation:
            raise ValueError("separately trained ablation cannot enable untrained modalities")
        rows = self._select(samples, ablation)
        torch = _torch()
        batch = self._batch(rows, torch)
        with torch.inference_mode():
            mean, sigma = self._network(batch, ablation)
        return GaussianForecast(
            np.asarray(mean.numpy(), dtype=float) * self._target_scale + self._target_mean,
            np.asarray(sigma.numpy(), dtype=float) * self._target_scale,
            np.asarray(batch["presence"].numpy(), dtype=float),
            ablation,
        )

    def score(
        self,
        examples: Sequence[MarketExample],
        *,
        ablation: Ablation | None = None,
    ) -> dict[str, float]:
        if self.training_cutoff is None:
            raise RuntimeError("model is not fitted")
        if not examples or any(not isinstance(e, MarketExample) for e in examples):
            raise ValueError("examples must be nonempty typed MarketExamples")
        if any(_utc(e.sample.cutoff) <= self.training_cutoff for e in examples):
            raise ValueError("scored decision cutoffs must be strictly after training cutoff")
        if any(
            _utc(e.target_event_time) - _utc(e.sample.cutoff) != self.target_horizon
            for e in examples
        ):
            raise ValueError("scored targets must match fitted forecast horizon")
        forecast = self.predict([e.sample for e in examples], ablation=ablation)
        target = np.array([e.target for e in examples])
        return {
            "gaussian_nll": float(np.mean(forecast.nll(target))),
            "crps": float(np.mean(forecast.crps(target))),
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "architecture": "learned_token_encoder_numeric_text_cross_attention_gaussian",
            "reference": "https://arxiv.org/abs/1706.03762",
            "device": "cpu",
            "pretrained_encoder": False,
            "fingpt_reproduction": False,
            "research_only": True,
            "market_evidence": False,
            "sota_claim": False,
            "implementation_version": _IMPLEMENTATION_VERSION,
            "implementation_sha256": self._implementation_sha256,
            "configuration": self._configuration(),
            "fit_ablation": self.fit_ablation,
            "training_cutoff": self.training_cutoff,
            "training_sha256": self.training_sha256,
            "target_horizon_seconds": self.target_horizon.total_seconds()
            if self.target_horizon is not None
            else None,
            "preprocessing_sha256": self.preprocessing_sha256,
            "state_sha256": self.state_sha256() if self._network is not None else None,
            "model_sha256": self.model_sha256() if self._network is not None else None,
            "vocabulary_size": len(self._vocabulary),
            "epochs_completed": len(self.fit_losses),
        }
