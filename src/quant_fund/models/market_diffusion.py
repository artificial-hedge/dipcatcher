"""Joint multivariate path DDPM for research scenario generation.

Ho, Jain & Abbeel (2020), https://arxiv.org/abs/2006.11239: forward
Gaussian corruption (Eq. 4), simplified epsilon loss (Eq. 14), and fixed
posterior-variance reverse transitions (Eq. 7). Nichol & Dhariwal (2021),
https://arxiv.org/abs/2102.09672: cosine cumulative-alpha schedule, offset
0.008 and beta clipping at 0.999. These image-model algorithms are adapted
to a jointly flattened horizon-by-asset return path, with a small dense MLP
and one-hot diffusion time. This is not an image-paper reproduction, a
conditional/event model, or a claim of realistic market tails.

The denoiser is actually trained on all path coordinates together; assets
and horizons are NOT independent marginal DiffPTS forecasts. CPU float64
torch optimization is seeded and single-threaded. Inference uses extracted
numpy weights and a local RNG; importing/sampling does not require torch.
Scaling, clipping bounds, optimization and hashes use only complete windows
available at the explicitly supplied training cutoff. Post-cutoff windows
are excluded, never used for early stopping or model selection.

Every generated path is SYNTHETIC even when the training input is empirical.
Optional denoised-coordinate clipping uses training support and therefore
limits extrapolation into unseen tails. Distribution gaps are descriptive
diagnostics, not a pass certificate, proper forecast score, or SOTA claim.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
ClockGrid = Sequence[Sequence[datetime]]
_MAX_CELLS = 2_000_000
_MAX_DIMENSION = 8_192
_MAX_UPDATES = 100_000
_REVISION = "joint_path_ddpm_v1"


def _count(value: int, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not minimum <= int(value) <= maximum:
        raise ValueError(f"{name} must be within [{minimum}, {maximum}]")
    return int(value)


def _clock(value: datetime, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} requires timezone-aware datetimes")
    return value.astimezone(UTC)


def _readonly(value: Array) -> Array:
    arr = np.array(value, dtype=np.float64, copy=True)
    arr.flags.writeable = False
    return arr


def _paths(value: Array, name: str, minimum: int = 1) -> Array:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[0] < minimum or arr.shape[1] < 2 or arr.shape[2] < 2:
        raise ValueError(f"{name} must have shape [N>={minimum}, H>=2, A>=2]")
    if (
        arr.size > _MAX_CELLS
        or arr.shape[1] * arr.shape[2] > _MAX_DIMENSION
        or arr.shape[1] > 256
        or arr.shape[2] > 128
    ):
        raise ValueError(f"{name} exceeds the bounded path resource budget")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must be finite")
    return arr


def _ids(value: Sequence[str], n_assets: int) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise ValueError("asset_ids must be a sequence of distinct identifiers")
    ids = tuple(value)
    if len(ids) != n_assets or any(not isinstance(x, str) or not x.strip() for x in ids):
        raise ValueError("asset_ids must be nonempty strings matching the asset axis")
    if len(set(ids)) != len(ids) or any(x != x.strip() for x in ids):
        raise ValueError("asset_ids must be distinct and have no surrounding whitespace")
    return ids


def _clocks(
    timestamps: ClockGrid, available_times: ClockGrid, shape: tuple[int, int]
) -> tuple[tuple[tuple[datetime, ...], ...], tuple[tuple[datetime, ...], ...]]:
    def parse(value: ClockGrid, name: str) -> tuple[tuple[datetime, ...], ...]:
        if len(value) != shape[0] or any(len(row) != shape[1] for row in value):
            raise ValueError(f"{name} must match the [N,H] path clock grid")
        return tuple(tuple(_clock(t, name) for t in row) for row in value)

    events, available = parse(timestamps, "timestamps"), parse(available_times, "available_times")
    for row, ready in zip(events, available, strict=True):
        if any(left >= right for left, right in zip(row[:-1], row[1:], strict=True)):
            raise ValueError("timestamps must be strictly increasing within each path")
        if any(a < t for t, a in zip(row, ready, strict=True)):
            raise ValueError("available_times must not precede event timestamps")
    if any(a[-1] >= b[-1] for a, b in zip(events[:-1], events[1:], strict=True)):
        raise ValueError("path end timestamps must be distinct and chronological")
    return events, available


def _provenance(data_source: str, synthetic: bool) -> None:
    if (
        not isinstance(data_source, str)
        or not data_source.strip()
        or data_source != data_source.strip()
    ):
        raise ValueError("data_source must be a nonempty unpadded source identity")
    if type(synthetic) is not bool:
        raise ValueError("synthetic must be an explicit bool")


@dataclass(frozen=True)
class DiffusionConfig:
    n_steps: int = 32
    hidden_width: int = 64
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 0.001
    seed: int = 42
    clip_denoised: bool = True

    def __post_init__(self) -> None:
        for name, minimum, maximum in (
            ("n_steps", 2, 256),
            ("hidden_width", 4, 512),
            ("epochs", 1, 2_000),
            ("batch_size", 1, 1_024),
            ("seed", 0, 2**63 - 1),
        ):
            object.__setattr__(self, name, _count(getattr(self, name), name, minimum, maximum))
        if not math.isfinite(self.learning_rate) or not 0 < self.learning_rate <= 0.1:
            raise ValueError("learning_rate must be finite and in (0, 0.1]")
        if type(self.clip_denoised) is not bool:
            raise ValueError("clip_denoised must be an explicit bool")


def cosine_betas(n_steps: int) -> Array:
    """Nichol–Dhariwal Eq. 16 discretization; no learned variance claim."""
    n = _count(n_steps, "n_steps", 2, 256)
    times = np.arange(n + 1, dtype=np.float64) / n
    f = np.cos((times + 0.008) / 1.008 * np.pi / 2.0) ** 2
    return _readonly(np.minimum(1.0 - f[1:] / f[:-1], 0.999))


@dataclass(frozen=True)
class DiffusionFitInfo:
    training_data_sha256: str
    model_sha256: str
    asset_ids: tuple[str, ...]
    cutoff: str
    max_training_available_time: str
    n_training_paths: int
    n_excluded_paths: int
    horizon: int
    data_source: str
    training_synthetic: bool
    epoch_losses: tuple[float, ...]
    training_probe_initial_loss: float
    training_probe_final_loss: float
    research_only: bool = field(default=True, init=False)
    live_pnl_claim: bool = field(default=False, init=False)


@dataclass(frozen=True)
class ScenarioBatch:
    paths: Array
    asset_ids: tuple[str, ...]
    model_sha256: str
    training_data_sha256: str
    training_cutoff: str
    seed: int
    training_data_source: str
    training_synthetic: bool
    label: str = field(default="SYNTHETIC", init=False)
    synthetic: bool = field(default=True, init=False)
    research_only: bool = field(default=True, init=False)
    live_pnl_claim: bool = field(default=False, init=False)
    market_evidence: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        arr = _paths(self.paths, "scenario paths")
        _ids(self.asset_ids, arr.shape[2])
        object.__setattr__(self, "paths", _readonly(arr))


@dataclass(frozen=True)
class _Params:
    mean: Array
    scale: Array
    lower: Array
    upper: Array
    layers: tuple[tuple[Array, Array], ...]
    betas: Array
    horizon: int
    n_assets: int


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ImportError("joint path DDPM needs the optional nn extra (torch)") from exc
    return torch


def _finite_loss(value: float) -> float:
    if not math.isfinite(value):
        raise FloatingPointError("non-finite DDPM training loss; refusing fitted artifact")
    return value


def _train(
    standardized: Array, betas: Array, config: DiffusionConfig
) -> tuple[tuple[tuple[Array, Array], ...], tuple[float, ...], float, float]:
    torch = _torch()
    flat = standardized.reshape(standardized.shape[0], -1)
    dimension = flat.shape[1]
    previous_threads = torch.get_num_threads()
    previous_determinism = torch.are_deterministic_algorithms_enabled()
    losses: list[float] = []
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.seed)
            model = torch.nn.Sequential(
                torch.nn.Linear(dimension + config.n_steps, config.hidden_width),
                torch.nn.Tanh(),
                torch.nn.Linear(config.hidden_width, config.hidden_width),
                torch.nn.Tanh(),
                torch.nn.Linear(config.hidden_width, dimension),
            ).double()
            optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
            data = torch.tensor(flat, dtype=torch.float64)
            alpha_bar = torch.tensor(np.cumprod(1.0 - betas), dtype=torch.float64)
            one_hot = torch.eye(config.n_steps, dtype=torch.float64)

            def noisy(rows: Any, times: Any, noise: Any) -> Any:
                ab = alpha_bar[times].reshape(-1, 1)
                xt = ab.sqrt() * rows + (1 - ab).sqrt() * noise
                return torch.cat((xt, one_hot[times]), dim=1)

            probe_rows = data[: min(len(data), 64)]
            probe_times = torch.randint(config.n_steps, (len(probe_rows),))
            probe_noise = torch.randn_like(probe_rows)
            probe_input = noisy(probe_rows, probe_times, probe_noise)
            with torch.no_grad():
                initial = _finite_loss(float(((model(probe_input) - probe_noise) ** 2).mean()))
            for _ in range(config.epochs):
                permutation = torch.randperm(len(data))
                loss_sum = 0.0
                for start in range(0, len(data), config.batch_size):
                    rows = data[permutation[start : start + config.batch_size]]
                    times = torch.randint(config.n_steps, (len(rows),))
                    noise = torch.randn_like(rows)
                    loss = ((model(noisy(rows, times, noise)) - noise) ** 2).mean()
                    value = _finite_loss(float(loss.detach()))
                    optimizer.zero_grad()
                    loss.backward()
                    if any(not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                        raise FloatingPointError("non-finite DDPM gradients")
                    optimizer.step()
                    if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
                        raise FloatingPointError("non-finite DDPM parameters")
                    loss_sum += value * len(rows)
                losses.append(_finite_loss(loss_sum / len(data)))
            with torch.no_grad():
                final = _finite_loss(float(((model(probe_input) - probe_noise) ** 2).mean()))
            layers = tuple(
                (_readonly(layer.weight.detach().numpy()), _readonly(layer.bias.detach().numpy()))
                for layer in model
                if isinstance(layer, torch.nn.Linear)
            )
    finally:
        torch.set_num_threads(previous_threads)
        torch.use_deterministic_algorithms(previous_determinism)
    return layers, tuple(losses), initial, final


def _hash_arrays(metadata: dict[str, Any], arrays: Sequence[Array]) -> str:
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True, allow_nan=False).encode())
    for array in arrays:
        arr = np.ascontiguousarray(array, dtype="<f8")
        digest.update(json.dumps(arr.shape).encode())
        digest.update(arr.tobytes())
    return digest.hexdigest()


def _model_hash(params: _Params, config: DiffusionConfig, training_hash: str) -> str:
    return _hash_arrays(
        {
            "revision": _REVISION,
            "config": asdict(config),
            "training_data": training_hash,
            "horizon": params.horizon,
            "n_assets": params.n_assets,
        },
        [params.mean, params.scale, params.lower, params.upper, params.betas]
        + [x for layer in params.layers for x in layer],
    )


class MarketDiffusion:
    """Train a joint path denoiser; generate labeled synthetic return paths."""

    def __init__(self, config: DiffusionConfig | None = None) -> None:
        self.config = config or DiffusionConfig()
        self._params: _Params | None = None
        self.fit_info: DiffusionFitInfo | None = None
        self._fit_config: DiffusionConfig | None = None
        self._sealed_fit_info: DiffusionFitInfo | None = None

    def fit(
        self,
        paths: Array,
        *,
        asset_ids: Sequence[str],
        timestamps: ClockGrid,
        available_times: ClockGrid,
        cutoff: datetime,
        data_source: str,
        synthetic: bool,
    ) -> MarketDiffusion:
        """Use only complete windows available by cutoff; never fit on suffix."""
        self._params, self.fit_info = None, None  # failed refits cannot serve an old model
        self._sealed_fit_info = None
        arr = _paths(paths, "paths", minimum=8)
        ids = _ids(asset_ids, arr.shape[2])
        events, ready = _clocks(timestamps, available_times, (arr.shape[0], arr.shape[1]))
        stop = _clock(cutoff, "cutoff")
        _provenance(data_source, synthetic)
        indices = [i for i, row in enumerate(ready) if max(row) <= stop]
        if len(indices) < 8:
            raise ValueError("at least eight complete paths must be available by cutoff")
        train = np.array(arr[indices], copy=True)
        if math.ceil(len(train) / self.config.batch_size) * self.config.epochs > _MAX_UPDATES:
            raise ValueError("training exceeds the bounded optimizer update budget")
        mean = np.mean(train, axis=(0, 1))
        scale = np.std(train, axis=(0, 1))
        if not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 1e-12):
            raise ValueError("training asset scales must be finite and nonzero")
        standardized = (train - mean) / scale
        data_hash = _hash_arrays(
            {
                "asset_ids": ids,
                "event_times": [[t.isoformat() for t in events[i]] for i in indices],
                "available_times": [[t.isoformat() for t in ready[i]] for i in indices],
                "cutoff": stop.isoformat(),
                "data_source": data_source,
                "synthetic": synthetic,
            },
            [train],
        )
        betas = cosine_betas(self.config.n_steps)
        layers, losses, initial, final = _train(standardized, betas, self.config)
        params = _Params(
            _readonly(mean),
            _readonly(scale),
            _readonly(standardized.min(axis=0).reshape(-1)),
            _readonly(standardized.max(axis=0).reshape(-1)),
            layers,
            betas,
            arr.shape[1],
            arr.shape[2],
        )
        model_hash = _model_hash(params, self.config, data_hash)
        self._params = params
        self._fit_config = self.config
        self.fit_info = DiffusionFitInfo(
            data_hash,
            model_hash,
            ids,
            stop.isoformat(),
            max(t for i in indices for t in ready[i]).isoformat(),
            len(train),
            len(arr) - len(train),
            arr.shape[1],
            data_source,
            synthetic,
            losses,
            initial,
            final,
        )
        self._sealed_fit_info = self.fit_info
        return self

    def _fitted(self) -> tuple[_Params, DiffusionFitInfo]:
        if self._params is None or self.fit_info is None:
            raise RuntimeError("MarketDiffusion must be successfully fitted")
        if self.config != self._fit_config:
            raise RuntimeError("config changed after fit; retraining is required")
        if self.fit_info != self._sealed_fit_info:
            raise RuntimeError("fit provenance changed after fit; retraining is required")
        if (
            _model_hash(self._params, self.config, self.fit_info.training_data_sha256)
            != self.fit_info.model_sha256
        ):
            raise RuntimeError("model parameters changed after fit; retraining is required")
        return self._params, self.fit_info

    def predict_noise(self, noisy_paths: Array, step: int) -> Array:
        """Joint denoiser inference with numpy weights; step is zero-based."""
        params, _ = self._fitted()
        arr = _paths(noisy_paths, "noisy_paths")
        if arr.shape[1:] != (params.horizon, params.n_assets):
            raise ValueError("noisy path axes do not match the fitted model")
        if (
            len(arr)
            * max(params.horizon * params.n_assets + self.config.n_steps, self.config.hidden_width)
            > _MAX_CELLS
        ):
            raise ValueError("denoising exceeds the bounded activation budget")
        t = _count(step, "step", 0, self.config.n_steps - 1)
        time_features = np.zeros((len(arr), self.config.n_steps))
        time_features[:, t] = 1
        x = np.concatenate((arr.reshape(len(arr), -1), time_features), axis=1)
        for i, (weights, bias) in enumerate(params.layers):
            x = x @ weights.T + bias
            if i + 1 < len(params.layers):
                x = np.tanh(x)
        if not np.isfinite(x).all():
            raise FloatingPointError("non-finite joint denoiser output")
        return np.asarray(x.reshape(arr.shape), dtype=np.float64)

    def sample_scenarios(self, n: int, *, seed: int = 0) -> ScenarioBatch:
        params, info = self._fitted()
        count = _count(n, "n", 1, 100_000)
        random_seed = _count(seed, "seed", 0, 2**63 - 1)
        if (
            count
            * max(params.horizon * params.n_assets + self.config.n_steps, self.config.hidden_width)
            > _MAX_CELLS
        ):
            raise ValueError("sampling exceeds the bounded path resource budget")
        rng = np.random.default_rng(random_seed)
        x = rng.normal(size=(count, params.horizon, params.n_assets))
        alpha = 1.0 - params.betas
        alpha_bar = np.cumprod(alpha)
        previous = np.concatenate(([1.0], alpha_bar[:-1]))
        for t in range(self.config.n_steps - 1, -1, -1):
            eps = self.predict_noise(x, t)
            x0 = (x - math.sqrt(1 - alpha_bar[t]) * eps) / math.sqrt(alpha_bar[t])
            if self.config.clip_denoised:
                x0 = np.clip(x0.reshape(count, -1), params.lower, params.upper).reshape(x.shape)
            # Ho Eq. 7 posterior mean and fixed-small variance. t=0 has no noise.
            c0 = params.betas[t] * math.sqrt(previous[t]) / (1 - alpha_bar[t])
            ct = math.sqrt(alpha[t]) * (1 - previous[t]) / (1 - alpha_bar[t])
            x = c0 * x0 + ct * x
            if t > 0:
                variance = params.betas[t] * (1 - previous[t]) / (1 - alpha_bar[t])
                x += math.sqrt(variance) * rng.normal(size=x.shape)
            if not np.isfinite(x).all():
                raise FloatingPointError("non-finite reverse diffusion path")
        raw = x * params.scale + params.mean
        return ScenarioBatch(
            raw,
            info.asset_ids,
            info.model_sha256,
            info.training_data_sha256,
            info.cutoff,
            random_seed,
            info.data_source,
            info.training_synthetic,
        )

    def diagnostics(
        self,
        heldout_paths: Array,
        *,
        asset_ids: Sequence[str],
        timestamps: ClockGrid,
        available_times: ClockGrid,
        data_source: str,
        synthetic: bool,
        n_scenarios: int = 512,
        seed: int = 0,
    ) -> dict[str, Any]:
        """Require post-cutoff nonoverlapping windows; report gaps, never pass."""
        params, info = self._fitted()
        arr = _paths(heldout_paths, "heldout_paths", minimum=2)
        if arr.shape[1:] != (params.horizon, params.n_assets):
            raise ValueError("heldout path axes do not match the fitted model")
        if _ids(asset_ids, arr.shape[2]) != info.asset_ids:
            raise ValueError("heldout asset order must match fitted asset_ids")
        events, ready = _clocks(timestamps, available_times, (arr.shape[0], arr.shape[1]))
        _provenance(data_source, synthetic)
        stop = datetime.fromisoformat(info.cutoff)
        if any(min(row) <= stop for row in events):
            raise ValueError("heldout windows must be entirely after the training cutoff")
        batch = self.sample_scenarios(n_scenarios, seed=seed)
        result = scenario_diagnostics(batch.paths, arr)
        result.update(
            {
                "model_sha256": info.model_sha256,
                "training_data_sha256": info.training_data_sha256,
                "heldout_sha256": _hash_arrays(
                    {
                        "asset_ids": info.asset_ids,
                        "events": [[t.isoformat() for t in r] for r in events],
                        "available_times": [[t.isoformat() for t in r] for r in ready],
                        "data_source": data_source,
                        "synthetic": synthetic,
                    },
                    [arr],
                ),
                "heldout_data_source": data_source,
                "heldout_synthetic": synthetic,
                "generated_label": "SYNTHETIC",
                "denoised_training_support_clipping": self.config.clip_denoised,
            }
        )
        return result


def train_diffusion(
    paths: Array,
    *,
    asset_ids: Sequence[str],
    timestamps: ClockGrid,
    available_times: ClockGrid,
    cutoff: datetime,
    data_source: str,
    synthetic: bool,
    config: DiffusionConfig | None = None,
) -> MarketDiffusion:
    return MarketDiffusion(config).fit(
        paths,
        asset_ids=asset_ids,
        timestamps=timestamps,
        available_times=available_times,
        cutoff=cutoff,
        data_source=data_source,
        synthetic=synthetic,
    )


def scenario_diagnostics(generated: Array, reference: Array) -> dict[str, Any]:
    """Mean, asset covariance, lag correlation and tail-quantile gaps.

    This low-level comparison does not establish that reference is a holdout.
    Use ``MarketDiffusion.diagnostics`` for the explicit temporal contract.
    Constant series have undefined lag correlation, reported as None rather
    than a fabricated zero or non-finite headline.
    """
    fake, real = _paths(generated, "generated", 2), _paths(reference, "reference", 2)
    if fake.shape[1:] != real.shape[1:]:
        raise ValueError("generated/reference horizon and asset axes must match")
    ff, rr = fake.reshape(-1, fake.shape[2]), real.reshape(-1, real.shape[2])
    mean_gap = np.abs(ff.mean(axis=0) - rr.mean(axis=0))
    covariance_gap = np.abs(np.cov(ff, rowvar=False) - np.cov(rr, rowvar=False))
    taus = (0.01, 0.05, 0.95, 0.99)
    tail_gap = np.abs(np.quantile(ff, taus, axis=0) - np.quantile(rr, taus, axis=0))

    def lag_correlation(paths: Array) -> list[float | None]:
        values: list[float | None] = []
        for a in range(paths.shape[2]):
            left, right = paths[:, :-1, a].ravel(), paths[:, 1:, a].ravel()
            if left.std() <= 1e-12 or right.std() <= 1e-12:
                values.append(None)
            else:
                values.append(float(np.corrcoef(left, right)[0, 1]))
        return values

    fake_lag, real_lag = lag_correlation(fake), lag_correlation(real)
    lag_gap = [
        abs(f - r) if f is not None and r is not None else None
        for f, r in zip(fake_lag, real_lag, strict=True)
    ]
    for values in (mean_gap, covariance_gap, tail_gap):
        if not np.isfinite(values).all():
            raise FloatingPointError("non-finite scenario diagnostics")
    return {
        "mean_abs_gap_by_asset": mean_gap.tolist(),
        "covariance_abs_gap": covariance_gap.tolist(),
        "lag1_correlation_abs_gap_by_asset": lag_gap,
        "tail_quantiles": list(taus),
        "tail_quantile_abs_gap": tail_gap.tolist(),
        "n_generated": len(fake),
        "n_reference": len(real),
        "status": "descriptive_distribution_gaps",
        "synthetic_generated": True,
        "research_only": True,
        "market_evidence": False,
        "live_pnl_claim": False,
        "sota_established": False,
    }
