"""SigCWGAN-lite: signature-based generative modeling of financial time series.

A minimal, seeded generative model whose training signal is the *expected
signature* of the path distribution: the generator is fit so that the mean
truncated signature of generated paths matches that of the real paths. No
neural critic is trained — the discriminator is the closed-form linear
witness on the signature feature space (see below), which is what makes this
a *lite* variant.

References (fetch-verified 2026-10):

- Liao, S., Ni, H., Szpruch, L., Wiese, M., Sabate-Vidales, M. & Xiao, B.
  (2023), "Conditional Sig-Wasserstein GANs for Time Series Generation",
  arXiv:2006.05421, https://arxiv.org/abs/2006.05421 (the Sig-WGAN framework:
  WGAN + signature feature extraction; the conditional Sig-W1 metric restricts
  the Wasserstein-1 critic to linear functionals of the signature of the
  augmented path, so the optimal discriminator is the *explicit* direction
  E[Sig(X_real)] - E[Sig(X_gen)] and no critic network needs training. This
  module implements the unconditional limit of that framework — no
  past-conditioning window (p = 0) and no lead-lag or invisibility-reset
  augmentation, only time augmentation — i.e. "SigCWGAN-lite". The generator
  is a residual GRU consuming Gaussian noise per step, mirroring the paper's
  autoregressive residual generator, Alg. 1).
- Chevyrev, I. & Oberhauser, H. (2022), "Signature moments to characterize
  laws of stochastic processes", arXiv:1810.10971,
  https://arxiv.org/abs/1810.10971, JMLR 23 (the signature-moment MMD:
  under regularity the expected signature characterizes the law of a
  stochastic process, and the induced metric is kernelizable via the
  signature kernel — the basis for both the training loss and the
  post-training signature-moment distance quality gate).
- Perez Arribas, I., Salvi, C. & Szpruch, L. (2020), "Sig-SDEs model for
  quantitative finance", arXiv:2006.00218, https://arxiv.org/abs/2006.00218,
  ACM ICAIF 2020 (expected-signature matching as the training objective for a
  generative SDE; the same loss is used here with a GRU generator in place of
  their signature-drift vector field).

Loss and quality gate. The loss is the level-normalized squared
signature-moment distance

    L(theta) = sum_k ||m_k(gen) - m_k(real)||^2 / max(1, ||m_k(real)||^2),

where m_k is the mean level-k signature over the batch — the squared norm of
the Sig-W1 linear witness direction, up to per-level rescaling (the scaling
keeps the objective scale-free; a pure ||.||^2 is recovered by setting the
normalizers to 1). The reported quality-gate distance is sqrt(L) computed
with the canonical ``quant_fund.models.path_signatures.signature`` (lazy
import — that module is NOT edited and NOT required at import time beyond
numpy/scipy). Per-coordinate marginal two-sample KS on pooled increments and
a per-coordinate ACF distance on increments complete the gate.

Honesty: all quantities are computed on SYNTHETIC seeded paths (the built-in
GBM/OU samplers) or caller-provided arrays. Outputs are distributional
distances (signature-moment distance, KS statistic, ACF distance) —
correctness/similarity evidence only, never market evidence; no
Sharpe/Sortino/P&L headline and no live-trading claims (AGENTS.md honesty
contract). Generated paths are research artifacts, not forecasts.

Composition: reuses ``quant_fund.models.path_signatures`` (lazy import) for
the canonical signature in the quality gate; complements the sibling
generative-model lanes ``models/gslice.py``, ``models/kit_paths.py`` and
``models/neural_sde.py`` (same lazy-torch + fail-closed conventions; they are
not edited here). numpy/scipy + lazy torch only; no new dependencies.

Conventions: torch is the optional ``nn`` extra, imported lazily via
:func:`_torch`; when it is absent the module still imports and every entry
point runs through a deterministic numpy fallback (a residual affine-AR
generator trained by Nelder-Mead on the same signature-moment objective —
a degraded, documented substitute, not the GRU). ``backend="torch"``
explicitly requested without torch fails closed with ``ImportError`` and
install guidance. Inputs fail closed with ``ValueError``; a non-finite
training loss raises ``ArithmeticError``. Determinism: ``torch.manual_seed``
plus seeded ``torch.Generator`` noise for the torch lane and
``numpy.random.default_rng``/fixed common-random-number batches for the
numpy lane; same seed -> identical generated paths (GPU determinism is not
claimed — training runs CPU float64 single-thread).
"""

from __future__ import annotations

import importlib.util
import math
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize as _minimize
from scipy.stats import ks_2samp as _ks_2samp

Array = NDArray[np.float64]

__all__ = [
    "SigGANConfig",
    "SigGANResult",
    "acf_distance",
    "bench_sig_gan",
    "expected_signature",
    "fit_sig_gan",
    "generate_paths",
    "marginal_ks_distance",
    "sample_gbm_paths",
    "sample_ou_paths",
    "signature_moment_distance",
    "torch_backend_available",
]

_BACKENDS = ("auto", "torch", "numpy")
_MAX_ORDER = 6  # matches path_signatures.signature's truncation bound
_MAX_SIG_FEATURES = 50_000  # fail closed before tensor layout explodes


def _torch() -> Any:
    """Lazily import torch (optional ``nn`` extra); fail closed with guidance."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "the torch SigCWGAN-lite backend needs the optional 'nn' extra: "
            "uv sync --extra nn (backend='auto' falls back to numpy without it)"
        ) from exc
    return torch


def torch_backend_available() -> bool:
    """True when the optional ``nn`` extra (torch) is importable."""
    try:
        return importlib.util.find_spec("torch") is not None
    except (ImportError, ValueError):  # blocked or halted torch imports
        return False


def _signatures() -> Any:
    """Lazily import the canonical signature module (never edited here)."""
    from quant_fund.models import path_signatures

    return path_signatures


def _as_paths(paths: Array | Iterable[Iterable[Iterable[float]]], name: str = "paths") -> Array:
    """Validate a path batch to a finite float64 (n_paths, T+1, d) array."""
    arr = np.asarray(paths, dtype=float)
    if arr.ndim != 3:
        raise ValueError(
            f"{name} must be a 3-D array of shape (n_paths, T+1, d); got ndim={arr.ndim}"
        )
    if arr.shape[0] < 2:
        raise ValueError(f"{name} must contain at least 2 paths; got {arr.shape[0]}")
    if arr.shape[1] < 2:
        raise ValueError(f"{name} must have at least 2 points per path; got {arr.shape[1]}")
    if arr.shape[2] < 1:
        raise ValueError(f"{name} must have at least 1 channel; got {arr.shape[2]}")
    if not bool(np.all(np.isfinite(arr))):
        raise ValueError(f"{name} entries must be finite (NaN/inf rejected)")
    return arr


def _check_order(order: int) -> int:
    m = int(order)
    if m != order or m < 1 or m > _MAX_ORDER:
        raise ValueError(f"order must be an integer in [1, {_MAX_ORDER}]; got {order!r}")
    return m


def _check_dims_fit(dims: int, order: int) -> None:
    total = sum(dims**k for k in range(1, order + 1))
    if total > _MAX_SIG_FEATURES:
        raise ValueError(f"signature too large: d_aug={dims}, order={order} gives {total} features")


def _check_config(config: SigGANConfig) -> None:
    _check_order(config.order)
    if config.backend not in _BACKENDS:
        raise ValueError(f"backend must be one of {_BACKENDS}; got {config.backend!r}")
    for name, val in (
        ("noise_dim", config.noise_dim),
        ("hidden_dim", config.hidden_dim),
        ("iterations", config.iterations),
        ("batch_size", config.batch_size),
        ("patience", config.patience),
    ):
        if int(val) != val or int(val) < 1:
            raise ValueError(f"{name} must be a positive integer; got {val!r}")
    if config.iterations > 300:
        raise ValueError(f"iterations must be <= 300 (lite budget); got {config.iterations}")
    if not math.isfinite(config.lr) or config.lr <= 0.0:
        raise ValueError(f"lr must be positive and finite; got {config.lr!r}")
    if not math.isfinite(config.tol) or config.tol < 0.0:
        raise ValueError(f"tol must be finite and >= 0; got {config.tol!r}")


def _time_augment(paths: Array) -> Array:
    """Append the normalized time channel t/T in [0, 1] (SigCWGAN augmentation)."""
    n, t1, _ = paths.shape
    time = np.linspace(0.0, 1.0, t1)
    grid = np.broadcast_to(time[None, :, None], (n, t1, 1))
    return np.concatenate([grid, paths], axis=2)


def _bouter(a: Array, b: Array) -> Array:
    """Batched tensor outer product: (n, *A) x (n, *B) -> (n, *A, *B)."""
    n = a.shape[0]
    out = a.reshape(n, -1, 1) * b.reshape(n, 1, -1)
    return out.reshape(n, *a.shape[1:], *b.shape[1:])


def _signature_levels_batch(paths: Array, order: int) -> list[Array]:
    """Batched truncated-signature levels, mirroring ``path_signatures.signature``.

    Same Chen-update math (S <- S x exp(dx), levels 1..order) vectorized over
    the batch axis so the numpy fallback objective stays fast; the canonical
    per-path ``path_signatures.signature`` remains the quality-gate reference
    (a unit test pins the two to identical values).
    """
    n, t1, d = paths.shape
    increments = np.diff(paths, axis=1)
    levels: list[Array] = [np.zeros((n,) + (d,) * k, dtype=float) for k in range(1, order + 1)]
    for t in range(t1 - 1):
        dx = increments[:, t, :]
        powers: list[Array] = [dx]
        for j in range(2, order + 1):
            powers.append(_bouter(powers[-1], dx) / float(j))
        for k in range(order, 0, -1):
            acc = powers[k - 1].copy()
            for j in range(1, k):
                acc = acc + _bouter(levels[k - j - 1], powers[j - 1])
            levels[k - 1] = levels[k - 1] + acc
    return levels


def _level_weights(levels_ref: list[Array]) -> list[float]:
    """Per-level normalizers max(1, ||m_k(ref)||^2) for the moment distance."""
    return [
        max(1.0, float(np.sum(level.reshape(level.shape[0], -1).mean(axis=0) ** 2)))
        for level in levels_ref
    ]


def _moment_sq(levels_a: list[Array], levels_b: list[Array]) -> float:
    """Level-normalized squared signature-moment distance between two batches.

    levels_* are per-path level tensors (n, d^k); their batch means m_k are
    compared as sum_k ||m_k(a) - m_k(b)||^2 / max(1, ||m_k(b)||^2), with b the
    reference batch. The reported distance is sqrt of this sum.
    """
    total = 0.0
    for la, lb, w in zip(levels_a, levels_b, _level_weights(levels_b), strict=True):
        ma = la.reshape(la.shape[0], -1).mean(axis=0)
        mb = lb.reshape(lb.shape[0], -1).mean(axis=0)
        total += float(np.sum((ma - mb) ** 2)) / w
    return total


def expected_signature(
    paths: Array | Iterable[Iterable[Iterable[float]]],
    order: int = 3,
    *,
    time_augment: bool = True,
) -> Array:
    """Expected truncated signature: batch mean of per-path flat signatures.

    Uses the canonical ``quant_fund.models.path_signatures.signature`` (lazy
    import). With ``time_augment`` the signature is taken of the
    time-augmented path (t/T, x), the SigCWGAN default (arXiv:2006.05421 §3).
    Levels 1..order are concatenated in word order, level 0 omitted.
    """
    sig = _signatures()
    arr = _as_paths(paths)
    m = _check_order(order)
    aug = _time_augment(arr) if time_augment else arr
    _check_dims_fit(int(aug.shape[2]), m)
    sigs = np.stack([sig.signature(aug[i], m) for i in range(aug.shape[0])])
    return np.asarray(sigs.mean(axis=0), dtype=float)


def _expected_levels(paths: Array, order: int, time_augment: bool) -> list[Array]:
    """Batch-mean signature as a list of level tensors (numpy, batch mirror)."""
    aug = _time_augment(paths) if time_augment else paths
    _check_dims_fit(int(aug.shape[2]), order)
    levels = _signature_levels_batch(aug, order)
    return [np.asarray(level, dtype=float) for level in levels]


def signature_moment_distance(
    real: Array | Iterable[Iterable[Iterable[float]]],
    generated: Array | Iterable[Iterable[Iterable[float]]],
    order: int = 3,
    *,
    time_augment: bool = True,
) -> float:
    """Signature-moment distance sqrt(L) between two path batches (quality gate).

    Per-path signatures come from the canonical ``path_signatures.signature``
    (lazy import); the distance is the level-normalized quadratic of the
    Chevyrev-Oberhauser signature-moment MMD restricted to its linear witness
    (arXiv:1810.10971; arXiv:2006.05421 §3). Both batches must share the
    channel count; path counts and lengths may differ.
    """
    sig = _signatures()
    arr_r = _as_paths(real, "real")
    arr_g = _as_paths(generated, "generated")
    if int(arr_r.shape[2]) != int(arr_g.shape[2]):
        raise ValueError(
            f"real and generated must share channel count; got {arr_r.shape[2]} and {arr_g.shape[2]}"
        )
    m = _check_order(order)
    aug_r = _time_augment(arr_r) if time_augment else arr_r
    aug_g = _time_augment(arr_g) if time_augment else arr_g
    _check_dims_fit(int(aug_r.shape[2]), m)
    levels_r = np.stack([sig.signature(aug_r[i], m) for i in range(aug_r.shape[0])])
    levels_g = np.stack([sig.signature(aug_g[i], m) for i in range(aug_g.shape[0])])
    dims_aug = int(aug_r.shape[2])
    offsets = np.cumsum([dims_aug**k for k in range(1, m + 1)])
    total = 0.0
    for k in range(m):
        lo = 0 if k == 0 else int(offsets[k - 1])
        hi = int(offsets[k])
        mr = levels_r[:, lo:hi].mean(axis=0)
        mg = levels_g[:, lo:hi].mean(axis=0)
        total += float(np.sum((mr - mg) ** 2)) / max(1.0, float(np.sum(mr**2)))
    return float(math.sqrt(total))


def marginal_ks_distance(
    real: Array | Iterable[Iterable[Iterable[float]]],
    generated: Array | Iterable[Iterable[Iterable[float]]],
) -> float:
    """Mean per-coordinate two-sample KS statistic on pooled increments.

    For each channel the real and generated increments are pooled over time
    and paths and compared with ``scipy.stats.ks_2samp``; the channel mean is
    returned (0 = identical empirical marginals).
    """
    arr_r = _as_paths(real, "real")
    arr_g = _as_paths(generated, "generated")
    if int(arr_r.shape[2]) != int(arr_g.shape[2]):
        raise ValueError(
            f"real and generated must share channel count; got {arr_r.shape[2]} and {arr_g.shape[2]}"
        )
    inc_r = np.diff(arr_r, axis=1)
    inc_g = np.diff(arr_g, axis=1)
    stats = [
        float(_ks_2samp(inc_r[:, :, j].ravel(), inc_g[:, :, j].ravel()).statistic)
        for j in range(arr_r.shape[2])
    ]
    return float(np.mean(stats))


def _acf(paths: Array, max_lag: int) -> Array:
    """Per-path autocorrelation of increments at lags 1..max_lag, (n, d, lag)."""
    inc = np.diff(paths, axis=1)  # (n, T, d)
    centered = inc - inc.mean(axis=1, keepdims=True)
    denom = np.sum(centered**2, axis=1)  # (n, d)
    out = np.zeros((inc.shape[0], inc.shape[2], max_lag), dtype=float)
    for lag in range(1, max_lag + 1):
        num = np.sum(centered[:, :-lag, :] * centered[:, lag:, :], axis=1)
        np.divide(num, denom, out=out[:, :, lag - 1], where=denom > 0.0)
    return out


def acf_distance(
    real: Array | Iterable[Iterable[Iterable[float]]],
    generated: Array | Iterable[Iterable[Iterable[float]]],
    max_lag: int = 5,
) -> float:
    """Mean |Δ ACF| over channels and lags 1..max_lag, on increments.

    ACFs are computed per path and averaged over paths before comparing —
    captures lag-dependence structure (mean reversion, vol clustering sign)
    that the signature loss only partially constrains.
    """
    arr_r = _as_paths(real, "real")
    arr_g = _as_paths(generated, "generated")
    if int(arr_r.shape[2]) != int(arr_g.shape[2]):
        raise ValueError(
            f"real and generated must share channel count; got {arr_r.shape[2]} and {arr_g.shape[2]}"
        )
    lag = int(max_lag)
    if lag != max_lag or lag < 1:
        raise ValueError(f"max_lag must be a positive integer; got {max_lag!r}")
    min_t = min(int(arr_r.shape[1]), int(arr_g.shape[1]))
    if lag >= min_t:
        raise ValueError(f"max_lag must be < path length; got {lag} >= {min_t}")
    return float(np.mean(np.abs(_acf(arr_r, lag).mean(axis=0) - _acf(arr_g, lag).mean(axis=0))))


def sample_gbm_paths(
    n_paths: int,
    steps: int,
    dims: int,
    *,
    drift: float | Iterable[float] = 0.05,
    vol: float | Iterable[float] = 0.25,
    rho: float = 0.0,
    s0: float = 1.0,
    horizon: float = 1.0,
    seed: int = 0,
) -> Array:
    """SYNTHETIC correlated GBM paths, log-Euler exact discretization.

    dlog S_i = (mu_i - sigma_i^2/2) dt + sigma_i dW_i with equicorrelation
    rho; returns (n_paths, steps+1, dims) price paths starting at s0.
    Correctness fixtures only — never market evidence.
    """
    n = _positive_int(n_paths, "n_paths")
    t = _positive_int(steps, "steps")
    d = _positive_int(dims, "dims")
    mu = _as_vector(drift, d, "drift")
    sigma = _as_vector(vol, d, "vol")
    if np.any(sigma <= 0.0) or not bool(np.all(np.isfinite(sigma))):
        raise ValueError(f"vol entries must be positive and finite; got {vol!r}")
    if not math.isfinite(rho) or not -1.0 < rho < 1.0:
        raise ValueError(f"rho must be in (-1, 1); got {rho!r}")
    if not math.isfinite(s0) or s0 <= 0.0:
        raise ValueError(f"s0 must be positive and finite; got {s0!r}")
    dt = _check_horizon(horizon) / t
    rng = np.random.default_rng(int(seed))
    corr = np.full((d, d), float(rho)) + (1.0 - float(rho)) * np.eye(d)
    chol = np.linalg.cholesky(corr)
    dw = rng.standard_normal((n, t, d)) @ chol.T * math.sqrt(dt)
    drift_step = (mu - 0.5 * sigma**2) * dt
    log_paths = np.concatenate(
        [
            np.zeros((n, 1, d)),
            np.cumsum(drift_step[None, None, :] + dw * sigma[None, None, :], axis=1),
        ],
        axis=1,
    )
    return np.asarray(s0 * np.exp(log_paths), dtype=float)


def sample_ou_paths(
    n_paths: int,
    steps: int,
    dims: int,
    *,
    theta: float | Iterable[float] = 1.0,
    mean: float | Iterable[float] = 0.0,
    vol: float | Iterable[float] = 0.2,
    x0: float | Iterable[float] | None = None,
    rho: float = 0.0,
    horizon: float = 1.0,
    seed: int = 0,
) -> Array:
    """SYNTHETIC Ornstein-Uhlenbeck paths, Euler-Maruyama discretization.

    dX_i = theta_i (mean_i - X_i) dt + vol_i dW_i, equicorrelated drivers;
    returns (n_paths, steps+1, dims). Defaults x0 to the stationary mean.
    Correctness fixtures only — never market evidence.
    """
    n = _positive_int(n_paths, "n_paths")
    t = _positive_int(steps, "steps")
    d = _positive_int(dims, "dims")
    th = _as_vector(theta, d, "theta")
    mu = _as_vector(mean, d, "mean")
    sigma = _as_vector(vol, d, "vol")
    if np.any(th <= 0.0):
        raise ValueError(f"theta entries must be positive; got {theta!r}")
    if np.any(sigma <= 0.0):
        raise ValueError(f"vol entries must be positive; got {vol!r}")
    if not math.isfinite(rho) or not -1.0 < rho < 1.0:
        raise ValueError(f"rho must be in (-1, 1); got {rho!r}")
    dt = _check_horizon(horizon) / t
    start = mu if x0 is None else _as_vector(x0, d, "x0")
    rng = np.random.default_rng(int(seed))
    corr = np.full((d, d), float(rho)) + (1.0 - float(rho)) * np.eye(d)
    chol = np.linalg.cholesky(corr)
    dw = rng.standard_normal((n, t, d)) @ chol.T * math.sqrt(dt)
    paths = np.empty((n, t + 1, d), dtype=float)
    paths[:, 0, :] = start[None, :]
    for i in range(t):
        x = paths[:, i, :]
        paths[:, i + 1, :] = x + th[None, :] * (mu[None, :] - x) * dt + sigma[None, :] * dw[:, i, :]
    return paths


def _positive_int(val: int, name: str) -> int:
    n = int(val)
    if n != val or n < 1:
        raise ValueError(f"{name} must be a positive integer; got {val!r}")
    return n


def _as_vector(val: float | Iterable[float], dims: int, name: str) -> Array:
    arr = np.asarray(list(val) if isinstance(val, Iterable) else [val] * dims, dtype=float)
    if arr.shape != (dims,):
        raise ValueError(f"{name} must be a scalar or length-{dims} vector; got shape {arr.shape}")
    if not bool(np.all(np.isfinite(arr))):
        raise ValueError(f"{name} entries must be finite; got {val!r}")
    return arr


def _check_horizon(horizon: float) -> float:
    h = float(horizon)
    if not math.isfinite(h) or h <= 0.0:
        raise ValueError(f"horizon must be positive and finite; got {horizon!r}")
    return h


@dataclass(frozen=True)
class SigGANConfig:
    """Training configuration for :func:`fit_sig_gan` (all seeded, CPU).

    backend: "auto" picks torch when importable else the numpy fallback;
    "torch" forces the GRU lane (ImportError when the ``nn`` extra is
    absent); "numpy" forces the deterministic affine-AR fallback trained by
    Nelder-Mead on the same signature-moment objective.
    """

    noise_dim: int = 4
    hidden_dim: int = 24
    order: int = 3
    iterations: int = 200
    batch_size: int = 64
    lr: float = 3e-3
    tol: float = 1e-7
    patience: int = 50
    time_augment: bool = True
    backend: str = "auto"
    seed: int = 0


@dataclass
class SigGANResult:
    """Fitted generator plus training evidence.

    ``paths`` is the fixed eval batch used for the before/after signature
    distances (common random numbers). ``loss_trace`` records the per-outer-
    iteration objective (torch: batch loss; numpy: Nelder-Mead best-so-far,
    hence non-increasing). ``_model`` is the fitted generator state needed by
    :func:`generate_paths`.
    """

    backend: str
    paths: Array
    loss_trace: tuple[float, ...]
    iterations_run: int
    sig_distance_before: float
    sig_distance_after: float
    n_parameters: int
    config: SigGANConfig
    _model: Any = field(repr=False, compare=False)

    def sample(self, n_paths: int, seed: int = 0) -> Array:
        """Draw n_paths from the fitted generator (seeded; see generate_paths)."""
        return _generate(self._model, self.backend, self.config, n_paths, seed)


def _generate(model: Any, backend: str, config: SigGANConfig, n_paths: int, seed: int) -> Array:
    n = _positive_int(n_paths, "n_paths")
    rng = np.random.default_rng(int(seed))
    if backend == "torch":
        torch = _torch()
        z = torch.as_tensor(
            rng.standard_normal((n, int(model["steps"]), config.noise_dim)),
            dtype=torch.float64,
        )
        with torch.no_grad():
            out = _torch_gen_forward(model, z)
        return np.asarray(out.numpy(), dtype=float)
    z_np = rng.standard_normal((n, model["steps"], config.noise_dim))
    return _numpy_gen_forward(model["theta"], z_np, model["dims"])


def _torch_gen_forward(model: dict[str, Any], z: Any) -> Any:
    """Residual GRU rollout: x_{t+1} = x_t + W_out h_t, h_t = GRUCell(z_t, h_{t-1})."""
    torch = _torch()
    cell = model["cell"]
    out = model["out"]
    x0 = model["x0"]
    bsz, steps, _ = z.shape
    h = torch.zeros(bsz, int(cell.hidden_size), dtype=z.dtype)
    x = x0.unsqueeze(0).expand(bsz, -1)
    columns = [x]
    for t in range(steps):
        h = cell(z[:, t, :], h)
        x = x + out(h)
        columns.append(x)
    return torch.stack(columns, dim=1)


def _torch_outer(a: Any, b: Any) -> Any:
    """Batched tensor outer product: (B, *A) x (B, *B) -> (B, *A, *B)."""
    bsz = a.shape[0]
    return (a.reshape(bsz, -1, 1) * b.reshape(bsz, 1, -1)).reshape(bsz, *a.shape[1:], *b.shape[1:])


def _torch_signature_levels(paths: Any, order: int) -> list[Any]:
    """Differentiable truncated signature levels (same Chen update as numpy)."""
    torch = _torch()
    bsz, t1, d = paths.shape
    increments = paths[:, 1:, :] - paths[:, :-1, :]
    levels = [torch.zeros((bsz,) + (d,) * k, dtype=paths.dtype) for k in range(1, order + 1)]
    for t in range(t1 - 1):
        dx = increments[:, t, :]
        powers = [dx]
        for j in range(2, order + 1):
            powers.append(_torch_outer(powers[-1], dx) / float(j))
        for k in range(order, 0, -1):
            acc = powers[k - 1]
            for j in range(1, k):
                acc = acc + _torch_outer(levels[k - j - 1], powers[j - 1])
            levels[k - 1] = levels[k - 1] + acc
    return levels


def _torch_moment_sq(levels_fake: list[Any], ref_means: list[Any], weights: list[float]) -> Any:
    """Torch twin of :func:`_moment_sq` over mean levels (differentiable)."""
    total = None
    for lf, mr, w in zip(levels_fake, ref_means, weights, strict=True):
        mf = lf.reshape(lf.shape[0], -1).mean(dim=0)
        term = ((mf - mr) ** 2).sum() / w
        total = term if total is None else total + term
    if not (total is not None):
        raise ValueError("total is not None")
    return total


def _fit_torch(paths: Array, config: SigGANConfig) -> SigGANResult:
    """Train the residual-GRU generator on the signature-moment loss."""
    torch = _torch()
    seed = int(config.seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    n_real, t1, d = paths.shape
    steps = t1 - 1
    q = config.noise_dim

    cell = torch.nn.GRUCell(q, config.hidden_dim, dtype=torch.float64)
    out = torch.nn.Linear(config.hidden_dim, d, dtype=torch.float64)
    x0 = torch.nn.Parameter(torch.zeros(d, dtype=torch.float64))
    model: dict[str, Any] = {
        "cell": cell,
        "out": out,
        "x0": x0,
        "steps": steps,
        "dims": d,
    }
    params = [x0, *cell.parameters(), *out.parameters()]
    n_parameters = int(sum(int(p.numel()) for p in params))

    aug_real = _time_augment(paths) if config.time_augment else paths
    with torch.no_grad():
        real_t = torch.as_tensor(aug_real, dtype=torch.float64)
        ref_levels = _torch_signature_levels(real_t, config.order)
        ref_means = [lvl.reshape(n_real, -1).mean(dim=0) for lvl in ref_levels]
        weights = [
            max(1.0, float(((lvl.reshape(n_real, -1).mean(dim=0)) ** 2).sum()))
            for lvl in ref_levels
        ]

    def augment(z_paths: Any) -> Any:
        if not config.time_augment:
            return z_paths
        time = torch.linspace(0.0, 1.0, t1, dtype=z_paths.dtype)
        grid = time.reshape(1, t1, 1).expand(z_paths.shape[0], t1, 1)
        return torch.cat([grid, z_paths], dim=2)

    rng_t = torch.Generator().manual_seed(seed)
    batch = config.batch_size
    z_fixed = torch.randn(batch, steps, q, generator=rng_t, dtype=torch.float64)

    def eval_loss(z: Any) -> Any:
        fake = _torch_gen_forward(model, z)
        levels = _torch_signature_levels(augment(fake), config.order)
        return _torch_moment_sq(levels, ref_means, weights)

    with torch.no_grad():
        before = float(eval_loss(z_fixed))

    opt = torch.optim.Adam(params, lr=config.lr)
    best = math.inf
    best_state: dict[str, Any] | None = None
    stall = 0
    trace: list[float] = []
    for _ in range(config.iterations):
        z = torch.randn(batch, steps, q, generator=rng_t, dtype=torch.float64)
        loss = eval_loss(z)
        val = float(loss.detach())
        if not math.isfinite(val):
            raise ArithmeticError(f"non-finite signature-moment loss {val!r} during training")
        opt.zero_grad()
        loss.backward()
        opt.step()
        trace.append(val)
        if val < best - config.tol:
            best = val
            stall = 0
            best_state = {
                "x0": x0.detach().clone(),
                "cell": {k: v.detach().clone() for k, v in cell.state_dict().items()},
                "out": {k: v.detach().clone() for k, v in out.state_dict().items()},
            }
        else:
            stall += 1
            if stall >= config.patience:
                break
    if best_state is not None:
        with torch.no_grad():
            x0.copy_(best_state["x0"])
            cell.load_state_dict(best_state["cell"])
            out.load_state_dict(best_state["out"])
    with torch.no_grad():
        after = float(eval_loss(z_fixed))
        paths_out = _torch_gen_forward(model, z_fixed).numpy()

    return SigGANResult(
        backend="torch",
        paths=np.asarray(paths_out, dtype=float),
        loss_trace=tuple(trace),
        iterations_run=len(trace),
        sig_distance_before=math.sqrt(before),
        sig_distance_after=math.sqrt(after),
        n_parameters=n_parameters,
        config=config,
        _model=model,
    )


def _unpack_theta(theta: Array, dims: int, noise_dim: int) -> tuple[Array, Array, Array, Array]:
    """theta = concat[x0(d), b(d), a(d), W(d x q)] of the affine-AR fallback."""
    x0 = theta[:dims]
    b = theta[dims : 2 * dims]
    a = theta[2 * dims : 3 * dims]
    w = theta[3 * dims :].reshape(dims, noise_dim)
    return x0, b, a, w


def _numpy_gen_forward(theta: Array, z: Array, dims: int) -> Array:
    """Affine-AR residual path: dx_t = (b - a x_t) + W z_t, x_0 learnable.

    dx_t = b - a * x_t + W z_t matches the Euler OU drift b = theta*mu,
    a = theta and collapses to a correlated Gaussian random walk at a = 0 —
    a deliberately small parametric family for the torch-free fallback.
    """
    bsz, steps, q = z.shape
    x0, b, a, w = _unpack_theta(theta, dims, q)
    paths = np.empty((bsz, steps + 1, dims), dtype=float)
    paths[:, 0, :] = x0[None, :]
    for t in range(steps):
        dx = b[None, :] - a[None, :] * paths[:, t, :] + z[:, t, :] @ w.T
        paths[:, t + 1, :] = paths[:, t, :] + dx
    return paths


def _fit_numpy(paths: Array, config: SigGANConfig) -> SigGANResult:
    """Deterministic torch-free fallback: Nelder-Mead on the same objective.

    The affine-AR generator is initialized at the method-of-moments scale
    (x0 = mean initial level, b = mean increment, W = increment std spread
    over the noise columns) and optimized on a FIXED seeded noise batch
    (common random numbers), so the objective and the whole run are
    deterministic. Iterations are Nelder-Mead simplex steps, capped at
    config.iterations; early-stop is NM's xatol/fatol convergence.
    """
    n_real, t1, d = paths.shape
    steps = t1 - 1
    q = config.noise_dim
    ref_levels = _expected_levels(paths, config.order, config.time_augment)
    rng = np.random.default_rng(int(config.seed))
    z_eval = rng.standard_normal((config.batch_size, steps, q))

    inc = np.diff(paths, axis=1)
    b0 = inc.mean(axis=(0, 1))
    s0 = inc.std(axis=(0, 1)) / math.sqrt(q)
    w0 = np.broadcast_to(s0[:, None], (d, q)).ravel()
    theta0 = np.concatenate([paths[:, 0, :].mean(axis=0), b0, np.zeros(d), w0])

    evals: list[float] = []

    def objective(theta: Array) -> float:
        fake = _numpy_gen_forward(np.asarray(theta, dtype=float), z_eval, d)
        levels = _signature_levels_batch(
            _time_augment(fake) if config.time_augment else fake, config.order
        )
        val = _moment_sq(levels, ref_levels)
        evals.append(val)
        if not math.isfinite(val):
            raise ArithmeticError("non-finite signature-moment loss in numpy fallback")
        return val

    before = math.sqrt(objective(theta0))
    trace: list[float] = []
    result = _minimize(
        objective,
        theta0,
        method="Nelder-Mead",
        options={
            "maxiter": int(config.iterations),
            "xatol": float(config.tol),
            "fatol": float(config.tol),
        },
        callback=lambda _xk: trace.append(min(evals)),
    )
    theta_hat = np.asarray(result.x, dtype=float)
    if not trace:
        trace = [before * before]
    after = math.sqrt(_moment_sq(_numpy_levels(theta_hat, z_eval, d, config), ref_levels))
    model = {"theta": theta_hat, "steps": steps, "dims": d}
    return SigGANResult(
        backend="numpy",
        paths=_numpy_gen_forward(theta_hat, z_eval, d),
        loss_trace=tuple(trace),
        iterations_run=len(trace),
        sig_distance_before=float(before),
        sig_distance_after=float(after),
        n_parameters=int(theta_hat.shape[0]),
        config=config,
        _model=model,
    )


def _numpy_levels(theta: Array, z: Array, dims: int, config: SigGANConfig) -> list[Array]:
    fake = _numpy_gen_forward(theta, z, dims)
    return _signature_levels_batch(
        _time_augment(fake) if config.time_augment else fake, config.order
    )


def fit_sig_gan(
    real_paths: Array | Iterable[Iterable[Iterable[float]]],
    *,
    config: SigGANConfig | None = None,
    seed: int | None = None,
    iterations: int | None = None,
    backend: str | None = None,
) -> SigGANResult:
    """Fit the signature-moment generator to a batch of real paths.

    real_paths: (n_paths >= 2, T+1 >= 2, d >= 1) finite array. The backend
    resolves as ``auto`` -> torch when the ``nn`` extra is importable else
    the numpy fallback; ``torch`` without torch fails closed with
    ``ImportError``; ``numpy`` forces the deterministic Nelder-Mead fallback.
    """
    cfg = config if config is not None else SigGANConfig()
    overrides: dict[str, Any] = {}
    if seed is not None:
        overrides["seed"] = int(seed)
    if iterations is not None:
        overrides["iterations"] = int(iterations)
    if backend is not None:
        overrides["backend"] = backend
    if overrides:
        cfg = replace(cfg, **overrides)
    _check_config(cfg)
    paths = _as_paths(real_paths, "real_paths")
    resolved = cfg.backend
    if resolved == "auto":
        resolved = "torch" if torch_backend_available() else "numpy"
    if resolved == "torch":
        _torch()  # fail closed with guidance when the extra is absent
        return _fit_torch(paths, cfg)
    return _fit_numpy(paths, cfg)


def generate_paths(result: SigGANResult, n_paths: int, seed: int = 0) -> Array:
    """Draw n_paths seeded sample paths from a fitted :class:`SigGANResult`."""
    if not isinstance(result, SigGANResult):
        raise ValueError(f"result must be a SigGANResult; got {type(result)!r}")
    return result.sample(n_paths, seed)


def _bench_target(
    real: Array, config: SigGANConfig, seed: int
) -> tuple[SigGANResult, float, float, float]:
    fitted = fit_sig_gan(real, config=config, seed=seed)
    gen = fitted.sample(int(real.shape[0]), seed=seed + 101)
    ks = marginal_ks_distance(real, gen)
    acf = acf_distance(real, gen, max_lag=5)
    ref = signature_moment_distance(real, gen, order=config.order)
    return fitted, ks, acf, ref


def bench_sig_gan(seed: int = 23) -> dict[str, float]:
    """Seeded SYNTHETIC bench: signature-GAN on GBM and OU targets.

    Trains the lite signature generator (torch GRU when available, else the
    numpy fallback) on seeded geometric-Brownian-motion and Ornstein-Uhlenbeck
    targets, then reports the signature-moment distance before/after
    training, per-coordinate marginal KS and ACF distance, a same-seed
    determinism delta (max |Δ| between two identical runs), and the torch
    availability flag. All values are floats, all keys carry the
    ``synthetic_`` prefix — correctness evidence only, never market evidence.
    """
    n_paths, steps, dims = 64, 24, 2
    cfg = SigGANConfig(seed=int(seed), iterations=200, batch_size=48, patience=60)
    gbm = sample_gbm_paths(
        n_paths,
        steps,
        dims,
        drift=(0.08, -0.04),
        vol=(0.25, 0.35),
        rho=0.4,
        seed=int(seed) + 1,
    )
    ou = sample_ou_paths(
        n_paths,
        steps,
        dims,
        theta=1.5,
        mean=(0.5, -0.4),
        vol=(0.25, 0.3),
        rho=-0.3,
        seed=int(seed) + 2,
    )
    gbm_res, gbm_ks, gbm_acf, gbm_gate = _bench_target(gbm, cfg, seed=int(seed))
    ou_res, ou_ks, ou_acf, ou_gate = _bench_target(ou, cfg, seed=int(seed) + 10)
    det_a = fit_sig_gan(ou, config=cfg, seed=int(seed) + 10)
    det_b = fit_sig_gan(ou, config=cfg, seed=int(seed) + 10)
    delta = float(np.max(np.abs(det_a.paths - det_b.paths)))
    return {
        "synthetic_torch_available": float(torch_backend_available()),
        "synthetic_gbm_sig_distance_before": float(gbm_res.sig_distance_before),
        "synthetic_gbm_sig_distance_after": float(gbm_res.sig_distance_after),
        "synthetic_gbm_sig_distance_gate": float(gbm_gate),
        "synthetic_gbm_marginal_ks": float(gbm_ks),
        "synthetic_gbm_acf_distance": float(gbm_acf),
        "synthetic_gbm_iterations": float(gbm_res.iterations_run),
        "synthetic_ou_sig_distance_before": float(ou_res.sig_distance_before),
        "synthetic_ou_sig_distance_after": float(ou_res.sig_distance_after),
        "synthetic_ou_sig_distance_gate": float(ou_gate),
        "synthetic_ou_marginal_ks": float(ou_ks),
        "synthetic_ou_acf_distance": float(ou_acf),
        "synthetic_ou_iterations": float(ou_res.iterations_run),
        "synthetic_determinism_delta": delta,
        "synthetic_n_paths": float(n_paths),
        "synthetic_path_length": float(steps + 1),
        "synthetic_seed": float(seed),
    }
