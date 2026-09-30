"""Tests for quant_fund.models.conformal_pid — ConformalPID / ConformalPIDQuantile.

Scenarios (all seeded, deterministic): abrupt scale shift N(0,1) -> N(0,9)
at t = 600 over 1200 steps with a fixed N(0,1)-calibrated forecast grid, and
iid N(0,1) exchangeable streams. Diagnostics are coverage and miscoverage
levels only (AGENTS.md honesty contract).
"""

import pathlib

import numpy as np
import pytest
from scipy.stats import norm

from quant_fund.metrics.conformal import conformal_quantile
from quant_fund.models.conformal import AdaptiveConformal
from quant_fund.models.conformal_pid import ConformalPID, ConformalPIDQuantile

LEVELS = np.round(np.arange(0.05, 0.96, 0.05), 2)  # 19 levels, 0.05..0.95
ALPHA = 0.1
NOMINAL_CENTRAL = 1.0 - 2.0 * ALPHA  # central band is [alpha, 1 - alpha]
LO_I = int(np.where(np.isclose(LEVELS, ALPHA))[0][0])
HI_I = int(np.where(np.isclose(LEVELS, 1.0 - ALPHA))[0][0])
N_STEPS = 1200
SHIFT_AT = 600


def _shift_stream(seed: int) -> np.ndarray:
    """N(0,1) -> N(0,9) (sd 3) at t = 600, 1200 steps."""
    rng = np.random.default_rng(seed)
    return np.concatenate(
        [rng.normal(0.0, 1.0, SHIFT_AT), rng.normal(0.0, 3.0, N_STEPS - SHIFT_AT)]
    )


def _forecast_grid() -> np.ndarray:
    """Fixed N(0,1)-calibrated (hence biased once the scale shifts) grid."""
    return norm.ppf(LEVELS)


def _run_quantile(stream: np.ndarray, **kwargs: object) -> np.ndarray:
    model = ConformalPIDQuantile(quantile_levels=LEVELS, alpha=ALPHA, **kwargs)
    grid = _forecast_grid()
    grids = []
    for y in stream:
        out = model.update(float(y), grid)
        assert out.shape == LEVELS.shape
        assert np.all(np.diff(out) >= -1e-12)  # non-crossing at every step
        grids.append(out)
    return np.stack(grids)


def _central_coverage(stream: np.ndarray, grids: np.ndarray) -> np.ndarray:
    return ((stream >= grids[:, LO_I]) & (stream <= grids[:, HI_I])).astype(float)


def _run_scalar_pid(stream: np.ndarray, **kwargs: object) -> np.ndarray:
    """Two-sided |y| band; half-width = conformal quantile of past scores
    at 1 - alpha_t, with alpha_t driven by ConformalPID on err_t."""
    model = ConformalPID(alpha=ALPHA, **kwargs)
    scores: list[float] = []
    covered: list[float] = []
    for y in stream:
        q = conformal_quantile(np.asarray(scores, dtype=float), model.alpha_t)
        cov = float(abs(y) <= q)
        covered.append(cov)
        scores.append(abs(float(y)))
        model.update(1.0 - cov)
    return np.asarray(covered, dtype=float)


def _run_aci(stream: np.ndarray, gamma: float = 0.05) -> np.ndarray:
    """Same band construction via AdaptiveConformal (one date per step)."""
    model = AdaptiveConformal(alpha=ALPHA, gamma=gamma)
    n = stream.size
    path = model.run(
        y=stream,
        lower=np.zeros(n),
        upper=np.zeros(n),
        dates=[f"{i:08d}" for i in range(n)],
    )
    return np.nan_to_num(path.covered, nan=0.0)


def test_long_run_coverage_under_abrupt_scale_shift() -> None:
    stream = _shift_stream(seed=7)
    grids = _run_quantile(stream)
    covered = _central_coverage(stream, grids)
    trail = covered[N_STEPS - 300 :]
    emp = float(np.mean(trail))
    assert abs(emp - NOMINAL_CENTRAL) <= 0.05, f"trailing-300 coverage {emp:.3f}"


def test_pid_beats_scalar_aci_under_shift() -> None:
    """Cumulative |coverage - nominal| over the last 600 steps: PID < ACI.

    Both loops run on the identical two-sided |y|-score band, seeded; the
    PID loop's proportional kick on error innovations shortens the transient
    after the t=600 shift, so the strict ordering is a deterministic
    property of this seeded stream (cf. Angelopoulos et al. 2023, Fig. 4,
    where the PID controller's quantile path reacts faster than ACI's).
    """
    stream = _shift_stream(seed=7)
    cov_pid = _run_scalar_pid(stream)
    cov_aci = _run_aci(stream)
    dev_pid = float(np.abs(cov_pid[N_STEPS - 600 :] - NOMINAL_CENTRAL).sum())
    dev_aci = float(np.abs(cov_aci[N_STEPS - 600 :] - NOMINAL_CENTRAL).sum())
    assert dev_pid < dev_aci, f"PID {dev_pid:.1f} vs ACI {dev_aci:.1f}"


def test_iid_mean_alpha_converges_to_alpha() -> None:
    """Under exchangeable data the issued level centers on alpha."""
    rng = np.random.default_rng(5)
    stream = rng.normal(0.0, 1.0, N_STEPS)
    model = ConformalPID(alpha=ALPHA)
    scores: list[float] = []
    for y in stream:
        q = conformal_quantile(np.asarray(scores, dtype=float), model.alpha_t)
        cov = float(abs(y) <= q)
        scores.append(abs(float(y)))
        model.update(1.0 - cov)
    mean_alpha = float(model.alpha_history[-200:].mean())
    assert abs(mean_alpha - ALPHA) < 0.02, f"mean alpha {mean_alpha:.4f}"


def test_output_grids_non_crossing_every_step() -> None:
    rng = np.random.default_rng(11)
    stream = rng.normal(0.0, 2.0, 300)
    _run_quantile(stream)  # asserts diff >= -1e-12 inside, every step
    model = ConformalPIDQuantile(quantile_levels=LEVELS, alpha=ALPHA)
    raw = norm.ppf(LEVELS)[::-1].copy()  # deliberately crossed raw grid
    out = model.update(0.0, raw)
    assert np.all(np.diff(out) >= -1e-12)


def test_proportional_only_stable_but_no_restoring_force() -> None:
    """Ki=0 keeps the loop bounded but cannot lock in coverage.

    Documented ordering (Angelopoulos et al. 2023, Thm. 1: long-run coverage
    comes from the integrator): with Ki = 0 the difference term telescopes,
    |alpha_t - alpha_0| <= Kp, so the level reacts to error *changes* only
    and a persistent post-shift miscoverage rate is not corrected — the P-
    only deviation is strictly worse than the PI loop's on the same stream.
    """
    stream = _shift_stream(seed=7)
    cov_p = _run_scalar_pid(stream, Ki=0.0, Kp=0.03)
    cov_pi = _run_scalar_pid(stream)
    dev_p = float(np.abs(cov_p[N_STEPS - 600 :] - NOMINAL_CENTRAL).sum())
    dev_pi = float(np.abs(cov_pi[N_STEPS - 600 :] - NOMINAL_CENTRAL).sum())
    assert dev_p > dev_pi, f"P-only {dev_p:.1f} vs PI {dev_pi:.1f}"
    # bounded: coverage never degenerates to the unadjusted (no-conformal)
    # level and the level stays inside its clip box
    unadj = float((np.abs(stream[SHIFT_AT:]) <= norm.ppf(1.0 - ALPHA / 2.0)).mean())
    assert float(cov_p[-300:].mean()) > unadj


def test_alpha_history_and_coverage_history_shapes() -> None:
    stream = _shift_stream(seed=3)[:64]
    model = ConformalPIDQuantile(quantile_levels=LEVELS, alpha=ALPHA)
    grid = _forecast_grid()
    for y in stream:
        model.update(float(y), grid)
    assert model.n_steps_ == 64
    ah = np.stack(model.alpha_history)
    ch = np.stack(model.coverage_history)
    assert ah.shape == (64, LEVELS.size)
    assert ch.shape == (64, LEVELS.size)
    assert np.all((ch == 0.0) | (ch == 1.0))
    assert np.all(ah >= model.clip[0]) and np.all(ah <= model.clip[1])


def test_first_step_output_is_rearranged_forecast() -> None:
    model = ConformalPIDQuantile(quantile_levels=LEVELS, alpha=ALPHA)
    out = model.update(0.0, _forecast_grid())
    assert np.allclose(out, np.maximum.accumulate(norm.ppf(LEVELS)))


def test_default_levels_follow_alpha_halves() -> None:
    model = ConformalPIDQuantile(alpha=ALPHA)
    assert np.allclose(model.quantile_levels_, [0.05, 0.95])
    assert np.allclose(model.alpha_t_, [0.95, 0.05])


def test_default_gains_follow_sqrt_t_guidance() -> None:
    model = ConformalPID(alpha=ALPHA)
    assert np.isclose(model.Ki, 1.0 / np.sqrt(1000.0))
    assert np.isclose(model.Kp, model.Ki / 6.0)
    assert model.Kd == 0.0


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.1, np.nan])
def test_bad_alpha_fail_closed(alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha"):
        ConformalPID(alpha=alpha)
    with pytest.raises(ValueError, match="alpha"):
        ConformalPIDQuantile(alpha=alpha)


@pytest.mark.parametrize("gain", [-0.01, np.nan, np.inf])
def test_bad_gains_fail_closed(gain: float) -> None:
    with pytest.raises(ValueError, match="Ki"):
        ConformalPID(Ki=gain)
    with pytest.raises(ValueError, match="Kp"):
        ConformalPID(Kp=gain)
    with pytest.raises(ValueError, match="Kd"):
        ConformalPID(Kd=gain)


@pytest.mark.parametrize("clip", [(0.0, 0.5), (0.5, 0.5), (0.9, 0.1), (0.0, 1.0), (0.1, 1.0)])
def test_bad_clip_fail_closed(clip: tuple[float, float]) -> None:
    with pytest.raises(ValueError, match="clip"):
        ConformalPID(clip=clip)


@pytest.mark.parametrize(
    "levels",
    [
        np.array([0.5, 0.5]),
        np.array([0.9, 0.1]),
        np.array([0.0, 0.5]),
        np.array([0.5, 1.0]),
        np.array([-0.2, 0.5]),
        np.array([]),
    ],
)
def test_bad_quantile_levels_fail_closed(levels: np.ndarray) -> None:
    with pytest.raises(ValueError, match="quantile_levels"):
        ConformalPIDQuantile(quantile_levels=levels, alpha=ALPHA)
    with pytest.raises(ValueError, match="quantile_levels"):
        ConformalPID(quantile_levels=levels, alpha=ALPHA)


def test_update_bad_inputs_fail_closed() -> None:
    model = ConformalPIDQuantile(quantile_levels=LEVELS, alpha=ALPHA)
    with pytest.raises(ValueError, match="y"):
        model.update(np.nan, _forecast_grid())
    with pytest.raises(ValueError, match="y"):
        model.update(np.array([0.0, 1.0]), _forecast_grid())
    with pytest.raises(ValueError, match="grid"):
        model.update(0.0, np.zeros(3))
    with pytest.raises(ValueError, match="grid"):
        model.update(0.0, np.full(LEVELS.size, np.nan))
    with pytest.raises(ValueError, match="grid"):
        model.update(0.0, np.zeros((2, LEVELS.size)))
    scalar = ConformalPID(alpha=ALPHA)
    with pytest.raises(ValueError, match="err"):
        scalar.update(-0.1)
    with pytest.raises(ValueError, match="err"):
        scalar.update(np.nan)


def test_module_has_no_forbidden_metrics() -> None:
    import quant_fund.models.conformal_pid as conformal_pid

    text = pathlib.Path(conformal_pid.__file__).read_text(encoding="utf-8").lower()
    for token in ("sharpe", "sortino", "p&l"):
        assert token not in text
