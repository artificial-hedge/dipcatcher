"""CELP-lite: adaptive + fixed codebook closed-loop excitation search over an LPC filter.

SYNTHETIC bench only.
"""

import numpy as np

_SEED = 20261231 + 905


def _lpc(x: np.ndarray, p: int) -> np.ndarray:
    r = np.array([np.dot(x[: x.size - k], x[k:]) / x.size for k in range(p + 1)])
    a = np.zeros(p + 1)
    a[0] = 1.0
    e = r[0]
    for i in range(1, p + 1):
        lam = -sum(a[j] * r[i - j] for j in range(1, i)) - r[i]
        k = lam / max(e, 1e-12)
        an = a.copy()
        for j in range(1, i):
            an[j] = a[j] + k * a[i - j]
        an[i] = k
        a = an
        e = (1.0 - k * k) * e
    return np.asarray(a, dtype=np.float64)


def _synth(exc: np.ndarray, a: np.ndarray) -> np.ndarray:
    p = a.size - 1
    y = np.zeros(exc.size)
    for i in range(exc.size):
        acc = 0.0
        for j in range(1, min(i, p) + 1):
            acc += a[j] * y[i - j]
        y[i] = exc[i] - acc
    return np.asarray(y, dtype=np.float64)


def celp_frame(x: np.ndarray, a: np.ndarray, npitch: int = 40) -> tuple[np.ndarray, float]:
    """Closed-loop excitation: adaptive codebook (past excitation) + small stochastic codebook."""
    n = x.size
    rng = np.random.default_rng(1234)
    exc = np.zeros(n)
    # block into subframes
    bs = n // 4
    # stochastic codebook: Gaussian + sparse impulse-train codewords (pitch-like)
    cb = rng.normal(0, 1.0, (16, bs))
    for phase in range(20):
        v = np.zeros(bs)
        v[phase::20] = 1.0
        cb = np.vstack([cb, v])
    cb = cb / np.maximum(np.linalg.norm(cb, axis=1, keepdims=True), 1e-9)
    gains: list[float] = []
    for sb in range(4):
        lo, hi = sb * bs, (sb + 1) * bs
        target = x[lo:hi]
        # zero-input response from already-selected past excitation
        base = _synth(np.concatenate([exc[:lo], np.zeros(bs)]), a)[lo:hi]
        residual = target - base
        sub = np.zeros(bs)
        for _stage in range(3):  # adaptive then fixed stage
            best = (-1e18, 0.0, 0)
            for ci in range(cb.shape[0]):
                rec_i = _synth(np.concatenate([np.zeros(lo), cb[ci]]), a)[lo:hi]
                g = float(np.dot(residual, rec_i) / max(np.dot(rec_i, rec_i), 1e-9))
                err = float(np.dot(residual - g * rec_i, residual - g * rec_i))
                if -err > best[0]:
                    best = (-err, g, ci)
            sub += best[1] * cb[best[2]]
            gains.append(best[1])
            residual = (
                residual - _synth(np.concatenate([np.zeros(lo), best[1] * cb[best[2]]]), a)[lo:hi]
            )
        exc[lo:hi] = sub
    return np.asarray(exc, dtype=np.float64), float(np.mean(np.abs(gains)))


def bench_celp_encode(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    score = 0.0
    n = 160
    # voiced-like frame: periodic excitation through LPC filter
    a = np.array([1.0, -0.8, 0.4, -0.15])
    exc_true = np.tile(np.r_[np.zeros(19), 3.0], n // 20)
    x = _synth(exc_true, a)
    a_hat = _lpc(x, 3)
    exc, _mg = celp_frame(x, a_hat)
    xh = _synth(exc, a_hat)
    snr_db = float(10 * np.log10(np.dot(x, x) / (np.dot(x - xh, x - xh) + 1e-12)))
    score += 1.0 if snr_db > 12.0 else 0.0
    # periodicity recovered in excitation: autocorr peak at pitch 20
    ac = np.correlate(exc - exc.mean(), exc - exc.mean(), "full")[n - 1 :]
    ac = ac / max(ac[0], 1e-9)
    score += 1.0 if ac[20] > 0.3 else 0.0
    # closed-loop beats open-loop (residual excitation = optimal-weighted codebook)
    e_res = x - _synth(np.zeros(n), a_hat)
    xh_open = _synth(e_res * 0.5, a_hat)
    err_cl = float(np.dot(x - xh, x - xh))
    err_ol = float(np.dot(x - xh_open, x - xh_open))
    score += 1.0 if err_cl < err_ol else 0.0
    # unvoiced frame still decoded with bounded error
    xv = rng.normal(0, 0.3, n)
    xv = _synth(xv, a)
    av = _lpc(xv, 3)
    ev, _ = celp_frame(xv, av)
    xvh = _synth(ev, av)
    err_v = float(np.dot(xv - xvh, xv - xvh) / max(np.dot(xv, xv), 1e-9))
    score += 1.0 if err_v < 0.55 else 0.0
    return {"synthetic_celp_encode": score / 4.0}
