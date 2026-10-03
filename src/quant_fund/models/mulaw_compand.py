"""G.711 μ-law companding: encode/decode + segment SNR oracle.

SYNTHETIC bench only — no live-audio claim.
"""

import numpy as np

_SEED = 20261231 + 902
_MU = 255.0


def mu_encode(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=np.float64), -1.0, 1.0)
    y = np.sign(x) * np.log1p(_MU * np.abs(x)) / np.log1p(_MU)
    return np.asarray(np.round((y + 1.0) * 127.5).astype(np.int16), dtype=np.float64)


def mu_decode(q: np.ndarray) -> np.ndarray:
    y = np.asarray(q, dtype=np.float64) / 127.5 - 1.0
    sgn = np.sign(y)
    mag = (np.exp(np.abs(y) * np.log1p(_MU)) - 1.0) / _MU
    return np.asarray(sgn * mag, dtype=np.float64)


def uniform_quant(x: np.ndarray, bits: int = 8) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=np.float64), -1.0, 1.0)
    lv = 2**bits
    return np.asarray(np.round((x + 1.0) * (lv - 1) / 2) / (lv - 1) * 2 - 1.0, dtype=np.float64)


def snr(x: np.ndarray, xh: np.ndarray) -> float:
    num = float(np.sum(x * x))
    den = float(np.sum((x - xh) ** 2)) + 1e-15
    return float(10.0 * np.log10(num / den))


def bench_mulaw_compand(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    score = 0.0
    # low-amplitude sinusoid: μ-law beats uniform (compander boosts small-signal resolution)
    t = np.arange(4000) / 4000.0
    x = 0.02 * np.sin(2 * np.pi * 3 * t) + rng.normal(0, 0.002, 4000)
    x = np.clip(x, -1.0, 1.0)
    sh_mu = snr(x, mu_decode(mu_encode(x)))
    sh_uni = snr(x, uniform_quant(x, 8))
    score += 1.0 if sh_mu > sh_uni else 0.0
    # roundtrip error bounded
    xh = mu_decode(mu_encode(x))
    score += 1.0 if np.max(np.abs(x - xh)) < 0.02 else 0.0
    # monotonicity of encoder
    xs = np.linspace(-1.0, 1.0, 200)
    q = mu_encode(xs)
    score += 1.0 if np.all(np.diff(q) >= 0) else 0.0
    # zero maps to midpoint code
    score += 1.0 if abs(mu_encode(np.array([0.0]))[0] - 128.0) <= 1.0 else 0.0
    # full-scale ≈ -1..1 preserved within 1%
    xf = mu_decode(mu_encode(np.array([0.95])))
    score += 1.0 if abs(xf[0] - 0.95) < 0.01 else 0.0
    return {"synthetic_mulaw_compand": score / 5.0}
