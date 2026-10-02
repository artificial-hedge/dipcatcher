"""Turbo decoder: parallel concatenated RSC (1,5/7) + BCJR iterations.

Encoder: feedback v = u ^ s1 ^ s2 (poly 7), parity = v ^ s2 (poly 5).
Two component decoders exchange extrinsic LLRs through an interleaver;
bench compares turbo BER after 4 iterations against single-pass BCJR
and uncoded hard decisions on the shared synthetic message at sigma=1.3
(deep enough in the noise floor that iteration has headroom).
"""

import numpy as np

from quant_fund.models._code_synth import TURBO_K, bpsk_awgn, msg_bits

_NSTATE = 4


def _rsc_step(s: int, u: int) -> tuple[int, int]:
    s1, s2 = (s >> 1) & 1, s & 1
    v = u ^ s1 ^ s2
    parity = v ^ s2
    return ((v << 1) | s1), parity


def _rsc_encode(bits: np.ndarray) -> np.ndarray:
    s = 0
    out = []
    for u in bits:
        s, p = _rsc_step(s, int(u))
        out.append(p)
    return np.array(out)


def _bcjr(y_sys: np.ndarray, y_par: np.ndarray, app: np.ndarray, sigma: float):
    """MAP decoder; LLR convention: positive => bit 1 more likely."""
    k = len(y_sys)
    inf = 30.0
    lc = 2.0 / sigma**2
    alpha = np.full((k + 1, _NSTATE), -inf)
    alpha[0, 0] = 0.0
    beta = np.full((k + 1, _NSTATE), -inf)
    beta[k] = 0.0
    gamma = np.full((k, _NSTATE, _NSTATE), -inf)
    trans = {}
    for t in range(k):
        for s in range(_NSTATE):
            for u in (0, 1):
                ns, p = _rsc_step(s, u)
                gamma[t, s, ns] = -lc * u * y_sys[t] - lc * p * y_par[t] + u * app[t]
                trans[(s, ns)] = u
    for t in range(k):
        for ns in range(_NSTATE):
            alpha[t + 1, ns] = np.logaddexp.reduce(alpha[t] + gamma[t, :, ns])
    for t in range(k - 1, -1, -1):
        for s in range(_NSTATE):
            beta[t, s] = np.logaddexp.reduce(beta[t + 1] + gamma[t, s])
    ext = np.zeros(k)
    llr = np.zeros(k)
    for t in range(k):
        num, den = -inf, -inf
        for s in range(_NSTATE):
            for ns in range(_NSTATE):
                if gamma[t, s, ns] <= -inf:
                    continue
                val = alpha[t, s] + gamma[t, s, ns] + beta[t + 1, ns]
                if trans[(s, ns)] == 1:
                    num = np.logaddexp(num, val)
                else:
                    den = np.logaddexp(den, val)
        llr[t] = num - den
        ext[t] = llr[t] + lc * y_sys[t] - app[t]
    return ext, llr


def _interleave(x: np.ndarray, perm: np.ndarray) -> np.ndarray:
    return np.asarray(x[perm], dtype=np.float64)


def _deinterleave(x: np.ndarray, perm: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    out[perm] = x
    return np.asarray(out, dtype=np.float64)


def _turbo(
    y_sys: np.ndarray,
    y_p1: np.ndarray,
    y_p2: np.ndarray,
    perm: np.ndarray,
    sigma: float,
    iters: int = 4,
) -> np.ndarray:
    app1 = np.zeros(len(y_sys))
    for _ in range(iters):
        ext1, _ = _bcjr(y_sys, y_p1, app1, sigma)
        app2 = _interleave(ext1, perm)
        ext2, _ = _bcjr(_interleave(y_sys, perm), y_p2, app2, sigma)
        app1 = _deinterleave(ext2, perm)
    ext1, llr = _bcjr(y_sys, y_p1, app1, sigma)
    return np.asarray(llr > 0, dtype=int)


def bench_turbo_decoder(seed: int = 5015, sigma: float = 1.3) -> dict[str, float]:
    m = msg_bits(seed, TURBO_K)
    rng = np.random.default_rng(seed + 9)
    perm = rng.permutation(TURBO_K)
    sys_bits = m.copy()
    p1 = _rsc_encode(m)
    p2 = _rsc_encode(_interleave(m, perm))
    codeword = np.concatenate([sys_bits, p1, p2])
    y = bpsk_awgn(codeword, sigma, seed + 1)
    ys, yp1, yp2 = y[:TURBO_K], y[TURBO_K : 2 * TURBO_K], y[2 * TURBO_K :]
    dec = _turbo(ys, yp1, yp2, perm, sigma)
    _, llr1 = _bcjr(ys, yp1, np.zeros(TURBO_K), sigma)
    single = (llr1 > 0).astype(int)
    unc = (ys < 0).astype(int)
    return {
        "synthetic_turbo_ber": float(np.mean(dec != m)),
        "synthetic_turbo_single_ber": float(np.mean(single != m)),
        "synthetic_turbo_uncoded_ber": float(np.mean(unc != m)),
        "synthetic_turbo_gain": float(np.mean(unc != m) - np.mean(dec != m)),
    }
