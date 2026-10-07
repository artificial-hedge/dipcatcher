"""LDPC sum-product decoding over AWGN (SYNTHETIC).

Systematic code: codeword c = [m | P m], check matrix H = [P | I].
Belief propagation: check-to-var messages via the tanh product rule,
var-to-check via LLR sums. Bench compares BP bit-error rate against a
hard-decision baseline on the shared (14,8) fixture code.
"""

import numpy as np

from quant_fund.models._code_synth import AWGN_SIGMA, LDPC_K, LDPC_P, bpsk_awgn, msg_bits


def _encode(m: np.ndarray) -> np.ndarray:
    p = (LDPC_P @ m) % 2
    return np.concatenate([m, p])


def _bp_decode(y: np.ndarray, sigma: float, iters: int = 25) -> np.ndarray:
    m_ch, n_ch = LDPC_P.shape
    n = LDPC_P.shape[1] + m_ch
    h = np.zeros((m_ch, n))
    h[:, : LDPC_P.shape[1]] = LDPC_P
    h[:, LDPC_P.shape[1] :] = np.eye(m_ch)
    lc = 2.0 * y / sigma**2  # channel LLR
    q = np.repeat(lc[None, :], m_ch, axis=0) * h  # var->check
    r = np.zeros_like(q)
    for _ in range(iters):
        for c in range(m_ch):
            vs = np.nonzero(h[c])[0]
            prod = np.ones(n)
            for v in vs:
                prod[v] = np.prod(np.tanh(0.5 * q[c, vs][vs != v]))
            r[c, vs] = 2.0 * np.arctanh(np.clip(prod[vs], -0.999999, 0.999999))
        l_tot = lc + (r * h).sum(axis=0)
        q = np.repeat(l_tot[None, :], m_ch, axis=0) * h - r * h
    return (l_tot < 0).astype(int)


def _hard(y: np.ndarray) -> np.ndarray:
    return (y < 0).astype(int)


def bench_ldpc_decoder(seed: int = 5001) -> dict[str, float]:
    m = msg_bits(seed, LDPC_K)
    c = _encode(m)
    y = bpsk_awgn(c, AWGN_SIGMA, seed + 1)
    dec = _bp_decode(y, AWGN_SIGMA)
    hard = _hard(y)
    syn = np.zeros(len(c))
    syn[: LDPC_P.shape[1]] = dec[: LDPC_P.shape[1]]
    syn[LDPC_P.shape[1] :] = dec[LDPC_P.shape[1] :]
    check = (LDPC_P @ dec[:8] + dec[8:]) % 2
    return {
        "synthetic_ldpc_bp_ber": float(np.mean(dec != c)),
        "synthetic_ldpc_hard_ber": float(np.mean(hard != c)),
        "synthetic_ldpc_ber_gain": float(np.mean(hard != c) - np.mean(dec != c)),
        "synthetic_ldpc_syndrome": float(np.sum(check)),
        "synthetic_ldpc_msg_ber": float(np.mean(dec[:8] != m)),
    }
