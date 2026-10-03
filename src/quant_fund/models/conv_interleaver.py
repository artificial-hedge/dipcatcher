"""Block interleaver for burst-error correction.

K Hamming(7,4) codewords are written row-wise and transmitted
column-wise, so a contiguous channel burst is spread across all
codewords after deinterleaving (<= burst/K errors each). Bench compares
Hamming-decoded BER with and without the interleaver under a planted
burst on the shared synthetic message.
"""

import numpy as np

from quant_fund.models._code_synth import burst_channel, msg_bits

_G = np.array(
    [
        [1, 0, 0, 0, 1, 1, 0],
        [0, 1, 0, 0, 1, 0, 1],
        [0, 0, 1, 0, 0, 1, 1],
        [0, 0, 0, 1, 1, 1, 1],
    ]
)
_H = np.array(
    [
        [1, 1, 0, 1, 1, 0, 0],
        [1, 0, 1, 1, 0, 1, 0],
        [0, 1, 1, 1, 0, 0, 1],
    ]
)
_ROWS = 12


def _ham_encode(m: np.ndarray) -> np.ndarray:
    return np.array([(m[4 * i : 4 * i + 4] @ _G) % 2 for i in range(len(m) // 4)]).ravel()


def _ham_decode(c: np.ndarray) -> tuple[np.ndarray, int]:
    out = np.zeros(len(c) // 7 * 4, dtype=int)
    n_corr = 0
    for i in range(len(c) // 7):
        w = c[7 * i : 7 * i + 7]
        syn = (_H @ w) % 2
        if np.any(syn):
            for j in range(7):
                if np.array_equal(syn, _H[:, j]):
                    w[j] ^= 1
                    n_corr += 1
                    break
        out[4 * i : 4 * i + 4] = w[:4]
    return out, n_corr


def bench_conv_interleaver(seed: int = 5011, burst: int = 12) -> dict[str, float]:
    k = 4 * _ROWS
    m = msg_bits(seed, k)
    c = _ham_encode(m)  # 7*ROWS bits
    n = len(c)
    # interleave: write rows of ROWS x (n/ROWS), read column-wise
    mat = c.reshape(_ROWS, n // _ROWS)
    tx = mat.T.ravel()
    rx = burst_channel(tx, 7, burst)
    de = rx.reshape(n // _ROWS, _ROWS).T.ravel()
    dec_i, corr_i = _ham_decode(de)
    # no interleaver baseline
    rx2 = burst_channel(c, 7, burst)
    dec_n, corr_n = _ham_decode(rx2)
    return {
        "synthetic_int_ber": float(np.mean(dec_i != m)),
        "synthetic_int_noint_ber": float(np.mean(dec_n != m)),
        "synthetic_int_gain": float(np.mean(dec_n != m) - np.mean(dec_i != m)),
        "synthetic_int_corr": float(corr_i),
        "synthetic_int_corr_noint": float(corr_n),
    }
