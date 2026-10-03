"""HMM voice-activity detection via Viterbi decoding over log-energy features.

SYNTHETIC bench only.
"""

import numpy as np

_SEED = 20261231 + 907

_TRANS = np.array([[0.98, 0.02], [0.02, 0.98]])  # sticky speech/silence states


def frame_energy(x: np.ndarray, flen: int) -> np.ndarray:
    n = x.size // flen
    e = np.array([np.sum(x[i * flen : (i + 1) * flen] ** 2) for i in range(n)])
    return np.asarray(np.log10(e + 1e-9), dtype=np.float64)


def viterbi_decode(
    loge: np.ndarray, mu_s: float, mu_v: float, var_s: float, var_v: float
) -> np.ndarray:
    n = loge.size
    emit = np.stack(
        [
            -0.5 * (loge - mu_s) ** 2 / var_s,
            -0.5 * (loge - mu_v) ** 2 / var_v,
        ],
        axis=1,
    )
    lt = np.log(_TRANS + 1e-12)
    dp = np.full((n, 2), -1e18)
    dp[0] = emit[0]
    back = np.zeros((n, 2), dtype=np.int64)
    for t in range(1, n):
        for s in range(2):
            cand = dp[t - 1] + lt[:, s]
            back[t, s] = int(np.argmax(cand))
            dp[t, s] = float(cand[back[t, s]]) + emit[t, s]
    path = np.zeros(n, dtype=np.int64)
    path[-1] = int(np.argmax(dp[-1]))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def bench_viterbi_vad(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    score = 0.0
    flen, nf = 80, 120
    # speech bursts on frames 20-50, 70-95
    truth = np.zeros(nf, dtype=np.int64)
    truth[20:50] = 1
    truth[70:95] = 1
    x = rng.normal(0, 0.02, nf * flen)
    for i in range(nf):
        if truth[i]:
            seg = np.arange(flen)
            x[i * flen : (i + 1) * flen] += 0.5 * np.sin(2 * np.pi * seg / 40)
    loge = frame_energy(x, flen)
    mus = float(np.mean(loge[truth == 0]))
    muv = float(np.mean(loge[truth == 1]))
    vs = float(np.var(loge[truth == 0])) + 1e-6
    vv = float(np.var(loge[truth == 1])) + 1e-6
    path = viterbi_decode(loge, mus, muv, vs, vv)
    acc = float(np.mean(path == truth))
    score += 1.0 if acc > 0.9 else 0.0
    # Viterbi finds global optimum: brute-force over a short instance
    n2 = 12
    le2 = loge[:n2]
    best_path = np.zeros(n2, dtype=np.int64)
    best_score = -1e18
    for bits in range(2**n2):
        p = np.array([(bits >> i) & 1 for i in range(n2)])
        sc = 0.0
        for t in range(n2):
            mu = muv if p[t] else mus
            va = vv if p[t] else vs
            sc += -0.5 * (le2[t] - mu) ** 2 / va
            if t:
                sc += np.log(_TRANS[p[t - 1], p[t]] + 1e-12)
        if sc > best_score:
            best_score = sc
            best_path = p.copy()
    vit = viterbi_decode(le2, mus, muv, vs, vv)
    score += 1.0 if np.array_equal(vit, best_path) else 0.0
    # sticky transitions suppress single-frame flicker vs framewise argmax
    flick = np.zeros(30)
    flick[15] = 1.5  # single high-energy outlier
    mus2, muv2 = -0.2, 1.2
    vs2 = vv2 = 0.05
    vitf = viterbi_decode(flick, mus2, muv2, vs2, vv2)
    score += 1.0 if np.sum(vitf) <= 3 else 0.0
    # contiguous speech region detected as one segment
    runs = int(np.sum(np.diff(np.r_[0, path, 0]) != 0) // 2)
    score += 1.0 if runs <= 3 else 0.0
    return {"synthetic_viterbi_vad": score / 4.0}
