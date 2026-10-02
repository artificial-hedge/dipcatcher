"""Bandt-Pompe permutation entropy and the complexity-entropy plane.

For a scalar series x_t and embedding dimension m, each window
(x_t, ..., x_{t+m-1}) maps to an ordinal pattern — the ranking of
its elements. The distribution over the m! patterns carries the
series' temporal structure:

    H = -sum pi log pi / log(m!)      (normalized Shannon entropy)

and the complexity-entropy plane (Lopez-Ruiz, Mancini, Calbet 1995;
Rosso et al. 2007) pairs H with the Jensen-Shannon complexity

    C = H * Q,  Q = Q0 * JS(pi, uniform)

so that white noise sits at (H~1, C~0), periodic signals at
(H~0, C moderate), and chaotic/structured signals at intermediate H
with high C — a joint characterization a single entropy cannot give.

References
----------
- Bandt, C., Pompe, B. (2002). "Permutation entropy: a natural
  complexity measure for time series." *PRL* 88, 174102.
- Rosso, O.A., Larrondo, H.A., Martin, M.T., Plastino, A., Fuentes,
  M.A. (2007). "Distinguishing noise from chaos." *PRL* 99, 154102.
- Zanin, M., Zunino, L., Rosso, O.A., Papo, D. (2012). "Permutation
  entropy and its main biomedical and econophysics applications."
  *Entropy* 14(8).

Honesty
-------
SYNTHETIC series only; bench checks ordering of H across regimes —
not a real-signal classification claim.

Composition
-----------
Called by ``quant_fund.research.benches_w66.bench_perm_entropy``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _pattern_counts(x: FloatArray, m: int, tau: int = 1) -> FloatArray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or m < 2 or m > 7 or tau < 1:
        raise ValueError("bad embedding spec")
    n_pat = x.size - (m - 1) * tau
    if n_pat < 5:
        raise ValueError("series too short")
    windows = np.lib.stride_tricks.sliding_window_view(x, m * tau)[: n_pat * tau : tau][:, ::tau]
    ranks = np.argsort(windows, axis=1)
    _, counts = np.unique(ranks, axis=0, return_counts=True)
    return counts.astype(float)


def permutation_entropy(x: FloatArray, m: int = 5, tau: int = 1) -> float:
    """Normalized Bandt-Pompe permutation entropy in [0,1]."""
    import math

    counts = _pattern_counts(x, m, tau)
    pi = counts / counts.sum()
    h = -float(np.sum(pi * np.log(pi)))
    return h / math.log(math.factorial(m))


def complexity_entropy(x: FloatArray, m: int = 5, tau: int = 1) -> tuple[float, float]:
    """(H, C) on the Rosso complexity-entropy plane.

    C = H * Q0 * JS(pi || uniform) where JS is the Jensen-Shannon
    divergence (log base 2) and Q0 the normalization making C
    bounded in [0,1] for the Bandt-Pompe simplex.
    """
    import math

    counts = _pattern_counts(x, m, tau)
    pi = counts / counts.sum()
    n_states = math.factorial(m)
    h = -float(np.sum(pi * np.log2(pi))) / math.log2(n_states)
    unif = np.full(pi.size, 1.0 / pi.size)
    mix = 0.5 * (pi + unif)
    js = float(
        0.5
        * (
            np.sum(np.where(pi > 0, pi * np.log2(pi / mix), 0.0))
            + np.sum(unif * np.log2(unif / mix))
        )
    )
    # JS max: delta distribution vs uniform over n states.
    n_ = float(pi.size)
    js_max = 0.5 * np.log2(2.0 * n_ / (n_ + 1.0)) + (n_ - np.log2(n_ + 1.0)) / (2.0 * n_)
    c = h * float(js / js_max)
    return h, c


def bench_permutation_entropy(seed: int = 20261231 + 388) -> dict[str, float]:
    """SYNTHETIC check — entropy ordering separates regimes."""
    rng = np.random.default_rng(seed)
    n = 6000
    noise = rng.standard_normal(n)
    periodic = np.sin(np.linspace(0, 200 * np.pi, n))
    # Persistent AR(1): structured but stochastic.
    ar = np.zeros(n)
    e = rng.standard_normal(n)
    for i in range(1, n):
        ar[i] = 0.9 * ar[i - 1] + np.sqrt(0.19) * e[i]
    h_noise = permutation_entropy(noise, m=5)
    h_per = permutation_entropy(periodic, m=5)
    h_ar = permutation_entropy(ar, m=5)
    if not (h_per < h_ar < h_noise):
        raise ValueError("entropy ordering failed")
    if not (h_noise > 0.9 and h_per < 0.3):
        raise ValueError("entropy anchors out of range")
    hn, cn = complexity_entropy(noise, m=5)
    _, c_ar = complexity_entropy(ar, m=5)
    if not (cn < c_ar):
        raise ValueError("complexity ordering failed")
    # Determinism: same series, same answer.
    if permutation_entropy(ar, m=5) != h_ar:
        raise ValueError("non-deterministic")
    return {
        "synthetic_pe_noise": h_noise,
        "synthetic_pe_periodic": h_per,
        "synthetic_pe_ar": h_ar,
        "synthetic_ce_noise_c": cn,
        "synthetic_ce_ar_c": c_ar,
        "score": 1.0,
    }
