"""Streaming summary sketches: t-digest, HyperLogLog,
Count-Min sketch, and GK quantile.

- t-digest (Dunning & Ertl 2019): centroid-compression
  quantile sketch with the k_1 (arcsine) scale function,
  giving relative accuracy in the tails.
- HyperLogLog (Flajolet et al. 2007): cardinality
  estimation via m=2^b registers of max leading-zero
  ranks + the standard small/large-range corrections.
- Count-Min (Cormode & Muthukrishnan 2005): d x w hash
  table for frequency estimation, conservative update.
- GK (Greenwald & Khanna 2001): epsilon-approximate
  quantiles via the tuple list + band merge.

References
----------
- Dunning & Ertl (2019) 'Computing Extremely Accurate
  Quantiles Using t-Digests' arXiv:1902.04023.
- Flajolet, Fusy, Gandouet, Meunier (2007)
  'HyperLogLog' AOFA.
- Cormode & Muthukrishnan (2005) J. Algorithms 55(1).
- Greenwald & Khanna (2001) SIGMOD.

Honesty
-------
SYNTHETIC self-check: error bounds on seeded streams —
quantile deviation, cardinality relative error,
frequency overestimate. No market claims.

Composition
-----------
Pure numpy. Each sketch is a small class with add/merge
plus a query method; the bench wires them together.
"""

from __future__ import annotations

import hashlib

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _check_stream(x: FloatArray) -> FloatArray:
    xa = np.asarray(x, dtype=np.float64).ravel()
    if xa.size < 4 or not np.isfinite(xa).all():
        raise ValueError("stream must be finite with >= 4 points")
    return xa


class TDigest:
    """t-digest with k1 scale function, delta compression."""

    def __init__(self, delta: float = 100.0) -> None:
        if delta <= 10:
            raise ValueError("delta must exceed 10")
        self.delta = float(delta)
        self._buf: list[float] = []
        self.centroids: list[tuple[float, float]] = []
        self.n = 0

    def add(self, x: float) -> None:
        v = float(x)
        if not np.isfinite(v):
            raise ValueError("non-finite value")
        self._buf.append(v)
        self.n += 1
        if len(self._buf) >= 4 * int(self.delta):
            self._flush()

    def _flush(self) -> None:
        pts = [(v, 1.0) for v in self._buf] + self.centroids
        self._buf = []
        pts.sort(key=lambda t: t[0])
        out: list[tuple[float, float]] = []
        cum = 0.0  # weight committed to completed centroids
        cur_m, cur_w = pts[0]
        for m, wt in pts[1:]:
            q_lo = cum / self.n
            q_hi = (cum + cur_w + wt) / self.n
            if self.delta * (self._k(q_hi) - self._k(q_lo)) <= 1.0:
                cur_m = (cur_m * cur_w + m * wt) / (cur_w + wt)
                cur_w += wt
            else:
                out.append((cur_m, cur_w))
                cum += cur_w
                cur_m, cur_w = m, wt
        out.append((cur_m, cur_w))
        self.centroids = out

    def _k(self, q: float) -> float:
        return float(np.arcsin(2 * min(max(q, 0.0), 1.0) - 1) / np.pi + 0.5)

    def _ensure(self) -> None:
        if self._buf:
            self._flush()

    def quantile(self, q: float) -> float:
        if not 0 <= q <= 1:
            raise ValueError("q in [0,1]")
        self._ensure()
        if not self.centroids:
            raise ValueError("empty digest")
        cs = np.array([c[0] for c in self.centroids])
        wt = np.array([c[1] for c in self.centroids])
        # centroid centers sit at cumulative midpoints
        mid = np.cumsum(wt) - wt / 2
        pos = q * self.n
        if pos <= mid[0]:
            return float(cs[0])
        if pos >= mid[-1]:
            return float(cs[-1])
        idx = int(np.searchsorted(mid, pos, side="right"))
        t = (pos - mid[idx - 1]) / (mid[idx] - mid[idx - 1])
        return float(cs[idx - 1] + t * (cs[idx] - cs[idx - 1]))


class HyperLogLog:
    """HLL cardinality estimator, m=2^b registers."""

    def __init__(self, b: int = 10) -> None:
        if not 4 <= b <= 16:
            raise ValueError("b in [4,16]")
        self.b = b
        self.m = 1 << b
        self.reg = np.zeros(self.m, dtype=np.uint8)

    def add(self, item: float) -> None:
        h = int.from_bytes(
            hashlib.blake2b(np.float64(item).tobytes(), digest_size=8).digest(),
            "little",
        )
        idx = h & (self.m - 1)
        w = h >> self.b
        rank = _rho(w, 64 - self.b)
        if rank > self.reg[idx]:
            self.reg[idx] = rank

    def estimate(self) -> float:
        m = self.m
        alpha = 0.7213 / (1 + 1.079 / m)
        z = float((2.0 ** (-self.reg.astype(np.float64))).sum())
        e = alpha * m * m / z
        n_zero = int((self.reg == 0).sum())
        if e <= 2.5 * m and n_zero:
            e = m * np.log(m / n_zero)
        elif e > (1.0 / 30.0) * (1 << 32):
            e = -(1 << 32) * np.log(1 - e / (1 << 32))
        return float(e)


class CountMinSketch:
    """Count-Min with conservative update, d rows x w cols."""

    def __init__(self, width: int = 512, depth: int = 5, seed: int = 0) -> None:
        if width < 8 or depth < 2:
            raise ValueError("width>=8 depth>=2")
        self.w = int(width)
        self.d = int(depth)
        rng = np.random.default_rng(seed)
        self.salts = rng.integers(0, 2**31 - 1, self.d).astype(np.int64)
        self.table = np.zeros((self.d, self.w))

    def _hash(self, item: float, i: int) -> int:
        h = hashlib.blake2b(
            np.float64(item).tobytes() + int(self.salts[i]).to_bytes(4, "little"),
            digest_size=4,
        ).digest()
        return int.from_bytes(h, "little") % self.w

    def add(self, item: float, c: float = 1.0) -> None:
        idxs = [self._hash(item, i) for i in range(self.d)]
        cur = min(self.table[i, j] for i, j in enumerate(idxs))
        for i, j in enumerate(idxs):
            self.table[i, j] = max(self.table[i, j], cur + c)

    def estimate(self, item: float) -> float:
        return float(min(self.table[i, self._hash(item, i)] for i in range(self.d)))


class GKSketch:
    """Greenwald-Khanna epsilon-approximate quantiles.

    Tuple list of (v, g, delta): g covers rank range, delta
    bounds the rank uncertainty. Compress every cap items.
    """

    def __init__(self, eps: float = 0.01) -> None:
        if not 0 < eps < 0.25:
            raise ValueError("eps in (0,0.25)")
        self.eps = float(eps)
        self.tuples: list[list[float]] = []
        self.n = 0
        self.cap = max(1, int(1.0 / (2 * eps)))

    def add(self, v: float) -> None:
        x = float(v)
        if not np.isfinite(x):
            raise ValueError("non-finite value")
        self.n += 1
        if self.n == 1:
            self.tuples.append([x, 1.0, 0.0])
            return
        # insertion position keeping order by v
        vals = [t[0] for t in self.tuples]
        i = int(np.searchsorted(np.asarray(vals), x, side="left"))
        if i == 0 or i == len(self.tuples):
            delta = 0.0  # extremes carry no rank uncertainty
        else:
            delta = float(int(2 * self.eps * self.n))
        self.tuples.insert(i, [x, 1.0, delta])
        if self.n % self.cap == 0:
            self._compress()

    def _compress(self) -> None:
        thresh = 2 * self.eps * self.n
        i = len(self.tuples) - 2
        while i >= 1:
            if self.tuples[i][1] + self.tuples[i + 1][1] + self.tuples[i + 1][2] <= thresh:
                self.tuples[i + 1][1] += self.tuples[i][1]
                del self.tuples[i]
            i -= 1

    def query(self, q: float) -> float:
        if not self.tuples:
            raise ValueError("empty sketch")
        if not 0 <= q <= 1:
            raise ValueError("q in [0,1]")
        r = int(np.ceil(q * self.n))
        r_min = 0.0
        for i in range(len(self.tuples)):
            r_min += self.tuples[i][1]
            if r_min + self.tuples[i][2] > r + self.eps * self.n:
                return float(self.tuples[i - 1][0]) if i else float(self.tuples[0][0])
        return float(self.tuples[-1][0])


def _rho(w: int, bits: int) -> int:
    """Position of first 1-bit counting from the left."""
    if w == 0:
        return bits + 1
    return int(bits - w.bit_length() + 1)


def bench_sketches(seed: int = 499) -> dict[str, float]:
    """SYNTHETIC: quantile/cardinality/frequency error bounds."""
    rng = np.random.default_rng(seed)
    n = 20000
    x = rng.lognormal(0.0, 1.0, n)

    td = TDigest(delta=80)
    for v in x:
        td.add(float(v))
    true_q = np.quantile(x, [0.5, 0.9, 0.99])
    est_q = np.array([td.quantile(0.5), td.quantile(0.9), td.quantile(0.99)])
    q_err = float(np.max(np.abs(est_q - true_q) / np.abs(true_q)))

    hll = HyperLogLog(b=10)
    items = rng.integers(0, 60000, n).astype(np.float64)
    for v in items:
        hll.add(float(v))
    card_est = hll.estimate()
    card_true = float(np.unique(items).size)
    card_err = abs(card_est - card_true) / card_true

    cms = CountMinSketch(width=1024, depth=5, seed=seed)
    y = rng.integers(0, 500, n).astype(np.float64)
    for v in y:
        cms.add(float(v))
    counts = np.bincount(y.astype(np.int64), minlength=500)
    probe = int(counts.argmax())
    fe = cms.estimate(float(probe)) - float(counts[probe])

    gk = GKSketch(eps=0.01)
    for v in x:
        gk.add(float(v))
    gk_err = float(np.abs(gk.query(0.9) - np.quantile(x, 0.9)) / np.quantile(x, 0.9))

    if q_err >= 0.05 or card_err >= 0.08:
        raise ValueError("sketch error bound violated")
    return {
        "synthetic_tdigest_qrelerr": q_err,
        "synthetic_hll_card_relerr": float(card_err),
        "synthetic_cms_freq_overest": float(fe),
        "synthetic_gk_q90_relerr": gk_err,
        "synthetic_hll_cardinality": card_est,
        "synthetic_cardinality_true": card_true,
    }
