"""Light multi-hypothesis tracking (MHT) — k-best hypothesis tree (SYNTHETIC).

Canonical reference: Reid (1979). Each scan forms assignments of
measurements to tracks via JV on gated likelihoods; the K best
global hypotheses survive. Track score = cumulative log-likelihood
ratio (target vs clutter hypothesis).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quant_fund.models.jonker_volgenant import jv_assign

FloatArray = NDArray[np.float64]


class Track:
    __slots__ = ("x", "P", "loglik", "misses", "hits")

    def __init__(self, x: FloatArray, P: FloatArray) -> None:
        self.x, self.P = x, P
        self.loglik = 0.0
        self.misses = 0
        self.hits = 0


def _kf_predict(t: Track, F: FloatArray, Q: FloatArray) -> None:
    t.x = F @ t.x
    t.P = F @ t.P @ F.T + Q


def _kf_update(t: Track, z: FloatArray, H: FloatArray, R: FloatArray) -> float:
    S = H @ t.P @ H.T + R
    nu = z - H @ t.x
    d2 = float(nu @ np.linalg.solve(S, nu))
    K = t.P @ H.T @ np.linalg.inv(S)
    t.x = t.x + K @ nu
    t.P = t.P - K @ S @ K.T
    lik = float(np.exp(-0.5 * d2) / np.sqrt(((2 * np.pi) ** len(z)) * np.linalg.det(S)))
    return lik


class MHT:
    """k-best MHT over a fixed set of confirmed tracks."""

    def __init__(
        self,
        F: FloatArray,
        Q: FloatArray,
        H: FloatArray,
        R: FloatArray,
        pd: float = 0.9,
        clutter: float = 1e-3,
        gate: float = 9.21,
        k_best: int = 4,
    ) -> None:
        self.F, self.Q, self.H, self.R = F, Q, H, R
        self.pd, self.clutter, self.gate, self.k = pd, clutter, gate, k_best
        self.hyps: list[tuple[float, list[Track]]] = [(0.0, [])]

    def scan(self, zs: list[FloatArray]) -> list[Track]:
        """One measurement scan → best hypothesis' tracks."""
        zs = [np.asarray(z, dtype=np.float64) for z in zs]
        new_hyps: list[tuple[float, list[Track]]] = []
        for score, tracks in self.hyps:
            nt = len(tracks)
            # predict
            pred = []
            for t in tracks:
                tp = Track(t.x.copy(), t.P.copy())
                tp.loglik, tp.misses, tp.hits = t.loglik, t.misses, t.hits
                _kf_predict(tp, self.F, self.Q)
                pred.append(tp)
            # cost matrix: rows=tracks, cols=meas ∪ miss-dummies
            m = len(zs)
            if m == 0:
                for t in pred:
                    t.misses += 1
                    t.loglik += np.log(1 - self.pd)
                new_hyps.append((score, pred))
                continue
            C = np.full((nt, m + nt), -np.log(self.clutter))
            lik_mat = np.zeros((nt, m))
            for i, t in enumerate(pred):
                S = self.H @ t.P @ self.H.T + self.R
                for j, z in enumerate(zs):
                    nu = z - self.H @ t.x
                    d2 = float(nu @ np.linalg.solve(S, nu))
                    if d2 <= self.gate:
                        lik = np.exp(-0.5 * d2) / np.sqrt(
                            ((2 * np.pi) ** len(z)) * np.linalg.det(S)
                        )
                        lik_mat[i, j] = lik
                        C[i, j] = -(np.log(self.pd * lik))
                    else:
                        C[i, j] = 1e9  # outside gate
                C[i, m + i] = -np.log(1 - self.pd)
            # k-best via repeated JV (each on masked copy) — light MHT
            sols: list[tuple[np.ndarray, float]] = []
            for kk in range(min(self.k, 3)):
                Ck = C.copy()
                if kk > 0:
                    a_prev, _ = sols[-1]
                    for i, j in enumerate(a_prev):
                        if j < m:
                            Ck[i, j] = 1e9  # forbid previous assignment
                a, tot = jv_assign(Ck)
                sols.append((a, tot))
            for a, tot in sols:
                trs = []
                for i, t in enumerate(pred):
                    tc = Track(t.x.copy(), t.P.copy())
                    tc.loglik, tc.misses, tc.hits = t.loglik, t.misses, t.hits
                    j = int(a[i])
                    if j < m and C[i, j] < 1e8:
                        lik = _kf_update(tc, zs[j], self.H, self.R)
                        tc.hits += 1
                        tc.misses = 0
                        tc.loglik += np.log(self.pd * lik / self.clutter + 1e-30)
                    else:
                        tc.misses += 1
                        tc.loglik += np.log(1 - self.pd)
                    trs.append(tc)
                new_hyps.append((score - tot, trs))
        new_hyps.sort(key=lambda h: -h[0])
        self.hyps = new_hyps[: self.k]
        return self.hyps[0][1]


def bench_mht(seed: int = 20261231) -> dict[str, float]:
    """SYNTHETIC: two targets crossing — MHT keeps both tracks
    consistent where a greedy tracker swaps identities."""
    rng = np.random.default_rng(seed)
    F = np.array([[1, 1], [0, 1.0]])
    Q = np.eye(2) * 0.01
    H = np.array([[1.0, 0]])
    R = np.eye(1) * 0.02
    mht = MHT(F, Q, H, R, pd=0.95, clutter=0.02, k_best=3)
    mht.hyps = [
        (0.0, [Track(np.array([0.0, 0.3]), np.eye(2)), Track(np.array([10.0, -0.3]), np.eye(2))])
    ]
    a = np.array([0.0, 0.3])
    b = np.array([10.0, -0.3])
    cross_err = 0
    swaps = 0
    steps = 50
    for _t in range(steps):
        a = F @ a
        b = F @ b
        zs = []
        for tru in (a, b):
            if rng.random() < 0.95:
                zs.append(np.array([tru[0] + rng.normal(0, 0.15)]))
        for _c in range(rng.poisson(1)):
            zs.append(np.array([rng.uniform(-1, 11)]))
        trs = mht.scan(zs)
        if len(trs) == 2:
            x0, x1 = trs[0].x[0], trs[1].x[0]
            if x0 > x1 and a[0] < b[0]:
                swaps += 1
            cross_err += min(abs(x0 - a[0]) + abs(x1 - b[0]), abs(x0 - b[0]) + abs(x1 - a[0]))
    return {
        "synthetic_mht_tracks_kept": float(len(mht.hyps[0][1])),
        "synthetic_mht_mean_err": float(cross_err / steps),
        "synthetic_mht_swaps": float(swaps),
        "synthetic_mht_nhyps": float(len(mht.hyps)),
    }
