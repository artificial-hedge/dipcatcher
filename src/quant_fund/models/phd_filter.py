"""Gaussian-mixture PHD filter (Vo & Ma 2006).

Predict/update/prune/merge/extract for the intensity function of a
multi-target RFS under linear-Gaussian dynamics, Poisson birth,
clutter and survival/detection probabilities. State extraction picks
components with weight ≥ 0.5 (expected count rounding).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _kalman_update(
    m: FloatArray, P: FloatArray, z: FloatArray, H: FloatArray, R: FloatArray
) -> tuple[FloatArray, FloatArray, float]:
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    nu = z - H @ m
    m2 = m + K @ nu
    P2 = P - K @ S @ K.T
    lik = float(
        np.exp(-0.5 * nu @ np.linalg.solve(S, nu))
        / np.sqrt(((2 * np.pi) ** len(z)) * np.linalg.det(S))
    )
    return m2, P2, lik


class GMPHD:
    """Linear-Gaussian GM-PHD filter."""

    def __init__(
        self,
        F: FloatArray,
        Q: FloatArray,
        H: FloatArray,
        R: FloatArray,
        ps: float = 0.95,
        pd: float = 0.9,
        clutter: float = 1e-3,
        w_prune: float = 1e-4,
        merge_dist: float = 4.0,
        w_extract: float = 0.5,
    ) -> None:
        self.F, self.Q, self.H, self.R = F, Q, H, R
        self.ps, self.pd = ps, pd
        self.clutter = clutter
        self.w_prune, self.merge_dist, self.w_extract = (w_prune, merge_dist, w_extract)
        self.comps: list[tuple[float, FloatArray, FloatArray]] = []

    def spawn(self, m: FloatArray, P: FloatArray, w: float) -> None:
        self.comps.append((w, np.asarray(m, float), np.asarray(P, float)))

    def predict(self, birth: list[tuple[float, FloatArray, FloatArray]] | None = None) -> None:
        surv = [(self.ps * w, self.F @ m, self.F @ P @ self.F.T + self.Q) for w, m, P in self.comps]
        self.comps = surv + list(birth or [])

    def update(self, zs: list[FloatArray]) -> None:
        if not zs:
            self.comps = [((1 - self.pd) * w, m, P) for w, m, P in self.comps]
            return
        miss = [((1 - self.pd) * w, m, P) for w, m, P in self.comps]
        hit: list[tuple[float, FloatArray, FloatArray]] = []
        for z in zs:
            z = np.asarray(z, dtype=np.float64)
            contribs = []
            for w, m, P in self.comps:
                m2, P2, lik = _kalman_update(m, P, z, self.H, self.R)
                contribs.append((self.pd * w * lik, m2, P2))
            tot = self.clutter + sum(c[0] for c in contribs)
            hit += [(float(c[0]) / tot, c[1], c[2]) for c in contribs]
        self.comps = miss + hit
        self._prune_merge()

    def _prune_merge(self) -> None:
        comps = sorted(self.comps, key=lambda c: -c[0])
        comps = [c for c in comps if c[0] > self.w_prune]
        merged: list[tuple[float, FloatArray, FloatArray]] = []
        used = [False] * len(comps)
        for i, (_wi, mi, Pi) in enumerate(comps):
            if used[i]:
                continue
            grp = [i]
            for j in range(i + 1, len(comps)):
                if used[j]:
                    continue
                d = float((mi - comps[j][1]) @ np.linalg.solve(Pi, mi - comps[j][1]))
                if d < self.merge_dist:
                    grp.append(j)
                    used[j] = True
            wsum = sum(comps[g][0] for g in grp)
            m: FloatArray = np.asarray(
                sum(comps[g][0] * comps[g][1] for g in grp) / wsum,
                dtype=np.float64,
            )
            P: FloatArray = np.asarray(
                sum(
                    comps[g][0] * (comps[g][2] + np.outer(comps[g][1] - m, comps[g][1] - m))
                    for g in grp
                )
                / wsum,
                dtype=np.float64,
            )
            merged.append((wsum, m, P))
        self.comps = merged

    def extract(self) -> list[FloatArray]:
        n = int(round(sum(w for w, _, _ in self.comps)))
        strong = [c for c in self.comps if c[0] >= self.w_extract]
        strong.sort(key=lambda c: -c[0])
        return [c[1] for c in strong[:n]]

    def cardinality(self) -> float:
        return float(sum(w for w, _, _ in self.comps))


def bench_phd(seed: int = 20261231) -> dict[str, float]:
    """SYNTHETIC: two crossing CV targets + Poisson clutter — track
    cardinality error and OSPA-like localization distance."""
    rng = np.random.default_rng(seed)
    F = np.array([[1, 1], [0, 1.0]])
    Q = np.eye(2) * 0.01
    H = np.array([[1.0, 0]])
    R = np.eye(1) * 0.04
    flt = GMPHD(
        F, Q, H, R, ps=0.99, pd=0.95, clutter=0.02, w_prune=1e-3, merge_dist=1.0, w_extract=0.3
    )
    a = np.array([0.0, 0.5])
    b = np.array([10.0, -0.5])
    flt.spawn(a + rng.normal(0, 0.1, 2), np.eye(2), 0.6)
    flt.spawn(b + rng.normal(0, 0.1, 2), np.eye(2), 0.6)
    card_err = 0.0
    loc_err = 0.0
    steps = 60
    for _t in range(steps):
        a = F @ a
        b = F @ b
        zs = []
        for tru in (a, b):
            if rng.random() < 0.95:
                zs.append(np.array([tru[0] + rng.normal(0, 0.2)]))
        for _c in range(rng.poisson(2)):
            zs.append(np.array([rng.uniform(-2, 12)]))
        # adaptive birth: unassociated strong measurements seed comps
        birth = [(0.05, np.array([z[0], 0.0]), np.diag([1.0, 0.2])) for z in zs]
        flt.predict(birth)
        flt.update(zs)
        est = flt.extract()
        card_err += abs(len(est) - 2)
        if len(est) == 2:
            loc_err += min(
                abs(est[0][0] - a[0]) + abs(est[1][0] - b[0]),
                abs(est[0][0] - b[0]) + abs(est[1][0] - a[0]),
            )
        else:
            loc_err += 2.0
    return {
        "synthetic_phd_card_mean": float(card_err / steps),
        "synthetic_phd_loc_err": float(loc_err / steps),
        "synthetic_phd_final_card": flt.cardinality(),
        "synthetic_phd_final_n": float(len(flt.extract())),
    }
