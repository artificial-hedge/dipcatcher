"""FastSLAM-lite: N particles, per-particle landmark EKF mean estimate."""

import numpy as np

_SEED = 20261231 + 634


def _fast_slam(lm: np.ndarray, rng: np.random.RandomState) -> float:
    n_p, n_l = 30, len(lm)
    poses = np.zeros((n_p, 3))
    w = np.full(n_p, 1.0 / n_p)
    est_lm = np.zeros((n_p, n_l, 2))
    truth = np.zeros(3)
    for t in range(60):
        v, om = 0.5, 0.1 * np.sin(t / 4)
        truth += np.array([v * np.cos(truth[2]), v * np.sin(truth[2]), om])
        for k in range(n_p):
            poses[k] += np.array(
                [v * np.cos(poses[k, 2]), v * np.sin(poses[k, 2]), om]
            ) + rng.normal(0, 0.02, 3)
        # observe closest landmark
        d = np.linalg.norm(lm - truth[:2], axis=1)
        j = int(np.argmin(d))
        if d[j] < 3.5:
            lx, ly = lm[j]
            z = np.array([lx, ly]) + rng.normal(0, 0.1, 2)
            for k in range(n_p):
                pred = est_lm[k, j]
                if np.all(pred == 0):
                    est_lm[k, j] = poses[k, :2] + rng.normal(0, 0.4, 2)
                innov = np.linalg.norm(z - poses[k, :2]) - np.linalg.norm(
                    est_lm[k, j] - poses[k, :2]
                )
                w[k] *= np.exp(-(innov**2) / 0.5) + 1e-12
        wsum = w.sum()
        w = w / wsum if wsum > 0 else np.full(n_p, 1.0 / n_p)
        if 1 / np.sum(w**2) < n_p / 2:
            idx = rng.choice(n_p, n_p, p=w / w.sum())
            poses = poses[idx]
            est_lm = est_lm[idx]
            w = np.full(n_p, 1.0 / n_p)
    est = (poses * w[:, None]).sum(0)
    return float(np.linalg.norm(truth[:2] - est[:2]))


def bench_particle_slam(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    errs = []
    for _ in range(6):
        lm = rng.uniform(0, 10, (5, 2))
        errs.append(_fast_slam(lm, rng))
    return {"synthetic_slam_err": float(np.mean(errs) < 2.5)}
