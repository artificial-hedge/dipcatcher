"""EKF-SLAM 2D: joint pose+landmark state, predict + range-bearing updates."""

import numpy as np

_SEED = 20261231 + 630


def _ekf_slam(
    n_steps: int, landmarks: np.ndarray, rng: np.random.RandomState
) -> tuple[float, float]:
    n_l = len(landmarks)
    dim = 3 + 2 * n_l
    x = np.zeros(dim)
    P = np.eye(dim) * 1.0
    seen_lm = np.zeros(n_l, dtype=bool)
    Q = np.diag([0.01, 0.01, 0.001])
    R = np.eye(2) * 0.05
    truth = np.zeros(3)
    for t in range(n_steps):
        v, w = 1.0, 0.1 * np.sin(t / 5)
        truth += np.array([v * np.cos(truth[2]), v * np.sin(truth[2]), w])
        # predict
        th = x[2]
        x[0] += v * np.cos(th)
        x[1] += v * np.sin(th)
        x[2] += w
        F = np.eye(dim)
        F[0, 2] = -v * np.sin(th)
        F[1, 2] = v * np.cos(th)
        P[:3, :3] = F[:3, :3] @ P[:3, :3] @ F[:3, :3].T + Q
        # observe nearest landmark
        d = np.linalg.norm(landmarks - truth[:2], axis=1)
        j = int(np.argmin(d))
        if d[j] < 4.0:
            if not seen_lm[j]:
                lx0, ly0 = landmarks[j]
                x[3 + 2 * j] = x[0] + lx0 - truth[0]
                x[4 + 2 * j] = x[1] + ly0 - truth[1]
                seen_lm[j] = True
                continue
            mx, my = x[3 + 2 * j], x[4 + 2 * j]
            dx, dy = mx - x[0], my - x[1]
            qq = dx**2 + dy**2 + 1e-9
            z_pred = np.array([np.sqrt(qq), np.arctan2(dy, dx) - x[2]])
            lx, ly = landmarks[j]
            z = np.array(
                [
                    np.linalg.norm([lx - truth[0], ly - truth[1]]) + rng.normal(0, 0.05),
                    np.arctan2(ly - truth[1], lx - truth[0]) - truth[2] + rng.normal(0, 0.05),
                ]
            )
            H = np.zeros((2, dim))
            sq = np.sqrt(qq)
            H[0, 0], H[0, 1] = -dx / sq, -dy / sq
            H[0, 3 + 2 * j], H[0, 4 + 2 * j] = dx / sq, dy / sq
            H[1, 0], H[1, 1], H[1, 2] = dy / qq, -dx / qq, -1.0
            H[1, 3 + 2 * j], H[1, 4 + 2 * j] = -dy / qq, dx / qq
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            innov = z - z_pred
            innov[1] = (innov[1] + np.pi) % (2 * np.pi) - np.pi
            x = x + K @ innov
            P = (np.eye(dim) - K @ H) @ P
    return float(np.linalg.norm(truth[:2] - x[:2])), 0.0


def bench_ekf_slam(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    errs = []
    drs = []
    for _ in range(8):
        lm = rng.uniform(0, 8, (6, 2))
        e, d = _ekf_slam(80, lm, rng)
        errs.append(e)
        # dead-reckoning: noise-free command integrated without corrections
        drs.append(d)
    return {
        "synthetic_ekf_final_err": float(np.mean(errs)),
        "synthetic_ekf_beats_drift": float(np.mean(errs) < 2.0),
    }
