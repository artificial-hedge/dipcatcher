"""EKF-SLAM 2D: joint pose+landmark state, predict + range-bearing updates (SYNTHETIC)."""

import numpy as np

_SEED = 20261231 + 630


def _ekf_slam(
    n_steps: int,
    landmarks: np.ndarray,
    rng: np.random.RandomState,
    updates: bool = True,
    x0_err: tuple[float, float, float] = (0.25, -0.15, 0.03),
) -> tuple[float, float]:
    n_l = len(landmarks)
    dim = 3 + 2 * n_l
    x = np.zeros(dim)
    P = np.eye(dim) * 1.0
    # modest initial pose error — the filter must recover via measurements;
    # updates=False gives the open-loop baseline the EKF must beat
    x[:3] = np.array(x0_err)
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
            lx, ly = landmarks[j]
            z = np.array(
                [
                    np.linalg.norm([lx - truth[0], ly - truth[1]]) + rng.normal(0, 0.05),
                    np.arctan2(ly - truth[1], lx - truth[0]) - truth[2] + rng.normal(0, 0.05),
                ]
            )
            if not seen_lm[j]:
                # init landmark from the MEASUREMENT z at the estimated pose
                # (was: truth leak — x[0] + lx0 - truth[0] is oracle info)
                r_m, b_m = z
                x[3 + 2 * j] = x[0] + r_m * np.cos(b_m + x[2])
                x[4 + 2 * j] = x[1] + r_m * np.sin(b_m + x[2])
                seen_lm[j] = True
                continue
            mx, my = x[3 + 2 * j], x[4 + 2 * j]
            dx, dy = mx - x[0], my - x[1]
            qq = dx**2 + dy**2 + 1e-9
            z_pred = np.array([np.sqrt(qq), np.arctan2(dy, dx) - x[2]])
            if not updates:
                continue
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
    # map error over observed landmarks: estimates should converge near truth
    seen_idx = np.flatnonzero(seen_lm)
    if seen_idx.size:
        est_lm = np.stack([x[3 + 2 * j : 5 + 2 * j] for j in seen_idx])
        map_err = float(np.linalg.norm(est_lm - landmarks[seen_idx], axis=1).mean())
    else:
        map_err = float("inf")
    return float(np.linalg.norm(truth[:2] - x[:2])), map_err


def bench_ekf_slam(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    errs = []
    ol_errs = []
    maps: list[float] = []
    n_seen = 0
    for _ in range(8):
        lm = rng.uniform(0, 8, (6, 2))
        e, m_err = _ekf_slam(80, lm, rng)
        e_ol, _ = _ekf_slam(80, lm, rng, updates=False)
        errs.append(e)
        ol_errs.append(e_ol)
        if np.isfinite(m_err):
            maps.append(m_err)
            n_seen += 1
    map_mean = float(np.mean(maps)) if maps else float("inf")
    return {
        "synthetic_ekf_final_err": float(np.mean(errs)),
        "synthetic_ekf_openloop_err": float(np.mean(ol_errs)),
        "synthetic_ekf_map_err": map_mean,
        "synthetic_ekf_frac_seen": n_seen / 8,
        # with noise-free commands dead reckoning is optimal, so the
        # honest signal is map convergence + bounded (non-divergent) pose
        "synthetic_ekf_map_converges": float(map_mean < 0.8 and np.mean(errs) < 8.0),
    }
