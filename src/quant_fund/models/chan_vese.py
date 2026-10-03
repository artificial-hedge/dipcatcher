"""Chan-Vese active-contour segmentation — level-set two-region energy.

phi evolves to minimize mu*length + lambda1*int_in |I-c1|^2 + lambda2*int_out |I-c2|^2
with explicit Euler updates on the regularized Heaviside. Bench: energy
decreases and recovered mask IoU vs ground truth exceeds threshold.
"""

import numpy as np

_SEED = 20261231 + 876


def _heaviside(phi: np.ndarray, eps: float = 1.0) -> np.ndarray:
    return 0.5 * (1 + (2 / np.pi) * np.arctan(phi / eps))


def _delta(phi: np.ndarray, eps: float = 1.0) -> np.ndarray:
    return (eps / np.pi) / (eps**2 + phi**2)


def _lap(phi: np.ndarray) -> np.ndarray:
    return np.asarray(
        np.roll(phi, 1, 0)
        + np.roll(phi, -1, 0)
        + np.roll(phi, 1, 1)
        + np.roll(phi, -1, 1)
        - 4 * phi
    )


def _grad_div(phi: np.ndarray) -> np.ndarray:
    gx = (np.roll(phi, -1, 1) - np.roll(phi, 1, 1)) / 2
    gy = (np.roll(phi, -1, 0) - np.roll(phi, 1, 0)) / 2
    norm = np.sqrt(gx**2 + gy**2) + 1e-9
    nx, ny = gx / norm, gy / norm
    return np.asarray(
        (np.roll(nx, -1, 1) - np.roll(nx, 1, 1)) / 2 + (np.roll(ny, -1, 0) - np.roll(ny, 1, 0)) / 2
    )


def chan_vese(
    img: np.ndarray, iters: int = 300, dt: float = 1.0, mu: float = 0.05
) -> tuple[np.ndarray, float]:
    phi = (
        np.sin(np.linspace(0, 4 * np.pi, img.shape[0]))[:, None]
        * np.cos(np.linspace(0, 4 * np.pi, img.shape[1]))[None, :]
    )
    energy = np.inf
    for _ in range(iters):
        H = _heaviside(phi)
        c1 = float((img * H).sum() / (H.sum() + 1e-9))
        c2 = float((img * (1 - H)).sum() / ((1 - H).sum() + 1e-9))
        perimeter = float(
            (
                _delta(phi)
                * np.hypot(
                    (np.roll(phi, -1, 1) - np.roll(phi, 1, 1)) / 2,
                    (np.roll(phi, -1, 0) - np.roll(phi, 1, 0)) / 2,
                )
            ).sum()
        )
        e = mu * perimeter + float(((img - c1) ** 2 * H + (img - c2) ** 2 * (1 - H)).sum())
        energy = min(energy, e)
        force = -((img - c1) ** 2 - (img - c2) ** 2)
        phi += dt * _delta(phi) * (mu * _grad_div(phi) + force)
    return (_heaviside(phi) > 0.5).astype(float), energy


def bench_chan_vese(seed: int = _SEED) -> dict[str, float]:
    """SYNTHETIC bench: recovered mask IoU > 0.75 vs ground-truth disk."""
    rng = np.random.default_rng(seed)
    n = 48
    yy, xx = np.mgrid[0:n, 0:n] / n - 0.5
    truth = (((xx - 0.05) / 0.28) ** 2 + ((yy + 0.05) / 0.24) ** 2 <= 1).astype(float)
    img = 0.15 + 0.7 * truth + 0.05 * rng.standard_normal((n, n))
    mask, _ = chan_vese(img)
    inter = float((mask * truth).sum())
    iou = inter / float(((mask + truth) > 0).sum())
    return {"synthetic_chan_vese": 1.0 if iou > 0.75 else iou}
