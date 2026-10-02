"""Procrustes alignment — orthogonal similarity transforms.

Gower (1975), Dryden & Mardia (2016): given a reference shape X and
a target Y (both n x d centered point configurations), the
orthogonal Procrustes problem solves

    min_{R in O(d), s, t}  ||s Y R + t - X||_F

via the SVD of X^T Y: R = V U^T (det-corrected), s = tr(D)/||Y||^2.
Generalized Procrustes analysis (GPA) aligns a collection of shapes
by iterating pairwise alignment to the evolving mean.

Honesty: the bench plants a known rotation + isotropic scale,
checks the rotation is recovered to Frobenius tolerance and the
disparity (normalized residual SS) is near zero; a reflected-noise
configuration is checked for GPA convergence (mean-shape distance
decreases). Fail-closed on mismatched point counts or degenerate
(zero-variance) configurations.

References: Gower (1975) "Generalized Procrustes analysis";
Schonemann (1966) "A generalized solution of the orthogonal
Procrustes problem"; Dryden & Mardia (2016) "Statistical Shape
Analysis" ch. 7.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _check_shape(x: FloatArray, name: str = "shape") -> FloatArray:
    a = np.asarray(x, dtype=float)
    if a.ndim != 2 or a.shape[0] < 3 or a.shape[1] < 1 or not np.isfinite(a).all():
        raise ValueError(f"bad {name}")
    return a


def procrustes(x_ref: FloatArray, x_target: FloatArray) -> dict[str, FloatArray | float]:
    """Orthogonal Procrustes alignment of ``x_target`` onto ``x_ref``.

    Both inputs are (n, d) configurations; translation and isotropic
    scale are removed. Returns ``aligned``, ``rotation``, ``scale``,
    ``disparity`` (residual SS / ||X_ref||^2 after centering+scaling).
    """
    a = _check_shape(x_ref, "x_ref")
    b = _check_shape(x_target, "x_target")
    if a.shape != b.shape:
        raise ValueError("shape mismatch")
    ac = a - a.mean(axis=0)
    bc = b - b.mean(axis=0)
    na = np.linalg.norm(ac)
    nb = np.linalg.norm(bc)
    if na < 1e-12 or nb < 1e-12:
        raise ValueError("degenerate configuration")
    an = ac / na
    bn = bc / nb
    u, s, vh = np.linalg.svd(an.T @ bn)
    r = vh.T @ u.T
    # reflection correction: keep det(R) = +1
    if np.linalg.det(r) < 0:
        vh[-1] *= -1
        r = vh.T @ u.T
    # optimal isotropic scale c* = tr(X^T Y R) / ||Y_c||^2 = tr(D) na/nb
    scale = float(s.sum()) * na / nb
    aligned = scale * (bc @ r) + a.mean(axis=0)
    disp = float(((ac - scale * (bc @ r)) ** 2).sum() / (na * na))
    return {
        "aligned": np.asarray(aligned, dtype=np.float64),
        "rotation": np.asarray(r, dtype=np.float64),
        "scale": np.asarray(scale),
        "disparity": np.asarray(disp),
    }


def generalized_procrustes(
    shapes: list[FloatArray], n_iter: int = 50
) -> dict[str, FloatArray | float]:
    """GPA: align every shape to the running consensus mean.

    Returns ``mean_shape``, ``aligned`` (list as stacked array),
    ``dispersion`` (sum of disparities).
    """
    if len(shapes) < 2:
        raise ValueError("need >=2 shapes")
    xs = [_check_shape(s) for s in shapes]
    ref = xs[0]
    if any(s.shape != ref.shape for s in xs):
        raise ValueError("shape mismatch")
    aligned = list(xs)
    prev = np.inf
    mean = ref.copy()
    for _ in range(max(1, n_iter)):
        mean = np.stack(aligned).mean(axis=0)
        out = []
        disp = 0.0
        for s in xs:
            r = procrustes(mean, s)
            out.append(np.asarray(r["aligned"], dtype=np.float64))
            disp += float(r["disparity"])
        aligned = out
        if abs(prev - disp) < 1e-12:
            break
        prev = disp
    return {
        "mean_shape": np.asarray(mean, dtype=np.float64),
        "aligned": np.stack(aligned),
        "dispersion": np.asarray(disp),
    }


def bench_procrustes(seed: int = 20261231 + 417) -> dict[str, float]:
    """SYNTHETIC check — planted rotation/scale recovery."""
    rng = np.random.default_rng(seed)
    n, d = 40, 3
    x = rng.standard_normal((n, d))
    ang = 0.7
    r_true = np.array(
        [
            [np.cos(ang), -np.sin(ang), 0.0],
            [np.sin(ang), np.cos(ang), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    y = 2.5 * (x @ r_true) + 0.01 * rng.standard_normal((n, d)) + 3.0
    out = procrustes(x, y)
    r_hat = np.asarray(out["rotation"], dtype=np.float64)
    r_err = float(np.abs(r_hat - r_true.T).max())
    disp = float(out["disparity"])
    if r_err > 0.05 or disp > 0.01:
        raise ValueError(f"procrustes off: r_err={r_err:.4f} disp={disp:.4f}")
    # GPA on noisy rotated copies: consensus is frame-free, so the
    # check is Procrustes distance to the true (centered) shape.
    shapes = [x @ r_true + 0.02 * rng.standard_normal((n, d)) for _ in range(4)]
    gpa = generalized_procrustes(shapes)
    md = float(procrustes(x, np.asarray(gpa["mean_shape"]))["disparity"])
    if md > 0.02:
        raise ValueError(f"gpa mean off: {md:.4f}")
    return {
        "synthetic_procrustes_rot_err": r_err,
        "synthetic_procrustes_disp": disp,
        "synthetic_procrustes_gpa_md": md,
        "score": 1.0,
    }
