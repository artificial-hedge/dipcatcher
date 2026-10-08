"""Synthetic interpretability fixture (SYNTHETIC).

Hidden model: tokens embed in D=16 via ground-truth directions
F_k (k=0..7 features); activations are superposition a = Σ s_k F_k
with sparse s. A toy "model" maps x → 2-layer MLP activations.
Interpretability tools (SAE, probes, steering, patching) recover the
true feature decomposition from activations only.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

D_ACT = 16
N_FEAT = 8

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def feature_directions(rng: np.random.Generator) -> FloatArray:
    f = rng.normal(0, 1, (N_FEAT, D_ACT))
    return f / np.linalg.norm(f, axis=-1, keepdims=True)


def synth_activations(n: int, dirs: FloatArray, rng: np.random.Generator, sparsity: float = 0.7):
    """s (n,8) sparse; a = s @ dirs + tiny noise."""
    s = rng.uniform(0.5, 1.5, (n, N_FEAT)) * (rng.random((n, N_FEAT)) > sparsity)
    a = s @ dirs + rng.normal(0, 0.02, (n, D_ACT))
    return a, s
