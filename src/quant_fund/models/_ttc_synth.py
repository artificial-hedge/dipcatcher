"""Synthetic multi-step reasoning fixture for test-time-compute canon.

Problems: (n,3) ints; hidden ops f_i = (a_i + a_{i+1} + i) mod 3 for
i=0,1 — deterministic learnable map. Evaluate v = ((a0 op0 a1) op1 a2)
with ops {+,-,x}. Models get features a only; a weak policy trained on
few labels is imperfect — test-time compute (sampling, verification,
search, debate) recovers accuracy.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

IntArray = NDArray[np.int64]
FloatArray = NDArray[np.float64]
_OPS = np.array([[1, 1, 0], [1, -1, 0], [0, 1, 0]])


def true_ops(a: IntArray) -> IntArray:
    n = a.shape[0]
    f = np.zeros((n, 2), dtype=np.int64)
    for i in range(2):
        f[:, i] = (a[:, i] + a[:, i + 1] + i) % 3
    return f


def eval_chain(a: IntArray, ops: IntArray) -> FloatArray:
    v = a[:, 0].astype(float)
    for i in range(2):
        o = ops[:, i]
        v = np.where(o == 0, v + a[:, i + 1], np.where(o == 1, v - a[:, i + 1], v * a[:, i + 1]))
    return v


def synth_problems(n: int, rng: np.random.Generator) -> tuple[IntArray, IntArray, FloatArray]:
    a = rng.integers(1, 10, (n, 3))
    ops = true_ops(a)
    return a, ops, eval_chain(a, ops)


def op_features(a: IntArray, step: int) -> FloatArray:
    """Per-step features for a policy head.

    true op_i = (a_i + a_{i+1} + i) mod 3 — periodic (sin/cos) features of
    the operand sum make it linearly separable; plain raw features don't.
    """
    n = a.shape[0]
    x = np.zeros((n, 10))
    ssum = a[:, step] + a[:, step + 1] + step
    x[:, 0] = a[:, step] / 10.0
    x[:, 1] = a[:, step + 1] / 10.0
    x[:, 2] = np.sin(2 * np.pi * ssum / 3.0)
    x[:, 3] = np.cos(2 * np.pi * ssum / 3.0)
    x[:, 4] = np.sin(2 * np.pi * ssum / 9.0)
    x[:, 5] = np.cos(2 * np.pi * ssum / 9.0)
    x[:, 6] = a[:, step] * a[:, step + 1] / 100.0
    x[:, 7] = a[:, (step + 2) % 3] / 10.0
    x[:, 8 + step] = 1.0
    return x
