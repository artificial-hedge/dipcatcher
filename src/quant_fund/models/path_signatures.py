"""Path signatures and rough-path features for financial time series (SYNTHETIC).

Truncated iterated-integral signatures computed in pure numpy, following
the tensor-algebra construction of

- Chen, K.-T. (1958), "Integration of paths — a faithful representation of
  paths by non-commutative formal power series", Trans. Amer. Math. Soc.
  89(2), 395–407, https://doi.org/10.1090/S0002-9947-1958-0106258-7
  (iterated integrals as a multiplicative homomorphism: concatenation of
  paths maps to the tensor (Chen) product — the identity used internally to
  update the running signature one increment at a time);
- Lyons, T. (1998), "Differential equations driven by rough paths", Rev. Mat.
  Iberoamericana 14(2), 215–310, https://doi.org/10.4171/RMI/240 (the
  signature as a graded object of the tensor algebra T((R^d)); the truncated
  signature is invariant under reparametrization and translation, so it
  captures the *order of events*, not the time axis — Lyons, Caruana &
  Lévy, 2007, *Differential Equations Driven by Rough Paths*, Springer LNM
  1908, §2.2);
- Chevyrev, I. & Kormilitzin, A. (2016), "A primer on the signature method in
  machine learning", arXiv:1603.03788, https://arxiv.org/abs/1603.03788
  (discretization: the signature of a sampled stream is that of its
  piecewise-linear interpolation, computed by solving the tensor ODE
  dS = S ⊗ dX one increment at a time);
- Bonnier, P., Kidger, P., Arribas, I.P., Salvi, C. & Lyons, T. (2019), "Deep
  Signature Transforms", NeurIPS 32, arXiv:1905.08494,
  https://arxiv.org/abs/1905.08494 (the lead-lag augmentation in practice).

The truncated signature of a piecewise-linear path through increments
dx_1, ..., dx_T is S = ⊗_i exp(dx_i), where exp(dx) = (1, dx, dx⊗dx/2!,
...); each increment therefore updates the running signature by the
truncated Chen product S ← S ⊗ exp(dx) (Chen 1958, Thm 5.1; Chevyrev &
Kormilitzin 2016, §2.1–2.2). The flattened layout is the graded tensor
algebra: level k holds d^k entries in word order — the word (i_1, ..., i_k)
sits at flat index i_1 d^{k-1} + ... + i_k (first letter slowest, C-order
reshape of the (d, ..., d) tensor), and levels 1..order are concatenated
(level 0, the scalar 1, is omitted from the output).

The logsignature is the tensor logarithm (Magnus, 1954, Comm. Pure Appl.
Math. 7, 649–673; Blanes, Casas, Oteo & Ros, 2009, Phys. Rep. 470,
151–238) of the truncated signature, log(S) = Σ_{j≥1} (-1)^{j+1} (S - 1)^⊗j
/ j, which lands in the free Lie algebra; coordinates are returned in the
Lyndon basis — a Hall basis (Hall, 1950, Proc. Amer. Math. Soc. 1, 575–581;
Reutenauer, 1993, *Free Lie Algebras*, Oxford Univ. Press, §4.1–4.5), with
basis elements built by the standard factorization w = uv (v the longest
proper Lyndon suffix) and the bracket [P_u, P_v] = P_u P_v - P_v P_u. The
dimension per level is the Witt number l_k(d) = (1/k) Σ_{j|k} μ(j) d^{k/j}
(Witt, 1937, J. Reine Angew. Math. 177, 152–160); the output packs levels
1..order, each sorted lexicographically over its Lyndon words. Restricted
to order ≤ 4 (dense d^4 Lyndon enumeration); anything above fails closed.

The lead-lag transform interleaves a d-dimensional path with a lagged copy
into a 2d-dimensional path (Chevyrev & Kormilitzin 2016, §3.2; Bonnier et
al. 2019, §2): on each sampling interval the lead coordinate moves first,
then the lag coordinate. For a 2-D closed loop the antisymmetric level-2
combination (1/2) (Sig^2_{lead_1, lag_2} - Sig^2_{lead_2, lag_1}) equals
the Lévy signed area A = (1/2) ∮ (x dy - y dx) = (1/2) Σ_i (x_i y_{i+1}
- x_{i+1} y_i) (the shoelace formula; Lévy, 1948, *Processus stochastiques
et mouvement brownien*, Gauthier-Villars; Chevyrev & Kormilitzin 2016,
§3.2) exactly for the piecewise-linear interpolation.

The signature kernel of Kidger & Foster (2020), "The signature kernel is
the solution of a Goursat PDE", arXiv:2005.08328,
https://arxiv.org/abs/2005.08328 (see also Salvi, Cass, Foster, Lyons &
Yang, 2021, SIAM J. Math. Data Sci. 3(3), 873–899), is the inner product
k(x, y) = ⟨Sig(x), Sig(y)⟩ = Σ_k ⟨Sig^k(x), Sig^k(y)⟩ of the two signatures
in the tensor-algebra feature space. This module implements the order-m
truncation of that kernel, k_m(x, y) = Σ_{k=0}^{m} σ^{2k} ⟨Sig^k(x),
Sig^k(y)⟩ (level 0 contributes 1), with σ rescaling the increments — i.e.
k_m(x, y; σ) = k_m(σx, σy; 1). It is a positive-definite kernel (feature map
x ↦ truncated signature of σx). This is the truncated inner-product form,
NOT the closed-form exp(σ² ⟨Sig, Sig⟩) expression, which only arises for the
full (untruncated) signature under additional structure; the untruncated
limit of k_m as m → ∞ is the kernel studied by Kidger & Foster (2020, §2).

Honesty: outputs are path features (signatures, logsignatures, areas,
kernel values) — geometry/rough-path quantities only. No Sharpe/Sortino/
P&L content; no live-trading claims (AGENTS.md honesty contract).

Conventions: numpy core, fail-closed edges (ValueError on non-2-D paths,
fewer than 2 points, non-finite entries, or order outside its supported
range), word-order tensor layout documented above, Lyndon/Hall logsignature
basis.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

__all__ = [
    "lead_lag_transform",
    "logsignature",
    "signature",
    "signature_kernel",
]

_MAX_ORDER = 6
_MAX_LOGSIGNATURE_ORDER = 4
_LYNDON_RESIDUAL_TOL = 1e-9


def _as_path(path: Array | Iterable[Iterable[float]]) -> Array:
    """Validate and normalize a path to a finite float64 (T+1, d) array."""
    arr = np.asarray(path, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"path must be a 2-D array of shape (T+1, d); got ndim={arr.ndim}")
    if arr.shape[0] < 2:
        raise ValueError(f"path must have at least 2 points (T+1 >= 2); got {arr.shape[0]}")
    if arr.shape[1] < 1:
        raise ValueError("path must have at least one channel (d >= 1)")
    if not bool(np.all(np.isfinite(arr))):
        raise ValueError("path entries must be finite (NaN/inf rejected)")
    return arr


def _check_order(order: int, max_order: int = _MAX_ORDER) -> int:
    m = int(order)
    if m != order or m < 1 or m > max_order:
        raise ValueError(f"order must be an integer in [1, {max_order}]; got {order!r}")
    return m


def _signature_tensors(path: Array, order: int) -> list[Array]:
    """Truncated signature of a piecewise-linear path as a list of tensors.

    Returns levels 1..order, each of shape (d,)*k, with the implied level-0
    scalar equal to 1. Computed as S = ⊗_i exp(dx_i) by Chen's identity,
    updating S ← S ⊗ exp(dx) one increment at a time (levels updated high
    to low so each level reads pre-update lower levels).
    """
    dims = int(path.shape[1])
    levels: list[Array] = [np.zeros((dims,) * k, dtype=float) for k in range(1, order + 1)]
    for t in range(path.shape[0] - 1):
        dx = path[t + 1] - path[t]
        # powers[j-1] = dx^{⊗ j} / j!
        powers: list[Array] = [dx]
        for j in range(2, order + 1):
            powers.append(np.tensordot(powers[-1], dx, axes=0) / float(j))
        for k in range(order, 0, -1):
            acc = powers[k - 1].copy()
            for j in range(1, k):
                acc = acc + np.tensordot(levels[k - j - 1], powers[j - 1], axes=0)
            levels[k - 1] = levels[k - 1] + acc
    return levels


def _flatten(tensors: list[Array]) -> Array:
    """Concatenate tensor levels 1..m into the word-order flat layout."""
    return np.concatenate([t.reshape(-1) for t in tensors])


def _chen_product(a: list[Array], b: list[Array], order: int) -> list[Array]:
    """Truncated tensor product (a ⊗ b) with both implied level-0 terms = 1.

    (a ⊗ b)_k = a_k + b_k + Σ_{i+j=k, i,j≥1} a_i ⊗ b_j — Chen's identity.
    """
    dims = int(a[0].shape[0])
    out: list[Array] = []
    for k in range(1, order + 1):
        acc = a[k - 1] + b[k - 1]
        for i in range(1, k):
            acc = acc + np.tensordot(a[i - 1], b[k - i - 1], axes=0)
        out.append(np.asarray(acc, dtype=float).reshape((dims,) * k))
    return out


def _strict_product(a: list[Array], b: list[Array], order: int) -> list[Array]:
    """Truncated tensor product with both implied level-0 terms = 0.

    (a ⊗ b)_k = Σ_{i+j=k, i,j≥1} a_i ⊗ b_j — used for powers of S - 1 in
    the series log.
    """
    dims = int(a[0].shape[0])
    out: list[Array] = []
    for k in range(1, order + 1):
        acc = np.zeros((dims,) * k, dtype=float)
        for i in range(1, k):
            acc = acc + np.tensordot(a[i - 1], b[k - i - 1], axes=0)
        out.append(acc)
    return out


def _tensor_log(tensors: list[Array], order: int) -> list[Array]:
    """Series log of a truncated signature: log(1 + x) = Σ (-1)^{j+1} x^{⊗j}/j.

    ``tensors`` are levels 1..order of a group-like element whose level-0
    term is 1. The result is a Lie series, returned in word coordinates.
    """
    dims = int(tensors[0].shape[0])
    log_levels: list[Array] = [np.zeros((dims,) * k, dtype=float) for k in range(1, order + 1)]
    power: list[Array] = [t.copy() for t in tensors]
    for j in range(1, order + 1):
        sign = 1.0 if j % 2 == 1 else -1.0
        for k in range(j, order + 1):
            log_levels[k - 1] = log_levels[k - 1] + (sign / float(j)) * power[k - 1]
        if j < order:
            power = _strict_product(power, tensors, order)
    return log_levels


def _tensor_exp(x_levels: list[Array], order: int) -> list[Array]:
    """Truncated tensor exponential of a level-0-free series: exp(x) = Σ x^{⊗j}/j!."""
    dims = int(x_levels[0].shape[0])
    exp_levels: list[Array] = [np.zeros((dims,) * k, dtype=float) for k in range(1, order + 1)]
    power: list[Array] = [t.copy() for t in x_levels]
    factorial = 1.0
    for j in range(1, order + 1):
        factorial *= float(j)
        for k in range(j, order + 1):
            exp_levels[k - 1] = exp_levels[k - 1] + power[k - 1] / factorial
        if j < order:
            power = _strict_product(power, x_levels, order)
    return exp_levels


def _mobius(n: int) -> int:
    """Möbius function μ(n), naive factorization (n ≤ order ≤ 6 here)."""
    result = 1
    rest = n
    p = 2
    while p * p <= rest:
        if rest % p == 0:
            rest //= p
            if rest % p == 0:
                return 0
            result = -result
        p += 1
    if rest > 1:
        result = -result
    return result


def _witt_dimension(dims: int, order: int) -> int:
    """Dimension of the degree-order component of the free Lie algebra on d letters."""
    return sum(
        sum(_mobius(j) * dims ** (k // j) for j in range(1, k + 1) if k % j == 0) // k
        for k in range(1, order + 1)
    )


def _is_lyndon(word: tuple[int, ...]) -> bool:
    """A word is Lyndon iff it is strictly smaller than every nontrivial rotation."""
    n = len(word)
    return all(word < word[i:] + word[:i] for i in range(1, n))


def _lyndon_words(dims: int, order: int) -> list[tuple[int, ...]]:
    """All Lyndon words over d letters, packed by length then lexicographic order."""
    out: list[tuple[int, ...]] = []
    for k in range(1, order + 1):
        for word in _product_words(dims, k):
            if _is_lyndon(word):
                out.append(word)
    return out


def _product_words(dims: int, length: int) -> Iterable[tuple[int, ...]]:
    if length == 1:
        for i in range(dims):
            yield (i,)
        return
    for head in _product_words(dims, length - 1):
        for i in range(dims):
            yield head + (i,)


def _standard_factorization(word: tuple[int, ...]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Standard factorization of a Lyndon word: v = longest proper Lyndon suffix."""
    for i in range(1, len(word)):
        cand = word[i:]
        if _is_lyndon(cand):
            return word[:i], cand
    raise ValueError(f"no Lyndon suffix found for {word!r} (not a Lyndon word?)")


def _lie_element(
    word: tuple[int, ...], memo: dict[tuple[int, ...], dict[tuple[int, ...], float]]
) -> dict[tuple[int, ...], float]:
    """Lyndon-basis Lie element P_w in word coordinates (tensor expansion).

    P_a = a for a letter; P_w = P_u P_v - P_v P_u for the standard
    factorization w = uv. The expansion is triangular: its smallest word
    (lex) is w itself with coefficient 1 (Hall 1950; Reutenauer 1993,
    Thm 4.5 / §5.3).
    """
    cached = memo.get(word)
    if cached is not None:
        return cached
    if len(word) == 1:
        out: dict[tuple[int, ...], float] = {word: 1.0}
    else:
        u, v = _standard_factorization(word)
        pu = _lie_element(u, memo)
        pv = _lie_element(v, memo)
        merged: dict[tuple[int, ...], float] = {}
        for w1, c1 in pu.items():
            for w2, c2 in pv.items():
                merged[w1 + w2] = merged.get(w1 + w2, 0.0) + c1 * c2
        for w1, c1 in pv.items():
            for w2, c2 in pu.items():
                merged[w1 + w2] = merged.get(w1 + w2, 0.0) - c1 * c2
        out = {w: c for w, c in merged.items() if c != 0.0}
    memo[word] = out
    return out


def _lyndon_coordinates(log_levels: list[Array], order: int) -> Array:
    """Coordinates of a Lie series (word coordinates) in the Lyndon basis.

    Triangular extraction: Lyndon words are visited by length then
    lexicographic order; at each word the residual coefficient equals its
    basis coordinate, which is then subtracted out (the basis expansion of
    P_w contains only words lexicographically ≥ w).
    """
    dims = int(log_levels[0].shape[0])
    words = _lyndon_words(dims, order)
    residual: dict[tuple[int, ...], float] = {}
    for tensor in log_levels:
        for idx in np.ndindex(tensor.shape):
            residual[idx] = float(tensor[idx])
    memo: dict[tuple[int, ...], dict[tuple[int, ...], float]] = {}
    coords: list[float] = []
    for w in words:
        c = residual.get(w, 0.0)
        coords.append(c)
        if c != 0.0:
            for pw, pc in _lie_element(w, memo).items():
                residual[pw] = residual.get(pw, 0.0) - c * pc
    leftover = max((abs(v) for v in residual.values()), default=0.0)
    scale = max((abs(c) for c in coords), default=1.0)
    if leftover > _LYNDON_RESIDUAL_TOL * (1.0 + scale):
        raise ArithmeticError(
            f"series-log extraction failed: residual {leftover:g} (input not Lie?)"
        )
    return np.asarray(coords, dtype=float)


def _lie_series_from_coordinates(coords: Array, dims: int, order: int) -> list[Array]:
    """Reconstruct a Lie series (word coordinates, levels 1..order) from Lyndon coords."""
    words = _lyndon_words(dims, order)
    if int(coords.shape[0]) != len(words):
        raise ValueError(f"expected {len(words)} Lyndon coordinates; got {coords.shape[0]}")
    levels: list[Array] = [np.zeros((dims,) * k, dtype=float) for k in range(1, order + 1)]
    memo: dict[tuple[int, ...], dict[tuple[int, ...], float]] = {}
    for w, c in zip(words, coords, strict=True):
        if c != 0.0:
            for pw, pc in _lie_element(w, memo).items():
                tensor = levels[len(pw) - 1]
                tensor[pw] = tensor[pw] + float(c) * pc
    return levels


def signature(path: Array | Iterable[Iterable[float]], order: int) -> Array:
    """Truncated signature of a piecewise-linear path, levels 1..order flattened.

    Parameters
    ----------
    path:
        Array-like of shape (T+1, d) — T increments, d channels. The
        signature is that of the piecewise-linear interpolation (Chevyrev &
        Kormilitzin 2016, §2.2), making it invariant to the sampling clock
        (Lyons 1998) and to translations of the path.
    order:
        Truncation level, 1..6.

    Returns
    -------
    Array
        Concatenated levels 1..order in word order (level k first, d^k
        entries; word (i_1..i_k) at flat offset i_1 d^{k-1} + ... + i_k).
        Level 0 (the scalar 1) is omitted.
    """
    arr = _as_path(path)
    m = _check_order(order)
    return _flatten(_signature_tensors(arr, m))


def logsignature(path: Array | Iterable[Iterable[float]], order: int) -> Array:
    """Logsignature of a path: Lyndon-basis coordinates of the series log.

    The truncated signature S is group-like, so log(S) lies in the free Lie
    algebra (Magnus 1954); coordinates are returned in the Lyndon basis
    (Hall 1950; Reutenauer 1993), packed by level 1..order, each level
    sorted lexicographically over its Lyndon words. Total dimension is the
    sum of Witt numbers l_k(d) (Witt 1937). Restricted to order ≤ 4.

    ``exp(logsignature)`` reconstructs the truncated signature (group-like
    property); the logsignature of a straight line is exactly its level-1
    term delta.
    """
    arr = _as_path(path)
    m = _check_order(order, max_order=_MAX_LOGSIGNATURE_ORDER)
    tensors = _signature_tensors(arr, m)
    log_levels = _tensor_log(tensors, m)
    return _lyndon_coordinates(log_levels, m)


def lead_lag_transform(path: Array | Iterable[Iterable[float]]) -> Array:
    """Lead-lag augmentation: a d-dimensional path becomes a 2d-dimensional path.

    On each sampling interval the lead coordinate moves first (lag held at
    its start), then the lag coordinate moves (lead held at its end); the
    two copies are interleaved column-wise as (lead, lag). For a 2-D closed
    loop the antisymmetric level-2 combination (1/2) (Sig^2_{lead_1, lag_2}
    - Sig^2_{lead_2, lag_1}) equals the Lévy signed area (1/2) ∮ (x dy - y
    dx) exactly (Chevyrev & Kormilitzin 2016, §3.2; Bonnier et al. 2019,
    §2); columns are ordered (lead_1..lead_d, lag_1..lag_d).
    """
    arr = _as_path(path)
    repeated = np.repeat(arr, 2, axis=0)
    lead = repeated[1:]
    lag = repeated[:-1]
    return np.concatenate([lead, lag], axis=1)


def signature_kernel(
    path_a: Array | Iterable[Iterable[float]],
    path_b: Array | Iterable[Iterable[float]],
    order: int,
    sigma: float = 1.0,
) -> float:
    """Order-m truncated signature kernel k_m(a, b; σ) between two paths.

    k_m(a, b; σ) = Σ_{k=0}^{m} σ^{2k} ⟨Sig^k(a), Sig^k(b)⟩ — the inner
    product of the truncated signatures of the σ-scaled paths in the graded
    tensor-algebra feature space (Kidger & Foster 2020, §2; the m → ∞ limit
    at σ = 1 is their signature kernel). Positive definite by construction
    (feature map a ↦ Sig^{≤m}(σa)); this is the truncated inner-product
    form, not the closed-form exp(σ² ⟨Sig, Sig⟩) expression.

    Parameters
    ----------
    path_a, path_b:
        Paths of shape (T+1, d) with the same channel count d.
    order:
        Truncation level, 1..6.
    sigma:
        Positive finite increment scale; k_m(a, b; σ) = k_m(σa, σb; 1).
    """
    arr_a = _as_path(path_a)
    arr_b = _as_path(path_b)
    if int(arr_a.shape[1]) != int(arr_b.shape[1]):
        raise ValueError(
            f"paths must share the same channel count d; got {arr_a.shape[1]} and {arr_b.shape[1]}"
        )
    m = _check_order(order)
    s = float(sigma)
    if not math.isfinite(s) or s <= 0.0:
        raise ValueError(f"sigma must be positive and finite; got {sigma!r}")
    tensors_a = _signature_tensors(arr_a, m)
    tensors_b = _signature_tensors(arr_b, m)
    total = 1.0  # level-0 inner product
    sigma_sq = s * s
    scale = sigma_sq
    for k in range(1, m + 1):
        total += scale * float(np.sum(tensors_a[k - 1] * tensors_b[k - 1]))
        scale *= sigma_sq
    return float(total)
