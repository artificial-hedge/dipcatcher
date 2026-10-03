"""Multipoint evaluation + fast interpolation over F_p (SYNTHETIC bench)."""

from __future__ import annotations


def trim(f: list[int]) -> list[int]:
    while f and f[-1] == 0:
        f = f[:-1]
    return f


def multipoint(f: list[int], xs: list[int], p: int) -> list[int]:
    return [_peval(f, x, p) for x in xs]


def _peval(f: list[int], x: int, p: int) -> int:
    out = 0
    for c in reversed(f):
        out = (out * x + c) % p
    return out


def build_prod_tree(xs: list[int], p: int) -> list:
    """Leaves (x - xi); pairwise product tree."""
    level = [[(-x) % p, 1] for x in xs]
    tree = [level]
    while len(level) > 1:
        nxt = []
        for i in range(0, len(level), 2):
            if i + 1 < len(level):
                nxt.append(_pmul(level[i], level[i + 1], p))
            else:
                nxt.append(level[i])
        tree.append(nxt)
        level = nxt
    return tree


def _pmul(a: list[int], b: list[int], p: int) -> list[int]:
    out = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i + j] = (out[i + j] + x * y) % p
    return trim(out)


def _pmod(a: list[int], b: list[int], p: int) -> list[int]:
    a = list(a)
    while len(a) >= len(b) and a:
        k = len(a) - len(b)
        c = a[-1] * pow(b[-1], -1, p) % p
        for i in range(len(b)):
            a[i + k] = (a[i + k] - c * b[i]) % p
        a = trim(a)
    return a


def fast_multipoint(f: list[int], xs: list[int], p: int) -> list[int]:
    """Evaluate f at all xs via product-tree remainder descent."""
    tree = build_prod_tree(xs, p)
    n = len(xs)
    vals = [0] * n
    idx = list(range(n))

    def descend(poly: list[int], node_idx: int, level: int, ids: list[int]) -> None:
        if level == 0:
            vals[ids[0]] = (
                _peval(poly, xs[ids[0]], p) if len(poly) <= 1 else _peval(poly, xs[ids[0]], p)
            )
            vals[ids[0]] = poly[0] % p if len(poly) == 1 else _peval(poly, xs[ids[0]], p)
            return
        left = tree[level - 1][2 * node_idx]
        right = (
            tree[level - 1][2 * node_idx + 1] if 2 * node_idx + 1 < len(tree[level - 1]) else [1]
        )
        half = len(ids) // 2
        descend(_pmod(poly, left, p), 2 * node_idx, level - 1, ids[: len(left) - 1] or ids[:half])
        if 2 * node_idx + 1 < len(tree[level - 1]):
            descend(
                _pmod(poly, right, p),
                2 * node_idx + 1,
                level - 1,
                ids[len(left) - 1 :] or ids[half:],
            )

    if n == 1:
        return [_peval(f, xs[0], p)]
    root = tree[-1][0]
    descend(_pmod(f, root, p), 0, len(tree) - 1, idx)
    return vals


def newton_interp(xs: list[int], ys: list[int], p: int) -> list[int]:
    """Newton divided-difference coefficients."""
    n = len(xs)
    coef = list(ys)
    for j in range(1, n):
        for i in range(n - 1, j - 1, -1):
            coef[i] = (coef[i] - coef[i - 1]) * pow(xs[i] - xs[i - j], -1, p) % p
    # expand to standard basis
    out = [coef[0]]
    basis = [1]
    for i in range(1, n):
        basis = _pmul(basis, [(-xs[i - 1]) % p, 1], p)
        term = [c * coef[i] % p for c in basis]
        out = _padd(out, term, p)
    return trim(out)


def _padd(a: list[int], b: list[int], p: int) -> list[int]:
    n = max(len(a), len(b))
    return trim([((a[i] if i < len(a) else 0) + (b[i] if i < len(b) else 0)) % p for i in range(n)])


def _bench_poly_eval_interp(seed: int = 0) -> float:
    p = 101
    checks = []
    f = [1, 2, 3]  # 3x^2+2x+1
    xs = [0, 1, 2, 5]
    checks.append(multipoint(f, xs, p) == [_peval(f, x, p) for x in xs])
    # multipoint == direct
    checks.append(multipoint(f, xs, p) == [1, 6, 17, 86])
    # interpolate roundtrip
    ys = multipoint(f, xs, p)
    g = newton_interp(xs[:3], ys[:3], p)
    checks.append(all(_peval(g, xs[i], p) == ys[i] for i in range(3)))
    # product tree root = prod (x - xi)
    tree = build_prod_tree([1, 2], p)
    root = tree[-1][0]
    checks.append(_peval(root, 1, p) == 0 and _peval(root, 2, p) == 0)
    return sum(checks) / len(checks)


def bench_poly_eval_interp(seed: int = 0) -> dict[str, float]:
    return {"synthetic_poly_eval_interp": _bench_poly_eval_interp(seed)}
