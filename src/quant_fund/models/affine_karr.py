"""Karr affine-equality abstract domain (affine relations a.x = b).

Internally a state is an affine subspace {p + span(D)}: a particular point
plus a direction basis — the representation under which Karr's join is the
affine hull. Constraints are rows [a | c] meaning a.x + c = 0; conversion to
point+span solves A.x = -c and takes the kernel basis. All arithmetic is
exact Fraction.
"""

from __future__ import annotations

from fractions import Fraction

_SEED = 20261231 + 1008

Vec = list[Fraction]
Rows = list[Vec]


def _rref(rows: Rows) -> Rows:
    rows = [r[:] for r in rows if any(v != 0 for v in r)]
    ncols = len(rows[0]) if rows else 0
    used: set[int] = set()
    out: Rows = []
    for col in range(ncols):
        piv = next((i for i, r in enumerate(rows) if i not in used and r[col] != 0), None)
        if piv is None:
            continue
        rows[piv] = [v / rows[piv][col] for v in rows[piv]]
        for i, r in enumerate(rows):
            if i != piv and r[col] != 0:
                rows[i] = [a - r[col] * b for a, b in zip(r, rows[piv], strict=True)]
        used.add(piv)
        out.append(rows[piv])
    return out


def _kernel(a: Rows, n: int) -> Rows:
    """Basis for {x : A.x = 0}."""
    rr = _rref(a)
    piv_cols: list[int] = []
    col = 0
    for r in rr:
        while col < n and r[col] == 0:
            col += 1
        piv_cols.append(col)
        col += 1
    free = [j for j in range(n) if j not in piv_cols]
    basis: Rows = []
    for f in free:
        v = [Fraction(0)] * n
        v[f] = Fraction(1)
        for r, pc in zip(rr, piv_cols, strict=True):
            if r[f] != 0:
                v[pc] = -r[f]
        basis.append(v)
    return basis


def _solve(a: Rows, n: int) -> Vec | None:
    """Particular solution of A.x = -c for rows [a|c]."""
    rr = _rref(a)
    x = [Fraction(0)] * n
    for r in rr:
        # find pivot col
        pc = next((j for j in range(n) if r[j] != 0), None)
        if pc is None:
            if r[n] != 0:
                return None  # inconsistent
            continue
        x[pc] = -r[n]  # row says x_pc + rest = -c... our rows: a.x + c = 0
    return x


def _in_span(v: Vec, basis: Rows) -> bool:
    if not basis:
        return all(x == 0 for x in v)
    m = [b[:] for b in basis] + [v[:]]
    return len(_rref(m)) == len(_rref(basis))


def _constraints_to_space(rows: Rows, n: int) -> tuple[Vec, Rows]:
    p = _solve(rows, n)
    if p is None:
        raise ValueError("inconsistent constraints")
    return p, _kernel(rows, n)


def _space_to_constraints(p: Vec, dirs: Rows, n: int) -> Rows:
    """Rows [a|c]: a.x + c = 0 satisfied by {p + span(dirs)}."""
    if not dirs:
        orth = [[Fraction(1) if i == j else Fraction(0) for i in range(n)] for j in range(n)]
    else:
        # orthogonal complement of direction space
        orth = _kernel(dirs, n)
    rows: Rows = []
    for a in orth:
        c = Fraction(sum(ai * pi for ai, pi in zip(a, p, strict=True)))
        rows.append(list(a) + [-c])
    return _rref(rows)


class AffineState:
    """Affine subspace {point + span(dirs)} with exact Fraction math."""

    def __init__(self, point: Vec | None = None, dirs: Rows | None = None, n: int = 0):
        self.n = n
        self.point = point if point is not None else [Fraction(0)] * n
        self.dirs = (
            dirs
            if dirs is not None
            else [[Fraction(1) if i == j else Fraction(0) for i in range(n)] for j in range(n)]
        )

    @staticmethod
    def top(n: int) -> AffineState:
        return AffineState(None, None, n)

    @staticmethod
    def from_constraints(rows: Rows, n: int) -> AffineState:
        p, dirs = _constraints_to_space(rows, n)
        return AffineState(p, dirs, n)

    def constraints(self) -> Rows:
        return _space_to_constraints(self.point, self.dirs, self.n)

    def assign(self, x: int, coeffs: Vec, const: Fraction) -> None:
        """x := coeffs·vars + const — affine image of the subspace."""
        new_dirs: Rows = []
        for d in self.dirs:
            nd = d[:]
            nd[x] = Fraction(sum(c * di for c, di in zip(coeffs, d, strict=True)))
            new_dirs.append(nd)
        new_point = self.point[:]
        new_point[x] = (
            Fraction(sum(c * pi for c, pi in zip(coeffs, self.point, strict=True))) + const
        )
        self.point, self.dirs = new_point, _rref(new_dirs)

    def join(self, other: AffineState) -> AffineState:
        """Affine hull of two subspaces."""
        delta = [a - b for a, b in zip(other.point, self.point, strict=True)]
        dirs = _rref(self.dirs + other.dirs + [delta])
        return AffineState(self.point[:], dirs, self.n)

    def implies(self, coeffs: Vec, const: Fraction) -> bool:
        """Does a.x = const hold on the whole subspace?"""
        return (
            _in_span(list(coeffs) + [-const], _rref(self.constraints()))
            if self.constraints()
            else (all(v == 0 for v in coeffs) and const == 0)
        )


def bench_affine_karr(seed: int = _SEED) -> dict[str, float]:
    del seed
    F = Fraction
    checks: list[bool] = []
    # x := 3; y := 2x  =>  y = 2x and x = 3 => y = 6 derivable
    s = AffineState.top(2)
    s.assign(0, [F(0), F(0)], F(3))
    s.assign(1, [F(2), F(0)], F(0))
    checks.append(s.implies([F(1), F(0)], F(3)))
    checks.append(s.implies([F(-2), F(1)], F(0)))
    checks.append(s.implies([F(0), F(1)], F(6)))
    # z = x + y with x=1,y=2 => z=3
    s2 = AffineState.top(3)
    s2.assign(0, [F(0)] * 3, F(1))
    s2.assign(1, [F(0)] * 3, F(2))
    s2.assign(2, [F(1), F(1), F(0)], F(0))
    checks.append(s2.implies([F(0), F(0), F(1)], F(3)))
    # join of x=1,y=5 and x=2,y=5: keeps y=5, drops x
    j = AffineState.from_constraints([[F(1), F(0), F(-1)], [F(0), F(1), F(-5)]], 2).join(
        AffineState.from_constraints([[F(1), F(0), F(-2)], [F(0), F(1), F(-5)]], 2)
    )
    checks.append(j.implies([F(0), F(1)], F(5)))
    checks.append(not j.implies([F(1), F(0)], F(1)) and not j.implies([F(1), F(0)], F(2)))
    return {"synthetic_affine_karr": float(sum(checks)) / len(checks)}
