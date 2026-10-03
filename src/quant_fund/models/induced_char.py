"""Induced characters and Frobenius reciprocity (SYNTHETIC)."""

from __future__ import annotations


def induced(h_char: dict, h: list, g: list, conj_class) -> dict:
    """Ind_H^G chi(g) = (1/|H|) sum_{x in G, x^-1 g x in H} chi(x^-1 g x)."""
    out = {}
    for gx in g:
        s = 0.0
        for x in g:
            t = conj_class(x, gx)
            if t in h:
                s += h_char.get(t, 0.0)
        out[gx] = s / len(h)
    return out


def _bench_induced_char(seed: int = 0) -> float:
    checks = []
    # G = Z/4 additive, H = {0, 2}; trivial char of H induces to
    # character that is 2 on {0,2} and 0 elsewhere
    g = [0, 1, 2, 3]
    h = [0, 2]

    def conj(x: int, y: int) -> int:
        return y  # abelian: x^-1 y x = y

    ind = induced({0: 1.0, 2: 1.0}, h, g, conj)
    checks.append(abs(ind[0] - 2.0) < 1e-9)
    checks.append(abs(ind[2] - 2.0) < 1e-9)
    checks.append(ind[1] == 0.0 and ind[3] == 0.0)
    # induced char degree = [G:H] * deg(chi) = 2
    checks.append(abs(ind[0] - 2 * 1.0) < 1e-9)
    # Frobenius reciprocity: <Ind 1_H, 1_G>_G = <1_H, Res 1_G>_H = 1
    inner = sum(ind[x] * 1.0 for x in g) / len(g)
    checks.append(abs(inner - 1.0) < 1e-9)
    return float(sum(checks) / len(checks))


def bench_induced_char(seed: int = 0) -> dict[str, float]:
    return {"synthetic_induced_char": _bench_induced_char(seed)}
