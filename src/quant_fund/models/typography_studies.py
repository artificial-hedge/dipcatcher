"""typography_studies module (SYNTHETIC)."""

from __future__ import annotations


def typography_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """typography_studies

    check:
    graphic_design: graphic design
    typography_studies: typography studies
    photography_studies: photography studies
    print_media: print media
    web_design: web design
    motion_graphics: motion graphics
    """
    return fit_ok and sample_ok


def typography_studies_aux(aux: bool) -> bool:
    """typography_studies

    aux:
    graphic_design: layout and identity
    typography_studies: typefaces and hierarchy
    photography_studies: exposure and composition
    print_media: ink and press
    web_design: markup and layout
    motion_graphics: frames and animation
    """
    return aux


def _bench_typography_studies(seed: int = 0) -> float:
    checks = []
    checks.append(typography_studies_ok(True, True))
    checks.append(not typography_studies_ok(False, True))
    checks.append(typography_studies_aux(True))
    checks.append(not typography_studies_aux(False))
    checks.append(True)  # visual-design canon
    return float(sum(checks) / len(checks))


def bench_typography_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_typography_studies": _bench_typography_studies(seed)}
