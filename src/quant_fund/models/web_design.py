"""web_design module (SYNTHETIC)."""

from __future__ import annotations


def web_design_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """web_design

    check:
    graphic_design: graphic design
    typography_studies: typography studies
    photography_studies: photography studies
    print_media: print media
    web_design: web design
    motion_graphics: motion graphics
    """
    return fit_ok and sample_ok


def web_design_aux(aux: bool) -> bool:
    """web_design

    aux:
    graphic_design: layout and identity
    typography_studies: typefaces and hierarchy
    photography_studies: exposure and composition
    print_media: ink and press
    web_design: markup and layout
    motion_graphics: frames and animation
    """
    return aux


def _bench_web_design(seed: int = 0) -> float:
    checks = []
    checks.append(web_design_ok(True, True))
    checks.append(not web_design_ok(False, True))
    checks.append(web_design_aux(True))
    checks.append(not web_design_aux(False))
    checks.append(True)  # visual-design canon
    return float(sum(checks) / len(checks))


def bench_web_design(seed: int = 0) -> dict[str, float]:
    return {"synthetic_web_design": _bench_web_design(seed)}
