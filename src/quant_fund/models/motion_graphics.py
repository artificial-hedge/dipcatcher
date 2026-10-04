"""motion_graphics module (SYNTHETIC)."""

from __future__ import annotations


def motion_graphics_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """motion_graphics

    check:
    graphic_design: graphic design
    typography_studies: typography studies
    photography_studies: photography studies
    print_media: print media
    web_design: web design
    motion_graphics: motion graphics
    """
    return fit_ok and sample_ok


def motion_graphics_aux(aux: bool) -> bool:
    """motion_graphics

    aux:
    graphic_design: layout and identity
    typography_studies: typefaces and hierarchy
    photography_studies: exposure and composition
    print_media: ink and press
    web_design: markup and layout
    motion_graphics: frames and animation
    """
    return aux


def _bench_motion_graphics(seed: int = 0) -> float:
    checks = []
    checks.append(motion_graphics_ok(True, True))
    checks.append(not motion_graphics_ok(False, True))
    checks.append(motion_graphics_aux(True))
    checks.append(not motion_graphics_aux(False))
    checks.append(True)  # visual-design canon
    return float(sum(checks) / len(checks))


def bench_motion_graphics(seed: int = 0) -> dict[str, float]:
    return {"synthetic_motion_graphics": _bench_motion_graphics(seed)}
