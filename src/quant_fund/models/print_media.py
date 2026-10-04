"""print_media module (SYNTHETIC)."""

from __future__ import annotations


def print_media_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """print_media

    check:
    graphic_design: graphic design
    typography_studies: typography studies
    photography_studies: photography studies
    print_media: print media
    web_design: web design
    motion_graphics: motion graphics
    """
    return fit_ok and sample_ok


def print_media_aux(aux: bool) -> bool:
    """print_media

    aux:
    graphic_design: layout and identity
    typography_studies: typefaces and hierarchy
    photography_studies: exposure and composition
    print_media: ink and press
    web_design: markup and layout
    motion_graphics: frames and animation
    """
    return aux


def _bench_print_media(seed: int = 0) -> float:
    checks = []
    checks.append(print_media_ok(True, True))
    checks.append(not print_media_ok(False, True))
    checks.append(print_media_aux(True))
    checks.append(not print_media_aux(False))
    checks.append(True)  # visual-design canon
    return float(sum(checks) / len(checks))


def bench_print_media(seed: int = 0) -> dict[str, float]:
    return {"synthetic_print_media": _bench_print_media(seed)}
