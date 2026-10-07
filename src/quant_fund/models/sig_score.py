"""Multi-feature rule scoring (defensive, YARA-style) — wave 286 (SYNTHETIC).

Rules = weighted feature matches; score >= threshold raises detection.
Validated: malware-ish sample hits the rule, benign sample does not.
"""

_SEED = 20261231 + 805


def score(
    features: dict[str, bool], rules: dict[str, float], threshold: float
) -> tuple[float, bool]:
    s = sum(w for f, w in rules.items() if features.get(f, False))
    return float(s), s >= threshold


def bench_sig_score(seed: int = _SEED) -> dict[str, float]:
    rules = {
        "packed_section": 3.0,
        "suspicious_imports": 2.5,
        "writes_temp": 1.0,
        "unsigned": 1.5,
        "long_entropy_section": 2.0,
        "signed_cert": -4.0,
    }
    mal = {
        "packed_section": True,
        "suspicious_imports": True,
        "writes_temp": True,
        "unsigned": True,
        "long_entropy_section": True,
    }
    ben = {"writes_temp": True, "signed_cert": True}
    s_mal, hit_mal = score(mal, rules, 6.0)
    s_ben, hit_ben = score(ben, rules, 6.0)
    return {"synthetic_sig_score": float(hit_mal and not hit_ben and s_mal >= 9.0 and s_ben < 0)}
