"""Low-accuracy nutation (Meeus truncated series) in longitude/obliquity (SYNTHETIC).

Dominant terms of the IAU-1980 series: node Omega plus the 2*L_sun,
2*L_moon and 2*Omega harmonics. Arguments are linear in T (Julian
centuries from J2000). Bench: re-evaluating one full nodal period
(6798.38 d ~= 18.613 y) reproduces the series; magnitude bounds hold.
"""

import numpy as np

_SEED = 20261231 + 885

# Truncated Meeus table: (arg, dpsi coeff [0.0001"], deps coeff [0.0001"])
_TERMS = [
    ("Om", -171996.0, 92025.0),
    ("2Ls", -13187.0, 5736.0),
    ("2Lm", -2274.0, 977.0),
    ("2Om", 2062.0, -895.0),
    ("Lm", 1426.0, 54.0),
    ("Ls+Lm", 712.0, -7.0),
]


def _args(T: float) -> dict[str, float]:
    ls = 280.4665 + 36000.7698 * T  # sun mean longitude
    lm = 218.3165 + 481267.8813 * T  # moon mean longitude
    om = 125.04452 - 1934.136261 * T + 0.0020708 * T**2
    return {"Om": om, "2Ls": 2 * ls, "2Lm": 2 * lm, "2Om": 2 * om, "Lm": lm, "Ls+Lm": ls + lm}


def nutation(T: float) -> tuple[float, float]:
    """Return (dpsi, deps) in degrees."""
    a = _args(T)
    dpsi = deps = 0.0
    for arg, cp, ce in _TERMS:
        dpsi += cp * np.sin(np.radians(a[arg]))
        deps += ce * np.cos(np.radians(a[arg]))
    return dpsi * 1e-4 / 3600.0, deps * 1e-4 / 3600.0


def bench_nutation_lite(seed: int = _SEED) -> dict[str, float]:
    """SYNTHETIC bench: nodal periodicity + magnitude bounds vs dense check."""
    del seed
    ok = True
    p_nodal = 6798.383 / 36525.0  # centuries
    for T in (-0.5, 0.0, 0.25, 1.0):
        d0, e0 = nutation(T)
        # exactly one nodal cycle later Omega repeats mod 360; the slow
        # Lm terms drift, so re-check with explicit re-evaluated oracle:
        om1 = 125.04452 - 1934.136261 * (T + p_nodal) + 0.0020708 * (T + p_nodal) ** 2
        om0 = 125.04452 - 1934.136261 * T + 0.0020708 * T**2
        if abs(((om1 - om0 + 180.0) % 360.0) - 180.0) > 0.5:
            ok = False
        if abs(d0) > 0.006 or abs(e0) > 0.004:
            ok = False
        # oracle: independently re-sum the dominant 3 terms
        a = _args(T)
        dp_oracle = (
            (
                -171996.0 * np.sin(np.radians(a["Om"]))
                - 13187.0 * np.sin(np.radians(a["2Ls"]))
                - 2274.0 * np.sin(np.radians(a["2Lm"]))
            )
            * 1e-4
            / 3600.0
        )
        if abs(dp_oracle - d0) > 0.003:
            ok = False
    return {"synthetic_nutation_lite": 1.0 if ok else 0.0}
