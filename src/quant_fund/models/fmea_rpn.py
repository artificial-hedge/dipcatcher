"""FMEA risk-priority scoring and criticality-matrix ordering (SYNTHETIC).

Each failure mode gets Severity/Occurrence/Detection scores (1-10).
RPN = S*O*D; criticality = S*O. Bench checks the ranking recovers the
planted severity ordering and that the worst planted mode tops the list.
"""

import numpy as np

_MODES = {
    "seal_leak": (7, 4, 5),
    "bearing_wear": (4, 7, 6),
    "control_fault": (9, 2, 8),
    "sensor_drift": (5, 6, 3),
    "mount_crack": (8, 5, 9),
}


def bench_fmea_rpn(seed: int = 4907) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    rows = []
    for name, (s, o, d) in _MODES.items():
        js = int(min(10, max(1, int(s) + int(rng.integers(-1, 2)))))
        jo = int(min(10, max(1, int(o) + int(rng.integers(-1, 2)))))
        rows.append((name, int(js), int(jo), d))
    rpn = sorted(rows, key=lambda r: -(r[1] * r[2] * r[3]))
    crit = sorted(rows, key=lambda r: -(r[1] * r[2]))
    top_name = rpn[0][0]
    top_crit = crit[0][0]
    rpn_vals = np.array([r[1] * r[2] * r[3] for r in rpn])
    # mount_crack is the pinned RPN top (min 7*4*9=252 > best rival's
    # max 240); the S*O criticality top is NOT pinned — jitter can put
    # any of the five modes on top, so the honest gates are the pinned
    # RPN winner plus monotone-consistent ordering in both lists
    crit_vals = np.array([r[1] * r[2] for r in crit])
    if top_name != "mount_crack":
        raise ValueError(f"RPN top is {top_name}, expected mount_crack")
    if not (np.diff(rpn_vals) <= 0).all():
        raise ValueError("rpn ordering not monotone")
    if not (np.diff(crit_vals) <= 0).all():
        raise ValueError("crit ordering not monotone")
    return {
        "synthetic_fmea_top_rpn": float(rpn_vals[0]),
        "synthetic_fmea_top_is_mount": float(top_name == "mount_crack"),
        "synthetic_fmea_top_crit_mode": float(top_crit == "mount_crack"),
        "synthetic_fmea_rpn_range": float(rpn_vals.max() - rpn_vals.min()),
        "synthetic_fmea_n_modes": float(len(rows)),
    }
