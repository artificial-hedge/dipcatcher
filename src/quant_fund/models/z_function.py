"""Z-function (Z-algorithm) + Z-based pattern matching.

SYNTHETIC bench only.
"""

import numpy as np

_SEED = 20261231 + 915


def z_function(s: str) -> list[int]:
    n = len(s)
    z = [0] * n
    left = r = 0
    for i in range(1, n):
        if i <= r:
            z[i] = min(r - i + 1, z[i - left])
        while i + z[i] < n and s[z[i]] == s[i + z[i]]:
            z[i] += 1
        if i + z[i] - 1 > r:
            left, r = i, i + z[i] - 1
    z[0] = n
    return z


def z_search(text: str, pat: str) -> list[int]:
    if not pat:
        return list(range(len(text) + 1))
    s = pat + "\x00" + text
    z = z_function(s)
    off = len(pat) + 1
    return [i - off for i in range(off, len(s)) if z[i] >= len(pat)]


def bench_z_function(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    score = 0.0
    s = "".join(rng.choice(list("ab"), 400))
    z = z_function(s)
    truth = [len(s)]
    for i in range(1, len(s)):
        h = 0
        while i + h < len(s) and s[h] == s[i + h]:
            h += 1
        truth.append(h)
    score += 1.0 if z == truth else 0.0
    # Z-box invariant: z[i] never exceeds n-i
    score += 1.0 if all(z[i] <= len(s) - i for i in range(len(s))) else 0.0
    # pattern matching equals str.find scan
    text = "".join(rng.choice(list("abc"), 500))
    pat = "bca"
    occ = z_search(text, pat)
    truth_occ = []
    i = text.find(pat)
    while i != -1:
        truth_occ.append(i)
        i = text.find(pat, i + 1)
    score += 1.0 if occ == truth_occ else 0.0
    # period detection: min period = n - z[i] where i + z[i] == n
    sp = "ababababab"
    zp = z_function(sp)
    period = min(i for i in range(1, len(sp)) if i + zp[i] == len(sp))
    score += 1.0 if period == 2 else 0.0
    return {"synthetic_z_function": score / 4.0}
