"""DNS-tunneling detection via label entropy (defensive) — wave 286.

Tunneled lookups carry high-entropy encoded subdomains; lexical Shannon
entropy of the query label separates them from human-typed names.
"""

import numpy as np

_SEED = 20261231 + 801


def label_entropy(name: str) -> float:
    s = name.split(".")[0].lower()
    if not s:
        return 0.0
    _, c = np.unique(np.array(list(s)), return_counts=True)
    p = c / c.sum()
    return float(-np.sum(p * np.log2(p)))


def bench_entropy_dns(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    normal = [
        "mail.example.com",
        "cdn-static.site.org",
        "api.app.io",
        "www.bank.net",
        "images.shop.co",
    ]
    tunneled = [
        "".join(rng.choice(list("abcdefghijklmnopqrstuvwxyz0123456789"), 24)) + ".exfil.net"
        for _ in range(5)
    ]
    h_norm = np.mean([label_entropy(x) for x in normal])
    h_tun = np.mean([label_entropy(x) for x in tunneled])
    margin = min(label_entropy(x) for x in tunneled) - max(label_entropy(x) for x in normal)
    return {"synthetic_dns_entropy": float(h_tun > h_norm * 1.15 and margin > 0)}
