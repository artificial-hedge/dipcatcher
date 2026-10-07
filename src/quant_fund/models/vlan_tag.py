"""802.1Q VLAN tagging: tag/untag + trunk-domain separation (SYNTHETIC)."""

import numpy as np

_SEED = 20261231 + 712


class Trunk:
    def __init__(self) -> None:
        self.domains: dict[int, list[int]] = {}

    def send(self, frame: int, vlan: int, port: int) -> None:
        self.domains.setdefault(vlan, []).append(frame)
        _ = port  # trunk ports share the domain


def bench_vlan_tag(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    ok = 0.0
    trials = 40
    for _ in range(trials):
        trunk = Trunk()
        n_frames = int(rng.randint(4, 20))
        vlans = rng.randint(1, 4, n_frames)
        frames = rng.randint(0, 1 << 20, n_frames)
        for f, v, p in zip(frames, vlans, rng.randint(0, 4, n_frames), strict=True):
            trunk.send(int(f), int(v), int(p))
        # isolation: each frame lands only in its own vlan
        iso = all(
            f in trunk.domains[v]
            and all(f not in trunk.domains.get(w, []) for w in trunk.domains if w != v)
            for f, v in zip(frames, vlans, strict=True)
        )
        ok += float(iso)
    return {"synthetic_vlan_isolation": ok / trials}
