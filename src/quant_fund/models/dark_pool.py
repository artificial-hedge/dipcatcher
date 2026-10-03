"""Dark-pool activity analyzer (Exec-Summary Feature 5). Consolidated-
tape simulation with hidden prints: episodes of heavy volume at a static
price (possible hidden offsetting), a rolling dark-imbalance index
(DIX-style), and an accumulation/distribution phase classifier.

Synthetic bench: injected hidden accumulation episodes — detection
precision/recall, phase-classification accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FloatArray = np.ndarray


@dataclass
class TapeSim:
    """Prints (price, size); hidden-buy episodes pin price near level."""

    p0: float = 100.0
    n: int = 3000
    vol: float = 0.02
    episode_len: int = 60

    def stream(
        self, rng: np.random.Generator
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        px = np.empty(self.n)
        sz = np.empty(self.n)
        aggr = np.empty(self.n)  # +1 buyer-initiated, -1 seller-initiated
        hidden = np.zeros(self.n)  # 1 = inside hidden-accumulation episode
        px[0] = self.p0
        aggr[0] = 1.0
        t = 1
        while t < self.n:
            if rng.random() < 0.004 and t + self.episode_len < self.n:
                # hidden accumulation: size spikes, price pinned, buy-initiated
                pin = px[t - 1]
                for k in range(self.episode_len):
                    px[t] = pin + 2e-3 * rng.standard_normal() + 5e-4 * k
                    sz[t] = float(rng.uniform(800, 3000))
                    aggr[t] = 1.0 if rng.random() < 0.9 else -1.0
                    hidden[t] = 1.0
                    t += 1
            else:
                px[t] = px[t - 1] * np.exp(self.vol * rng.standard_normal() / np.sqrt(200))
                sz[t] = float(np.exp(rng.normal(4.0, 1.0)))
                aggr[t] = 1.0 if rng.random() < 0.5 else -1.0
                t += 1
        return px, sz, aggr, hidden


def dix_index(
    px: FloatArray,
    sz: FloatArray,
    aggr: FloatArray,
    window: int = 50,
) -> FloatArray:
    """Rolling dark-imbalance: signed volume share of pinned-price prints.

    A print is 'pinned' when |dp| is in the smallest quartile of recent
    moves; imbalance = signed(aggressor) volume share over the window.
    """
    n = len(px)
    d = np.abs(np.diff(px, prepend=px[0]))
    out = np.zeros(n)
    for t in range(window, n):
        dw = d[t - window : t]
        sw = sz[t - window : t]
        aw = aggr[t - window : t]
        thresh = np.quantile(dw, 0.4) + 1e-12
        pinned = dw <= thresh
        tot = sw[pinned].sum() + 1e-9
        out[t] = float((sw[pinned] * aw[pinned]).sum() / tot)
    return out


def detect_hidden(px: FloatArray, sz: FloatArray, window: int = 40) -> FloatArray:
    """Flag windows with heavy volume + abnormally static price."""
    n = len(px)
    flag = np.zeros(n)
    rv = np.abs(np.diff(px, prepend=px[0]))
    for t in range(window, n):
        z_sz = (sz[t - window : t].mean() - sz.mean()) / (sz.std() + 1e-9)
        static = np.quantile(rv[t - window : t], 0.9) < np.quantile(rv, 0.4)
        flag[t] = float(z_sz > 0.8 and static)
    return flag


def phase_of(dix: FloatArray) -> FloatArray:
    """Accumulation (+1) when DIX rising & positive; distribution (-1) falling."""
    slope = np.diff(dix, prepend=dix[0])
    out = np.zeros(len(dix))
    out[(dix > 0) & (slope > -0.002)] = 1.0
    out[(dix < 0) & (slope < 0.002)] = -1.0
    return out


def bench_dark_pool(seed: int = 7) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    sim = TapeSim()
    px, sz, aggr, hidden = sim.stream(rng)
    flag = detect_hidden(px, sz)
    mid = slice(50, len(px) - 20)
    tp = float(np.sum(flag[mid] * hidden[mid]))
    prec = tp / max(float(flag[mid].sum()), 1.0)
    rec = tp / max(float(hidden[mid].sum()), 1.0)
    dix = dix_index(px, sz, aggr)
    phase = phase_of(dix)
    # phase accuracy on hidden-episode windows: should read accumulation
    in_epi = hidden[mid] == 1
    acc = float(np.mean(phase[mid][in_epi] == 1.0)) if in_epi.any() else 0.0
    dix_epi = float(dix[mid][in_epi].mean()) if in_epi.any() else 0.0
    dix_off = float(dix[mid][hidden[mid] == 0].mean())
    return {
        "synthetic_dark_pool_detect_precision": prec,
        "synthetic_dark_pool_detect_recall": rec,
        "synthetic_dark_pool_phase_accuracy": acc,
        "synthetic_dark_pool_dix_in_episode": dix_epi,
        "synthetic_dark_pool_dix_off_episode": dix_off,
        "synthetic_dark_pool_dix_separation": dix_epi - dix_off,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(bench_dark_pool(), indent=1))
