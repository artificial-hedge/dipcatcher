"""Order-flow-imbalance detector (Exec-Summary OFI item). Cont–Kukanov– (SYNTHETIC)
Stoikov event-level OFI from quote updates plus a Kalman-smoothed
short-horizon price-move predictor.

Synthetic bench: book where mid drift is proportional to OFI — sign
accuracy of OFI vs next-tick move and Kalman R² on move prediction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FloatArray = np.ndarray


@dataclass
class BookSim:
    """L1 quote stream; flow pressure lifts mid; depth replenishes."""

    mid0: float = 100.0
    spread: float = 0.02
    depth0: float = 200.0
    ofi_to_price: float = 4e-4
    n: int = 4000

    def stream(
        self, rng: np.random.Generator
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
        bid_p = np.empty(self.n)
        ask_p = np.empty(self.n)
        bid_d = np.empty(self.n)
        ask_d = np.empty(self.n)
        mid = np.empty(self.n)
        mid[0] = self.mid0
        bid_p[0] = self.mid0 - self.spread / 2
        ask_p[0] = self.mid0 + self.spread / 2
        bid_d[0] = ask_d[0] = self.depth0
        flow = 0.0
        for t in range(1, self.n):
            dflow = rng.standard_normal()
            flow = 0.9 * flow + dflow
            delta = self.ofi_to_price * flow * self.depth0 + 1e-4 * rng.standard_normal()
            mid[t] = mid[t - 1] + delta
            bid_p[t] = mid[t] - self.spread / 2
            ask_p[t] = mid[t] + self.spread / 2
            # depth responds to the *sign* of flow (adds on bid when buying)
            bid_d[t] = max(20.0, self.depth0 + 60.0 * flow + 15.0 * rng.standard_normal())
            ask_d[t] = max(20.0, self.depth0 - 60.0 * flow + 15.0 * rng.standard_normal())
        return bid_p, ask_p, bid_d, ask_d, mid


def ofi_events(
    bid_p: FloatArray,
    ask_p: FloatArray,
    bid_d: FloatArray,
    ask_d: FloatArray,
) -> FloatArray:
    """Cont–Kukanov–Stoikov OFI e_n per quote event."""
    n = len(bid_p)
    e = np.zeros(n)
    for t in range(1, n):
        db_p = bid_p[t] - bid_p[t - 1]
        da_p = ask_p[t] - ask_p[t - 1]
        bid_term = (
            bid_d[t] if db_p > 0 else (bid_d[t] - bid_d[t - 1] if db_p == 0 else -bid_d[t - 1])
        )
        ask_term = (
            ask_d[t] if da_p < 0 else (ask_d[t] - ask_d[t - 1] if da_p == 0 else -ask_d[t - 1])
        )
        e[t] = bid_term - ask_term
    return e


@dataclass
class OfiKalman:
    """Scalar Kalman filter on OFI->move gain over standardized flow."""

    q: float = 1e-4
    r: float = 1e-2
    gain0: float = 1e-4
    e_alpha: float = 0.01

    def __post_init__(self) -> None:
        self.gain = self.gain0
        self.p = 1e-3
        self.e_var = 1.0

    def step(self, e_t: float, dmid: float | None = None) -> float:
        """Predicted next move `gain*z`; update on realized move."""
        self.e_var = (1 - self.e_alpha) * self.e_var + self.e_alpha * e_t * e_t
        z = e_t / max(np.sqrt(self.e_var), 1e-9)
        if dmid is not None and abs(z) > 0.3:
            obs = float(np.clip(dmid / z, -5.0 * self.r, 5.0 * self.r))
            self.p += self.q
            k = self.p / (self.p + self.r)
            self.gain += k * (obs - self.gain)
            self.p *= 1.0 - k
        return float(self.gain * z)


def bench_order_flow_imbalance(seed: int = 7) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    sim = BookSim()
    bp, ap, bd, ad, mid = sim.stream(rng)
    e = ofi_events(bp, ap, bd, ad)
    dmid = np.diff(mid)
    e_prev = e[:-1]
    # sign accuracy: does OFI sign predict next move direction?
    sign_hit = np.mean(np.sign(e_prev) == np.sign(dmid))
    # simple regression OFI -> move
    x = (e_prev - e_prev.mean()) / e_prev.std()
    slope = float(np.sum(x * (dmid - dmid.mean())) / np.sum(x * x))
    r2_lin = slope**2 * np.sum(x * x) / np.sum((dmid - dmid.mean()) ** 2)
    # Kalman online predictor
    kf = OfiKalman(gain0=max(slope, 1e-6))
    preds = np.array([kf.step(e_prev[t], dmid[t - 1] if t > 0 else None) for t in range(len(dmid))])
    ss_res = float(np.sum((dmid - preds) ** 2))
    r2_kal = 1.0 - ss_res / np.sum((dmid - dmid.mean()) ** 2)
    imb = (bd - ad) / (bd + ad)
    imb_hit = np.mean(np.sign(imb[1:]) == np.sign(dmid))
    return {
        "synthetic_ofi_sign_accuracy": float(sign_hit),
        "synthetic_ofi_lin_r2": float(r2_lin),
        "synthetic_ofi_kalman_r2": float(r2_kal),
        "synthetic_ofi_imbalance_sign_accuracy": float(imb_hit),
        "synthetic_ofi_events": float(len(e)),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(bench_order_flow_imbalance(), indent=1))
