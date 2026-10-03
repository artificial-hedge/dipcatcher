"""UTXO set validation — SYNTHETIC.

Tx: (inputs: list[(txid,vout)], outputs: list[(addr,amt)]). Verified:
value conservation (in = out + fee), double-spend rejection, orphan
rejection.
"""

from __future__ import annotations


class UTXO:
    def __init__(self) -> None:
        self.set: dict[tuple[str, int], tuple[str, int]] = {}
        self.txid = 0

    def coinbase(self, addr: str, amt: int) -> str:
        txid = f"cb{self.txid}"
        self.txid += 1
        self.set[(txid, 0)] = (addr, amt)
        return txid

    def spend(self, ins: list[tuple[str, int]], outs: list[tuple[str, int]]) -> str | None:
        total_in = 0
        refs = []
        for r in ins:
            if r not in self.set:
                return None  # orphan / double-spend
            refs.append(self.set[r])
            total_in += self.set[r][1]
        total_out = sum(a for _a, a in outs)
        if total_out > total_in:
            return None
        for r in ins:
            del self.set[r]
        txid = f"tx{self.txid}"
        self.txid += 1
        for i, (a, amt) in enumerate(outs):
            self.set[(txid, i)] = (a, amt)
        return txid


def bench_utxo_set(seed: int = 20261231 + 342) -> dict[str, float]:
    ok_cons = ok_ds = ok_orph = 0
    trials = 30
    for _ in range(trials):
        u = UTXO()
        t0 = u.coinbase("alice", 50)
        t1 = u.spend([(t0, 0)], [("bob", 30), ("alice", 15)])  # 5 fee
        ok_cons += int(t1 is not None and u.set[(t1, 0)][1] == 30)
        # double spend of consumed outpoint
        ok_ds += int(u.spend([(t0, 0)], [("eve", 50)]) is None)
        # orphan input
        ok_orph += int(u.spend([("nonexistent", 0)], [("eve", 1)]) is None)
    return {
        "synthetic_valid_spend": float(ok_cons / trials),
        "synthetic_double_spend_rejected": float(ok_ds / trials),
        "synthetic_orphan_rejected": float(ok_orph / trials),
    }
