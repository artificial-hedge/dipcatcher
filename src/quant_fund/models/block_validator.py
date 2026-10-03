"""Block header/tx validation — SYNTHETIC.

Block = (prev_hash, merkle_root, nonce, txids). Verified: valid chain
accepted; bad prev-link / bad merkle / bad PoW rejected.
"""

from __future__ import annotations

import hashlib
import random


def _h(b: bytes) -> bytes:
    return hashlib.sha256(b).digest()


def _merkle(txids: list[str]) -> bytes:
    lvl = [_h(t.encode()) for t in txids]
    while len(lvl) > 1:
        if len(lvl) % 2:
            lvl.append(lvl[-1])
        lvl = [_h(lvl[i] + lvl[i + 1]) for i in range(0, len(lvl), 2)]
    return lvl[0]


class Block:
    def __init__(self, prev: bytes, txids: list[str], bits: int) -> None:
        self.prev = prev
        self.txids = txids
        self.mr = _merkle(txids)
        self.bits = bits
        self.nonce = 0
        self.hash = b""
        self._mine()

    def _raw(self, nonce: int) -> int:
        return int.from_bytes(_h(self.prev + self.mr + nonce.to_bytes(8, "little")), "big")

    def _mine(self) -> None:
        target = 1 << (256 - self.bits)
        n = 0
        while self._raw(n) >= target:
            n += 1
        self.nonce = n
        self.hash = self._raw(n).to_bytes(32, "big")


def validate(b: Block, prev_hash: bytes) -> bool:
    return (
        b.prev == prev_hash
        and b.mr == _merkle(b.txids)
        and int.from_bytes(b.hash, "big") < (1 << (256 - b.bits))
        and b.hash == _h(b.prev + b.mr + b.nonce.to_bytes(8, "little"))
    )


def bench_block_validator(seed: int = 20261231 + 345) -> dict[str, float]:
    rng = random.Random(seed)
    accept = rej_prev = rej_mr = rej_pow = 0
    trials = 15
    for _ in range(trials):
        gen = Block(b"\x00" * 32, ["cb"], 10)
        b1 = Block(gen.hash, ["a", "b"], 10)
        b2 = Block(b1.hash, ["c"], 10)
        accept += int(validate(b1, gen.hash) and validate(b2, b1.hash))
        bad = Block(gen.hash, ["x"], 10)
        bad.prev = b"\x99" * 32
        rej_prev += int(not validate(bad, gen.hash))
        bad2 = Block(gen.hash, ["x"], 10)
        bad2.mr = b"\x00" * 32
        rej_mr += int(not validate(bad2, gen.hash))
        bad3 = Block(gen.hash, ["x"], 10)
        bad3.nonce += 1  # recompute-free forged nonce → hash mismatch
        bad3.hash = rng.randbytes(32)
        rej_pow += int(not validate(bad3, gen.hash))
    return {
        "synthetic_valid_accepted": float(accept / trials),
        "synthetic_bad_prev_rejected": float(rej_prev / trials),
        "synthetic_bad_merkle_rejected": float(rej_mr / trials),
        "synthetic_bad_pow_rejected": float(rej_pow / trials),
    }
