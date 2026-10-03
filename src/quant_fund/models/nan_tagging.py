"""SYNTHETIC tagged pointer representation (tag-in-low-bits).

Values: small-int (tag 00), bool (tag 01), ptr-id (tag 10), none
(singleton). Encode/decode round-trip preserves type + payload;
distinct types never collide.
"""

from __future__ import annotations

import random

TAG_INT, TAG_BOOL, TAG_PTR = 0b00, 0b01, 0b10
NONE_BITS = 0b11


def encode(v: object) -> int:
    if v is None:
        return NONE_BITS
    if isinstance(v, bool):
        return (int(v) << 2) | TAG_BOOL
    if isinstance(v, int):
        return (v << 2) | TAG_INT
    if isinstance(v, tuple) and v[0] == "ptr":
        pv = int(v[1])
        return (pv << 2) | TAG_PTR
    raise ValueError(v)


def decode(bits: int) -> object:
    tag = bits & 0b11
    if tag == TAG_INT:
        return bits >> 2  # arithmetic shift: sign preserved
    if tag == TAG_BOOL:
        return bool(bits >> 2)
    if tag == TAG_PTR:
        return ("ptr", bits >> 2)
    return None


def bench_nan_tagging(seed: int = 20261231 + 485) -> dict[str, float]:
    rng = random.Random(seed)
    rt = dist = bools = 0
    trials = 60
    for _ in range(trials):
        vals: list[object] = [
            rng.randrange(-(1 << 40), 1 << 40),
            rng.choice([True, False]),
            ("ptr", rng.randrange(1 << 50)),
            None,
        ]
        ok = True
        seen = set()
        for v in vals:
            b = encode(v)
            if b in seen and len(set(map(type, vals))) == 1:
                pass
            seen.add(b)
            if decode(b) != v:
                ok = False
        rt += int(ok)
        # distinct types → distinct tags when payloads differ
        e1, e2 = encode(vals[0]), encode(vals[2])
        dist += int((e1 & 3) != (e2 & 3))
        bools += int(
            decode(encode(vals[1])) == vals[1] and isinstance(decode(encode(vals[1])), bool)
        )
    return {
        "synthetic_roundtrip": float(rt / trials),
        "synthetic_tag_discrimination": float(dist / trials),
        "synthetic_bool_exact": float(bools / trials),
    }
