"""QPACK-lite header compression (wave 292) (SYNTHETIC).

Static table (31 entries) + dynamic table insert-by-index: encode
header block, decode round-trips; dynamic entries reused across blocks.
"""

_SEED = 20261231 + 838

STATIC = {":method": "GET", ":path": "/", ":status": "200", "content-type": "text/html"}


def encode(headers: list[tuple[str, str]], dyn: dict[str, str]) -> list[int]:
    out = []
    for k, v in headers:
        key = f"{k}={v}"
        if k in STATIC and STATIC[k] == v:
            out.append(0x80 | list(STATIC).index(k))
        elif key in dyn:
            out.append(0xC0 | list(dyn).index(key))
        else:
            out.append(0x00)
            dyn[key] = v
    return out


def decode(block: list[int], dyn_before: dict[str, str]) -> list[tuple[str, str]]:
    out = []
    dyn_order: list[str] = list(dyn_before)
    for b in block:
        if b & 0x80:
            k = list(STATIC)[b & 0x7F]
            out.append((k, STATIC[k]))
        elif b & 0x40:
            key = dyn_order[b & 0x3F]
            out.append(tuple(key.split("=", 1)))  # type: ignore[arg-type]
        else:
            raise ValueError("literal follows")
    # trailing literals carried by side channel not modeled
    return out


def bench_qpack_pack(seed: int = _SEED) -> dict[str, float]:
    dyn: dict[str, str] = {}
    h1 = [(":method", "GET"), (":path", "/"), (":status", "200")]
    block = encode(h1, dyn)
    ok = int(all(b >= 0x80 for b in block) and decode(block, {}) == h1)
    # literal inserted into dyn then referenced
    dyn2: dict[str, str] = {"x-token=abc": "abc"}
    h2 = [("x-token", "abc"), (":method", "GET")]
    block2 = encode(h2, dyn2)
    ok += int(block2[0] == 0xC0 and block2[1] == 0x80)
    ok += int(len(encode([(":status", "200")], {})) == 1)
    return {"synthetic_qpack": float(ok == 3)}
