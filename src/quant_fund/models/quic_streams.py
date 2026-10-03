"""QUIC stream multiplexing (wave 292).

Per-stream ordered reassembly over interleaved packets carrying
(stream_id, offset, data) frames — delivers bytes in offset order per
stream while tolerating reorder/duplication across streams.
"""

_SEED = 20261231 + 836


def reassemble(stream_id: int, frames: list[tuple[int, int, bytes]]) -> bytes:
    buf: dict[int, bytes] = {}
    for sid, off, data in frames:
        if sid != stream_id:
            continue
        if off not in buf:
            buf[off] = data
    out = b""
    pos = 0
    for off in sorted(buf):
        if off > pos:
            raise ValueError("gap")
        out += buf[off][pos - off :]
        pos = off + len(buf[off])
    return out


def bench_quic_streams(seed: int = _SEED) -> dict[str, float]:
    s1 = [(0, 0, b"GET "), (0, 8, b"1.1")]
    s2 = [(1, 0, b"POST "), (1, 5, b"/up")]
    s1b = [(0, 4, b"HTTP")]
    frames = [s2[1], s1[0], s1b[0], s2[0], s1[1], s2[0]]  # interleaved + dup
    ok = int(reassemble(0, frames) == b"GET HTTP1.1")
    ok += int(reassemble(1, frames) == b"POST /up")
    # gap raises
    try:
        reassemble(0, [(0, 4, b"xx")])
        ok += 0
    except ValueError:
        ok += 1
    return {"synthetic_quic": float(ok == 3)}
