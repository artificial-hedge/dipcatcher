"""SCTP multi-stream delivery (wave 292).

Per-association TSN cumulative ack + per-stream SSN ordering:
out-of-order delivery across streams allowed, in-order within stream;
retransmit of a lost TSN gap resumes delivery.
"""

_SEED = 20261231 + 841


def deliver(chunks: list[tuple[int, int, int, bytes]]) -> dict[int, bytes]:
    """chunks: (tsn, stream_id, ssn, data). Returns stream->reassembled."""
    by_tsn = {t: (s, n, d) for t, s, n, d in chunks}
    out: dict[int, list[tuple[int, bytes]]] = {}
    for tsn in sorted(by_tsn):
        s, n, d = by_tsn[tsn]
        out.setdefault(s, []).append((n, d))
    res = {}
    for s, items in out.items():
        items.sort(key=lambda x: x[0])
        res[s] = b"".join(d for _n, d in items)
    return res


def cum_ack(chunks: list[tuple[int, int, int, bytes]]) -> int:
    tsns = sorted(t for t, _s, _n, _d in chunks)
    ack = tsns[0] - 1
    for t in tsns:
        if t == ack + 1:
            ack = t
        else:
            break
    return ack


def bench_sctp_tsn(seed: int = _SEED) -> dict[str, float]:
    # stream 0 ordered, stream 1 interleaved — both deliver correctly
    chunks = [
        (10, 0, 0, b"A"),
        (11, 1, 0, b"x"),
        (12, 0, 1, b"B"),
        (13, 1, 1, b"y"),
        (14, 0, 2, b"C"),
    ]
    d = deliver(chunks)
    ok = int(d[0] == b"ABC" and d[1] == b"xy")
    ok += int(cum_ack(chunks) == 14)
    # gap at TSN 12: cum ack stops at 11
    ok += int(cum_ack([c for c in chunks if c[0] != 12]) == 11)
    return {"synthetic_sctp": float(ok == 3)}
