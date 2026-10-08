"""DoH/DNS wire format (wave 292) (SYNTHETIC).

RFC1035 name encoding (length-prefixed labels), header + question +
answer encode/decode round-trip; pointers not used in question names.
"""

import struct

_SEED = 20261231 + 840


def encode_name(name: str) -> bytes:
    return b"".join(bytes([len(p)]) + p.encode() for p in name.split(".")) + b"\x00"


def decode_name(buf: bytes, off: int) -> tuple[str, int]:
    parts = []
    while buf[off]:
        ln = buf[off]
        off += 1
        parts.append(buf[off : off + ln].decode())
        off += ln
    return ".".join(parts), off + 1


def encode_query(qid: int, name: str, qtype: int = 1) -> bytes:
    return (
        struct.pack(">HHHHHH", qid, 0x0100, 1, 0, 0, 0)
        + encode_name(name)
        + struct.pack(">HH", qtype, 1)
    )


def decode_query(buf: bytes) -> tuple[int, str, int]:
    qid, _flags, qd, _an, _ns, _ar = struct.unpack(">HHHHHH", buf[:12])
    name, off = decode_name(buf, 12)
    qtype, _qclass = struct.unpack(">HH", buf[off : off + 4])
    return qid, name, qtype


def bench_doh_wire(seed: int = _SEED) -> dict[str, float]:
    ok = int(encode_name("www.example.com") == b"\x03www\x07example\x03com\x00")
    wire = encode_query(0x1234, "www.example.com")
    qid, name, qtype = decode_query(wire)
    ok += int(qid == 0x1234 and name == "www.example.com" and qtype == 1)
    ok += int(decode_name(encode_name("a.b.c"), 0)[0] == "a.b.c")
    return {"synthetic_doh": float(ok == 3)}
