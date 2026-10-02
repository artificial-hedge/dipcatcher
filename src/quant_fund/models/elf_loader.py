"""SYNTHETIC ELF-64 parser + loader round-trip.

Builds a minimal ELF64 image in-memory (header + section table +
payloads), parses it back, and verifies section bytes and entry point.
"""

from __future__ import annotations

import random
import struct

_MAGIC = b"\x7fELF"


def build_elf(sections: dict[str, bytes], entry: int = 0x401000) -> bytes:
    names = list(sections)
    ident = _MAGIC + bytes([2, 1, 1, 0]) + b"\x00" * 8
    ehdr = struct.pack(
        "<16sHHIQQQIHHHHHH",
        ident,
        2,
        62,
        1,
        entry,
        0,
        64,
        0,
        64,
        0,
        0,
        64,
        len(names) + 1,
        0,
    )
    off = 64 + 64 * (len(names) + 1)  # payload starts after shdr table
    shdrs = [b"\x00" * 64]
    payloads = b""
    for n in names:
        data = sections[n]
        name_b = n.encode()
        shdrs.append(
            struct.pack(
                "<IIQQQQIIQQ", len(name_b), 1, 0, 0x400000 + off, off, len(data), 0, 0, 1, 0
            )
        )
        payloads += data
        off += len(data)
    return ehdr + b"".join(shdrs) + payloads


def parse_elf(blob: bytes) -> tuple[int, dict[str, bytes]]:
    (ident, _ty, _mach, _ver, entry, _phoff, shoff, _fl, _esz, _ps, _pn, _shentsz, shnum, _str) = (
        struct.unpack("<16sHHIQQQIHHHHHH", blob[:64])
    )
    if ident[:4] != _MAGIC:
        raise ValueError("bad magic")
    out: dict[str, bytes] = {}
    for i in range(1, shnum):
        sh = blob[shoff + i * 64 : shoff + (i + 1) * 64]
        if not sh:
            continue
        (name_len, _typ, _fl2, _va, off, size, _l, _inf, _al, _es) = struct.unpack(
            "<IIQQQQIIQQ", sh
        )
        del name_len
        out[f"sec{i - 1}"] = blob[off : off + size]
    return entry, out


def bench_elf_loader(seed: int = 20261231 + 470) -> dict[str, float]:
    rng = random.Random(seed)
    payload_ok = entry_ok = count_ok = 0
    trials = 30
    for _ in range(trials):
        secs = {
            f"s{i}": bytes(rng.randrange(256) for _ in range(rng.randrange(1, 64)))
            for i in range(rng.randrange(1, 6))
        }
        entry = rng.randrange(0x400000, 0x500000)
        blob = build_elf(secs, entry)
        ent, parsed = parse_elf(blob)
        entry_ok += int(ent == entry)
        payload_ok += int(sorted(parsed.values()) == sorted(secs.values()))
        count_ok += int(len(parsed) == len(secs))
    return {
        "synthetic_payload_roundtrip": float(payload_ok / trials),
        "synthetic_entry_point": float(entry_ok / trials),
        "synthetic_section_count": float(count_ok / trials),
    }
