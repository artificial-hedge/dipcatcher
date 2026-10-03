"""CRC-32 (reflected Ethernet poly) and CRC-16-CCITT codecs.

Bitwise table-free implementations; verified against the standard
check vectors ("123456789" -> 0xCBF43926 / 0x29B1) and then exercised
on the synthetic channel: single-bit and burst error detection rates.
"""

import numpy as np

from quant_fund.models._code_synth import CRC16_TRUE, CRC32_TRUE, CRC_KAT_MSG

_P32 = 0xEDB88320
_P16 = 0x1021


def _crc32(data: bytes) -> int:
    crc = 0xFFFFFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ (_P32 if crc & 1 else 0)
    return crc ^ 0xFFFFFFFF


def _crc16(data: bytes) -> int:
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ _P16) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def bench_crc_check(seed: int = 5009) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    det1 = det_burst = miss = 0
    trials = 300
    base = bytes(rng.integers(0, 256, 48).tolist())
    crc_b = _crc32(base)
    for _ in range(trials):
        buf = bytearray(base)
        buf[rng.integers(0, 48)] ^= 1 << rng.integers(0, 8)
        det1 += int(_crc32(bytes(buf)) != crc_b)
    for _ in range(trials):
        buf = bytearray(base)
        start = int(rng.integers(0, 40))
        for j in range(32):  # 32-bit burst inside one dword window
            pos = start * 8 + j
            if pos < 48 * 8:
                buf[pos // 8] ^= 1 << (pos % 8)
        det_burst += int(_crc32(bytes(buf)) != crc_b)
        miss += int(_crc32(bytes(buf)) == crc_b)
    return {
        "synthetic_crc32_kat": float(_crc32(CRC_KAT_MSG) == CRC32_TRUE),
        "synthetic_crc16_kat": float(_crc16(CRC_KAT_MSG) == CRC16_TRUE),
        "synthetic_crc32_single_detect": det1 / trials,
        "synthetic_crc32_burst_detect": det_burst / trials,
        "synthetic_crc32_undetected": miss / trials,
    }
