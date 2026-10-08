"""Shared synthetic fixtures for the wave-210 coding-theory canon (SYNTHETIC)."""

import numpy as np

# LDPC: systematic code, H = [P | I_6], codeword [m(8) | p(6)], n=14
LDPC_P = np.array(
    [
        [1, 0, 1, 0, 1, 0, 0, 1],
        [0, 1, 0, 1, 0, 1, 1, 0],
        [1, 0, 0, 1, 0, 1, 1, 0],
        [0, 1, 1, 0, 1, 0, 0, 1],
        [0, 0, 1, 0, 1, 0, 1, 1],
        [1, 1, 0, 1, 0, 1, 0, 0],
    ]
)
LDPC_K = 8
LDPC_N = 14

# turbo: RSC (1,5/7) component codes, message length
TURBO_K = 24

# polar: N=16, k=8 over BSC design channel
POLAR_N = 16
POLAR_K = 8
POLAR_DESIGN = 0.11

# BCH(15,7,t=2) over GF(16), poly x^4+x+1
BCH_POLY = 0b10011
BCH_N = 15
BCH_K = 7
BCH_G = 0b111010001  # x^8+x^7+x^6+x^4+1

CRC_KAT_MSG = b"123456789"
CRC32_TRUE = 0xCBF43926
CRC16_TRUE = 0x29B1

AWGN_SIGMA = 0.8


def msg_bits(seed: int, k: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 2, k)


def bpsk_awgn(bits: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (1.0 - 2.0 * bits.astype(float)) + rng.normal(0.0, sigma, len(bits))


def bsc(bits: np.ndarray, p: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.asarray(bits ^ (rng.uniform(size=len(bits)) < p).astype(int), dtype=int)


def burst_channel(bits: np.ndarray, start: int, length: int) -> np.ndarray:
    out = bits.copy()
    out[start : start + length] ^= 1
    return out
