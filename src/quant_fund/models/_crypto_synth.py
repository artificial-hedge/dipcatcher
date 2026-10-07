"""Synthetic cryptography fixtures shared by the primitives canon (SYNTHETIC).

Small-but-real parameters: a safe prime group for DH/Pedersen, secp256k1
field constants, and known-answer vectors for SHA-256 and the AES S-box.
All arithmetic is integer-only (no external crypto deps beyond hashlib
as the cross-check oracle).
"""

import hashlib

# SHA-256 KAT
SHA_MSG = b"dipcatcher"
SHA_EXPECT = hashlib.sha256(SHA_MSG).hexdigest()

# safe prime p = 2q+1 (q = 419 prime, p = 839 prime); generator of the
# order-q quadratic-residue subgroup
P_SAFE = 839
G_SAFE = 5

# AES S-box known values
AES_SBOX_KNOWN = {0x00: 0x63, 0x01: 0x7C, 0x53: 0xED, 0xFF: 0x16}

# secp256k1: y^2 = x^3 + 7 mod p
SECP_P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
SECP_N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
SECP_GX = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
SECP_GY = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8
