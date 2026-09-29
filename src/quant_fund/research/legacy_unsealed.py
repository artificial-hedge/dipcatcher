"""Hash-pinned allowlist of pre-seal-era receipts.

Every committed ``receipts/*.json`` must seal-verify — except the seven
legacy artifacts listed here, which predate the seal format and are pinned
by file bytes so the exemption can't mask edits to their claims.

Shrink-only: removing an entry is always safe; adding one is a defect
(commit the new receipt sealed instead, or extend the sealed format).
Shared by the committed-receipts test ratchet and ``suite-health --strict``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

# filename → sha256 of the committed file bytes.
KNOWN_UNSEALED: dict[str, str] = {
    "adaptive_mix_20asset_1d_20260922.json": "b59422415e556f674ea03fb8e9a3b867408bf6d163cf7c11c64feeb7e33b4313",
    "adaptive_mix_band_search_20asset_1d_20260922.json": "5e02918850944a68d1d45815b5928f1eed462d456eac4723ac9290514e7b97f4",
    "basis_pair_candidate_20asset_1d_20260922.json": "d021801c94e33d8720f9084762138893978a61e36ef670b4c0140edeaf37c1ec",
    "basis_reversion_screen_20asset_1d_20260922.json": "61c9c8f5b3e502f0c5218dd743ad030ecbb962d6822bb7b81a2b962445aa378f",
    "dip_bench_crypto_1d_20260925.json": "584eb681dcd18fc65835c1573360b4b5fc9ebb72aa662bcbb9e7c101b06e2f03",
    "fast_replay_p42_conformance_20260927.json": "8236a26489e9253dfcc1f3d879a2fd276c0023fb4d167c764c5330406ce950d8",
    "incumbent_bench_qlib.json": "f455123351b44151d68876d1a93fa3a2dc449c51d280ee83926f22cd9861984f",
}


def is_known_unsealed(path: Path, errors: list[str]) -> bool:
    """True iff ``path`` failed verification solely because it is a
    byte-pinned legacy receipt that predates the seal format."""
    expected = KNOWN_UNSEALED.get(path.name)
    if expected is None or errors != ["receipt_sha256_missing_or_invalid"]:
        return False
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest() == expected
    except OSError:
        return False
