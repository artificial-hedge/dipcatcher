"""Hash-pinned allowlists of byte-exact legacy receipts.

Every committed ``receipts/*.json`` must seal-verify. ``KNOWN_UNSEALED``
previously exempted the seven pre-seal-era artifacts; those migrated to
``receipts/legacy-unsealed/`` (a quarantined subdir — see
``utils.receipt.QUARANTINED_SUBDIRS``), so the map is
empty — it must stay empty: a new unsealed receipt in ``receipts/`` fails.

``KNOWN_CONTRACT_LEGACY`` pins sealed receipts written by drill scripts
before their kind's deep contract existed — they pass the seal check but
fail contract re-derivation on shape alone. Exempt ONLY for the exact
error listed; any other failure still fails.

``KNOWN_TAPE_LEGACY`` pins sealed receipts whose bound tape digest refers
to a tape-manifest entry that is gitignored (the tape bytes are
provider-private — see ``data/manifests/README.md`` for the policy). The
receipt seals cleanly under the SYNTHETIC/CORPUS lane exemption but the
registry's strict lookup fails because the committed manifest directory
doesn't carry the tape. Exempt ONLY for the exact `tape_manifest_unknown`
error — any other failure still fails.

Shrink-only: removing an entry is always safe; adding one is a defect
(commit the new receipt sealed and contract-conformant instead).
Shared by the committed-receipts test ratchet and ``suite-health --strict``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

# filename → sha256 of the committed file bytes. Empty: the seven pre-seal
# artifacts live under `receipts/legacy-unsealed/`, a quarantined subdir.
KNOWN_UNSEALED: dict[str, str] = {}

# Sealed receipts that predate their kind's deep contract (drill artifacts
# written before `changepoint_localize.v1` required a `params` block).
# Exempt ONLY for the exact `missing_params` contract error — any other
# failure still fails. Values pin the file bytes.
KNOWN_CONTRACT_LEGACY: dict[str, str] = {
    "cp_real_drill_gaussian_minus_conf_t_pinball.json": "64a38aba74a56b44aa64dfd68ca1859d4d142c9dc2f4e0a657b447a041e434b7",
    "cp_real_drill_gaussian_pit.json": "6deaa336ee288e28ea35f93caa5d8b8b8436c99fd47efb97ceed792f6d3b8a62",
}

# Sealed receipts that bind a tape digest for which no committed
# `tape_manifest.v1` exists (the tape is provider-private and gitignored;
# the receipt is honest about the lane — `data_label: kraken` / `kraken+okx` —
# but the strict registry lookup is the only thing that fails). Exempt ONLY
# for the exact `tape_manifest_unknown` error. Values pin the file bytes.
KNOWN_TAPE_LEGACY: dict[str, str] = {
    "basis_carry_dd7705fc0f2f1c25.json": "852f56a1e2aeb3654d406fbb14a4a2baa7cd20949d45153a98c5a99a2248f741",
    "crossvenue_basis_3f4ff76f517655a7.json": "14ce799a46e282849e8e7852ff74489ed4747274c164040fcc52974bbb72a087",
}


def _bytes_match(path: Path, expected: str) -> bool:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest() == expected
    except OSError:
        return False


def is_known_unsealed(path: Path, errors: list[str]) -> bool:
    """True iff ``path`` failed verification solely because it is a
    byte-pinned legacy receipt that predates the seal format."""
    expected = KNOWN_UNSEALED.get(path.name)
    if expected is None or errors != ["receipt_sha256_missing_or_invalid"]:
        return False
    return _bytes_match(path, expected)


def is_known_contract_legacy(path: Path, errors: list[str]) -> bool:
    """True iff ``path`` is a byte-pinned sealed receipt whose only failure
    is the `missing_params` contract error of a contract that postdates it.

    The error list may carry duplicates when multiple contract paths flag the
    same gap — the exemption key is the error *set*, not the multiset."""
    expected = KNOWN_CONTRACT_LEGACY.get(path.name)
    if expected is None or set(errors) != {"missing_params"}:
        return False
    return _bytes_match(path, expected)


def is_known_tape_legacy(path: Path, errors: list[str]) -> bool:
    """True iff ``path`` is a byte-pinned sealed receipt whose only failure
    is the `tape_manifest_unknown` registry error (the bound tape is
    provider-private and gitignored)."""
    expected = KNOWN_TAPE_LEGACY.get(path.name)
    if expected is None or set(errors) != {"tape_manifest_unknown"}:
        return False
    return _bytes_match(path, expected)
