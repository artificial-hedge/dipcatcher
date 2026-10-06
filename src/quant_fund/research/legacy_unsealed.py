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

# Sealed receipts that predate the contract they now fail: drill artifacts
# written before `changepoint_localize.v1` required a `params` block, tape
# receipts sealed before the tape-manifest registry covered their dataset,
# and probe receipts committed before the honesty envelope required every
# stamp. Each entry pins the file bytes AND the exact error set the
# exemption admits — any other failure or a drifted byte still fails.
KNOWN_CONTRACT_LEGACY: dict[str, tuple[str, frozenset[str]]] = {
    "cp_real_drill_gaussian_minus_conf_t_pinball.json": (
        "64a38aba74a56b44aa64dfd68ca1859d4d142c9dc2f4e0a657b447a041e434b7",
        frozenset({"missing_params"}),
    ),
    "cp_real_drill_gaussian_pit.json": (
        "6deaa336ee288e28ea35f93caa5d8b8b8436c99fd47efb97ceed792f6d3b8a62",
        frozenset({"missing_params"}),
    ),
    # Pre-stamp receipts: committed before the honesty envelope required
    # `research_only`, `git_revision`, and the provenance-label enum, and
    # before the tape-manifest registry covered kraken/okx bindings. The
    # sealed bytes are sound; only the ratcheted contract postdates them.
    "mid_dark_amzn.json": (
        "da7afa773a1280ac1bff785c8fa1fbf9b10e2512ac39d3120f528ccc735207e1",
        frozenset({"research_only_not_true"}),
    ),
    "basis_carry_dd7705fc0f2f1c25.json": (
        "852f56a1e2aeb3654d406fbb14a4a2baa7cd20949d45153a98c5a99a2248f741",
        frozenset({"tape_manifest_unknown"}),
    ),
    "crossvenue_basis_3f4ff76f517655a7.json": (
        "14ce799a46e282849e8e7852ff74489ed4747274c164040fcc52974bbb72a087",
        frozenset({"tape_manifest_unknown"}),
    ),
    "lobster_replay_amzn_2012-06-21.json": (
        "44f53ca3f6e42bad9617d284fc37542ca64d3b4373642fe54deae83302a9dd10",
        frozenset({"data_label_bad:LOBSTER-AMZN-2012-06-21-sample"}),
    ),
    "queue_priority.json": (
        "c03de3a987fe19ee42eadf2f5ac6e97e61da5d55bf89f3a81493e0abf399e7d4",
        frozenset({"git_revision_missing"}),
    ),
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
    """True iff ``path`` is a byte-pinned sealed receipt whose only failures
    are the exact error set pinned by ``KNOWN_CONTRACT_LEGACY`` — a contract
    that postdates the committed bytes.

    The error list may carry duplicates when multiple contract paths flag the
    same gap — the exemption key is the error *set*, not the multiset."""
    entry = KNOWN_CONTRACT_LEGACY.get(path.name)
    if entry is None or set(errors) != entry[1]:
        return False
    return _bytes_match(path, entry[0])
