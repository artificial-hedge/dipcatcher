"""FitCache audit — key semantics under layout, mutation, and eviction.

A content-hash cache is only as honest as its key: two logically-equal
arrays in different memory layouts should either collide (fast hit) or
honestly miss — but a mutated array must NEVER collide with its prior
key, and ``get`` must return exactly what ``put`` stored.

Probes:

- ``layout``: ``ascontiguousarray`` normalizes F-order to the same key
  (required hit); transposes and slices are different content (required
  miss).
- ``mutation``: flip one bit/element; the key must change. Non-negoti­able.
- ``roundtrip``: put/get returns the stored object.
- ``bound``: over-insert; len(cache) never exceeds ``max_entries`` and
  eviction is LRU-ordered.
- ``empty``/``scalar`` edge inputs must hash, not crash.

Sealed ``cache_audit.v1``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.compute.cache import FitCache
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision


def cache_audit() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def rec(label: str, ok: bool, detail: str = "") -> None:
        checks.append({"check": label, "ok": ok, "detail": detail})

    cache = FitCache(max_entries=8)
    base = np.arange(64.0).reshape(8, 8)

    # --- layout probes -------------------------------------------------
    key_c = cache.hash_key(("m",), (base,))
    key_f = cache.hash_key(("m",), (np.asfortranarray(base),))
    key_strided = cache.hash_key(("m",), (base[::2],))
    key_T = cache.hash_key(("m",), (base.T,))
    same_c = cache.hash_key(("m",), (base.copy(),))
    rec("layout:c_copy_equal", key_c == same_c, "identical C-order copy")
    rec(
        "layout:forder_hit",
        key_f == key_c,
        "ascontiguousarray normalizes F-order to the same key",
    )
    rec(
        "layout:transpose_distinct",
        key_T != key_c,
        "transposed logical content must differ",
    )
    rec(
        "layout:strided_slice_distinct",
        key_strided != key_c,
        "sliced view is different content",
    )

    # --- mutation probes -----------------------------------------------
    mutated = base.copy()
    mutated[3, 3] += 1e-9
    rec("mutation:value_flip", cache.hash_key(("m",), (mutated,)) != key_c)
    mut_i = base.copy().view(np.int64)
    mut_i[0, 0] ^= 1
    rec(
        "mutation:bit_flip_same_dtype",
        cache.hash_key(("m",), (mut_i.view(np.float64),)) != key_c,
    )
    rec(
        "mutation:tag_change",
        cache.hash_key(("m2",), (base,)) != key_c,
    )

    # --- roundtrip / bound ---------------------------------------------
    cache.put(key_c, {"weights": [1, 2, 3]})
    rec("roundtrip", cache.get(key_c) == {"weights": [1, 2, 3]})
    rec("miss_absent", cache.get("0" * 64) is None)

    big = FitCache(max_entries=4)
    for i in range(12):
        big.put(f"k{i}", i)
    rec("bound:len", len(big._store) <= 4, f"len={len(big._store)}")  # noqa: SLF001
    rec(
        "bound:lru_order",
        big.get("k0") is None and big.get("k11") == 11,
        "oldest evicted, newest kept",
    )

    # --- edge inputs ----------------------------------------------------
    try:
        cache.hash_key(("e",), (np.array([]), np.array(5.0)))
        rec("edge:empty_and_scalar", True)
    except Exception as exc:  # noqa: BLE001
        rec("edge:empty_and_scalar", False, str(exc)[:120])

    n_bad = sum(1 for c in checks if not c["ok"])
    return {
        "n_checks": len(checks),
        "n_violations": n_bad,
        "verdict": "ok" if n_bad == 0 else "violations",
        "checks": checks,
    }


def cache_audit_bench() -> dict[str, Any]:
    report = cache_audit()
    payload: dict[str, Any] = {
        "kind": "cache_audit",
        "schema": "cache_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {
            "invariant": "mutated inputs never collide; bound is respected; get returns what put stored",
            "verdict": report["verdict"],
            "n_checks": report["n_checks"],
            "n_violations": report["n_violations"],
        },
        "interpretation": report,
    }
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
