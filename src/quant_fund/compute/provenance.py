"""provenance — replay manifests for parallel research sweeps.

``parallel.process_map`` gives every task a deterministic derived seed,
but once the sweep finishes the only evidence is the result list. This
wrapper additionally emits a manifest recording, per task:

- the derived seed that task actually ran under,
- the task index (the seed's only input besides ``base_seed``),
- the sha256 of the canonical JSON serialization of its result,
- an optional caller-supplied item digest.

``replay_task`` re-executes a single task by index and re-hashes — a
manifest that can't replay is reported, not silently trusted. The whole
manifest is sealable as a ``sweep_provenance.v1`` receipt.

Results must be JSON-serializable by the repo's canonical encoder — a
result that cannot be digested cannot be pinned.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from quant_fund.compute.parallel import derive_seed, process_map
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision


def _assert_jsonable(value: Any, path: str = "$.") -> None:
    """Recursive JSON-primitive check — canonical_json_bytes falls back to
    repr() for foreign objects, which embeds memory addresses and produces
    non-deterministic digests; reject instead of pinning garbage."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return
    if isinstance(value, dict):
        for k, v in value.items():
            if not isinstance(k, str):
                raise ValueError(f"{path}: dict keys must be strings, got {type(k).__name__}")
            _assert_jsonable(v, f"{path}{k}.")
        return
    if isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            _assert_jsonable(v, f"{path}[{i}].")
        return
    raise ValueError(f"{path}: result contains non-JSON type {type(value).__name__}")


def _digest_result(result: Any) -> str:
    _assert_jsonable(result)
    try:
        return hash_bytes(canonical_json_bytes(result))
    except (TypeError, ValueError) as exc:
        raise ValueError("sweep results must be canonical-JSON serializable to be pinned") from exc


def provenance_map[T, R](
    fn: Callable[[T, int], R],
    items: Sequence[T],
    *,
    base_seed: int,
    sweep: str,
    min_items: int = 4,
    max_workers: int | None = None,
) -> tuple[list[R], dict[str, Any]]:
    """Run ``fn`` via ``process_map`` and return ``(results, manifest)``.

    The manifest binds ``sweep`` name, ``base_seed``, git revision, and a
    per-task ``{index, seed, result_sha256}`` triple — sufficient to
    replay any single task or verify the whole sweep's outputs.
    """
    if not sweep or not isinstance(sweep, str):
        raise ValueError("sweep must be a non-empty name")
    sequenced = list(items)
    results = process_map(
        fn, sequenced, base_seed=base_seed, min_items=min_items, max_workers=max_workers
    )
    tasks = [
        {
            "index": i,
            "seed": derive_seed(base_seed, i),
            "result_sha256": _digest_result(results[i]),
        }
        for i in range(len(sequenced))
    ]
    manifest: dict[str, Any] = {
        "kind": "sweep_provenance",
        "schema": "sweep_provenance.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "claim": {
            "sweep": sweep,
            "base_seed": int(base_seed),
            "n_tasks": len(sequenced),
            "tasks": tasks,
        },
        "interpretation": {
            "replayable": "each task's output is pinned to (base_seed, index) and its own digest",
        },
    }
    manifest["receipt_sha256"] = hash_bytes(canonical_json_bytes(manifest))
    return results, manifest


def replay_task[T, R](
    fn: Callable[[T, int], R],
    items: Sequence[T],
    manifest: dict[str, Any],
    index: int,
) -> dict[str, Any]:
    """Re-run one task from a manifest and compare the digest.

    Fails closed: unknown index, malformed manifest, or a digest
    mismatch are reported as ``replayable: False`` with a reason.
    """
    claim = manifest.get("claim", {})
    tasks = claim.get("tasks")
    base_seed = claim.get("base_seed")
    if not isinstance(tasks, list) or not isinstance(base_seed, int):
        raise ValueError("manifest has no tasks/base_seed claim")
    sequenced = list(items)
    if index < 0 or index >= len(tasks) or index >= len(sequenced):
        raise ValueError(f"index {index} outside sweep")
    entry = tasks[index]
    seed = entry.get("seed")
    if seed != derive_seed(base_seed, index):
        return {
            "replayable": False,
            "reason": "manifest seed disagrees with derive_seed(base_seed, index)",
            "index": index,
        }
    result = fn(sequenced[index], seed)
    digest = _digest_result(result)
    ok = digest == entry.get("result_sha256")
    return {
        "replayable": bool(ok),
        "index": index,
        "seed": seed,
        "result_sha256": digest,
        "manifest_sha256": entry.get("result_sha256"),
        "reason": "match" if ok else "result digest mismatch",
    }


def verify_sweep[T, R](
    fn: Callable[[T, int], R],
    items: Sequence[T],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Replay every task and report the mismatches (usually small sweeps)."""
    claim = manifest.get("claim", {})
    tasks = claim.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError("manifest has no tasks claim")
    bad = []
    for i in range(len(tasks)):
        rep = replay_task(fn, items, manifest, i)
        if not rep["replayable"]:
            bad.append({"index": i, "reason": rep["reason"]})
    return {
        "ok": not bad,
        "n_tasks": len(tasks),
        "n_failed": len(bad),
        "failures": bad,
    }


__all__ = ["provenance_map", "replay_task", "verify_sweep"]
