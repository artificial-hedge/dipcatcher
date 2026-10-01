"""Receipt lineage DAG over the committed evidence corpus.

``receipt_lattice`` checks *horizontal* consistency — two receipts
making claims over the same shard must not contradict. This module
maps the *vertical* direction: which receipts consume which inputs,
and which inputs no receipt produced (unprovenanced evidence).

Each receipt contributes a node keyed by ``receipt_sha256`` and a set
of input references harvested from its payload: sha256 digests
(``inputs_sha256``, ``dataset_hash``, ``dataset_sha256``,
``code_sha256``, ``fingerprint_sha256``), committed file paths,
``git_revision`` markers, and ``drill.shard.*`` source descriptors.
A digest that matches another receipt's seal is a receipt→receipt
edge; anything else is a receipt→artifact edge. The graph report
names roots (artifacts nothing produced — the trust boundary),
sinks, orphan receipts, dangling inputs, and any cycle (cycles are
impossible under honest ordering — finding one is an anomaly).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_DIGEST_KEYS = (
    "inputs_sha256",
    "dataset_sha256",
    "dataset_hash",
    "code_sha256",
    "fingerprint_sha256",
    "payload_sha256",
    "manifest_sha256",
    "receipt_sha256",
)
_FILE_SUFFIXES = (".json", ".jsonl", ".parquet", ".csv", ".yaml", ".yml", ".md")


def _walk_refs(obj: Any, prefix: str, refs: list[dict[str, str]]) -> None:
    """Harvest input references from a receipt payload."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            path = f"{prefix}{k}"
            if isinstance(v, str) and _SHA256.match(v) and k in _DIGEST_KEYS:
                refs.append({"kind": "digest", "via": path, "value": v})
            elif isinstance(v, str) and v.endswith(_FILE_SUFFIXES) and "/" in v and len(v) < 300:
                kind = "url" if "://" in v else "file"
                refs.append({"kind": kind, "via": path, "value": v})
            else:
                _walk_refs(v, path + ".", refs)
    elif isinstance(obj, list):
        for v in obj:
            if isinstance(v, str) and v.endswith(_FILE_SUFFIXES) and "/" in v and len(v) < 300:
                kind = "url" if "://" in v else "file"
                refs.append({"kind": kind, "via": prefix.rstrip("."), "value": v})
            elif isinstance(v, str) and _SHA256.match(v):
                refs.append({"kind": "digest", "via": prefix.rstrip("."), "value": v})
            else:
                _walk_refs(v, prefix, refs)


def collect_receipt_refs(path: Path) -> dict[str, Any]:
    """One receipt's digest + its declared input references."""
    try:
        doc = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {"file": str(path), "digest": None, "kind": None, "refs": []}
    refs: list[dict[str, str]] = []
    _walk_refs(doc, "", refs)
    # the receipt's own seal is not an input
    own = doc.get("receipt_sha256")
    refs = [r for r in refs if not (r["kind"] == "digest" and r["value"] == own)]
    if isinstance(doc.get("git_revision"), str):
        refs.append({"kind": "git_revision", "via": "git_revision", "value": doc["git_revision"]})
    shard = doc.get("drill", {}).get("shard") if isinstance(doc.get("drill"), dict) else None
    if isinstance(shard, dict) and shard.get("source"):
        refs.append(
            {
                "kind": "shard",
                "via": "drill.shard",
                "value": f"{shard.get('source')}:{shard.get('symbol')}",
            }
        )
    return {
        "file": str(path),
        "digest": own if isinstance(own, str) else None,
        "kind": doc.get("kind"),
        "data_label": doc.get("data_label"),
        "refs": refs,
    }


def lineage_dag(root: Path | str) -> dict[str, Any]:
    """Build the receipt→input DAG over a corpus directory."""
    root = Path(root)
    receipts = [collect_receipt_refs(p) for p in sorted(root.glob("*.json")) if p.is_file()]
    known_digests = {r["digest"] for r in receipts if r["digest"]}

    edges: list[dict[str, str]] = []
    artifacts: set[str] = set()
    dangling: list[str] = []
    consumers: set[str] = set()
    consumed: set[str] = set()
    for r in receipts:
        src = r["digest"] or r["file"]
        for ref in r["refs"]:
            if ref["kind"] == "digest" and ref["value"] in known_digests:
                edges.append(
                    {"from": src, "to": ref["value"], "via": ref["via"], "type": "receipt"}
                )
                consumers.add(src)
                consumed.add(ref["value"])
            else:
                key = f"{ref['kind']}:{ref['value']}"
                artifacts.add(key)
                edges.append({"from": src, "to": key, "via": ref["via"], "type": "artifact"})
                consumers.add(src)
                # dangling = file ref that does not resolve on disk
                if ref["kind"] == "file" and not (
                    Path(ref["value"]).exists() or (root / ref["value"]).exists()
                ):
                    dangling.append(f"{r['file']}:{ref['via']}={ref['value']}")

    artifact_nodes = artifacts - consumed
    adj: dict[str, list[str]] = {}
    for e in edges:
        adj.setdefault(e["from"], []).append(e["to"])

    # cycle check (DFS, receipts only — artifacts are leaves)
    color: dict[str, int] = {}
    cycles: list[str] = []

    def dfs(node: str, stack: list[str]) -> None:
        color[node] = 1
        for nxt in adj.get(node, []):
            if nxt in color and color[nxt] == 1:
                cycles.append(" -> ".join(stack + [nxt]))
            elif color.get(nxt, 0) == 0:
                dfs(nxt, stack + [nxt])
        color[node] = 2

    for r in receipts:
        node = r["digest"] or r["file"]
        if color.get(node, 0) == 0:
            dfs(node, [node])

    all_receipt_nodes = {r["digest"] or r["file"] for r in receipts}
    orphans = sorted(all_receipt_nodes - consumers - consumed)
    sinks = sorted(consumers - consumed)
    roots = sorted(a for a in artifact_nodes if a.startswith(("file:", "digest:")))
    external = sorted(a for a in artifact_nodes if not a.startswith(("file:", "digest:")))
    return {
        "n_receipts": len(receipts),
        "n_edges": len(edges),
        "receipt_edges": sum(1 for e in edges if e["type"] == "receipt"),
        "artifact_edges": sum(1 for e in edges if e["type"] == "artifact"),
        "n_artifact_nodes": len(artifact_nodes),
        "roots": roots,
        "external_inputs": external,
        "sinks": sinks,
        "orphan_receipts": orphans,
        "dangling_inputs": dangling,
        "cycles": cycles,
        "verdict": "provenance_complete" if not cycles and not dangling else "open",
    }


def lineage_dag_bench(root: Path | str = "receipts") -> dict[str, Any]:
    """Seal the corpus lineage report (capped lists keep the receipt small)."""
    dag = lineage_dag(root)
    interpretation = {
        "n_receipts": dag["n_receipts"],
        "n_edges": dag["n_edges"],
        "receipt_edges": dag["receipt_edges"],
        "artifact_edges": dag["artifact_edges"],
        "n_roots": len(dag["roots"]),
        "n_external_inputs": len(dag["external_inputs"]),
        "n_sinks": len(dag["sinks"]),
        "n_orphans": len(dag["orphan_receipts"]),
        "n_dangling": len(dag["dangling_inputs"]),
        "cycles": dag["cycles"][:8],
        "verdict": dag["verdict"],
    }
    payload: dict[str, Any] = {
        "kind": "lineage_dag",
        "schema": "lineage_dag.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC" if dag["n_receipts"] else "UNKNOWN",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {
            "invariant": "the evidence corpus is acyclic and its inputs are named",
            "verdict": dag["verdict"],
        },
        "interpretation": interpretation,
    }
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
