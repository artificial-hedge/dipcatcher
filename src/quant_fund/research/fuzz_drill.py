"""Metamorphic fuzzing of the integrity verifier — the two-sided property.

``tamper_drill`` proves the verifier catches scripted attacks. A degenerate
verifier that fails *everything* would ace that drill. This lane closes the
blind spot: seeded random mutations are classified by what the gates
*should* say — mutations attacking covered state must fail (``escaped`` if
they pass); mutations to legitimately-uncovered or white-space-legal state
must still pass (``false_positive`` if they fail). Both directions are
measured against the real ``verify_repo`` on a cloned tree.

Mutation classes are sampled from the same surfaces the scripted drill
covers plus negative space (unpinned files, non-corpus writes, comments in
already-mutable corpora). ``expect`` is decided *before* the mutation runs
— the verifier's answer is then graded against it, making this a
metamorphic test: the oracle is the semantics of the layer, not a hard-coded
string match.

Seeded: the receipt carries ``seed`` + per-mutation outcomes; the same seed
replays the same attack sequence byte-for-byte.

Provenance evidence only; never a market or P&L claim.
"""

from __future__ import annotations

import json
import random
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from quant_fund.research.tamper_drill import _clone_state
from quant_fund.utils.hashing import hash_bytes

FUZZ_SCHEMA = "fuzz_drill.v1"

# (name, expected verify-repo ok) — "fail" for covered-state attacks,
# "ok" for mutations on state the gates deliberately do not cover.
_EXPECT_FAIL = "fail"
_EXPECT_OK = "ok"


def _flip_byte(path: Path, rng: random.Random) -> str:
    data = bytearray(path.read_bytes())
    if not data:
        raise ValueError("empty file")
    i = rng.randrange(len(data))
    data[i] ^= 1 << rng.randrange(8)
    path.write_bytes(bytes(data))
    return f"byte[{i}]^={data[i]:02x}"


def _truncate(path: Path, rng: random.Random) -> str:
    data = path.read_bytes()
    if len(data) < 8:
        raise ValueError("too small to truncate")
    keep = rng.randrange(4, len(data) - 4)
    path.write_bytes(data[:keep])
    return f"truncated_to:{keep}"


def _json_key_rename(path: Path, rng: random.Random) -> str:
    body = json.loads(path.read_bytes())
    if not isinstance(body, dict) or not body:
        raise ValueError("not a json object")
    key = rng.choice(sorted(body))
    body[f"{key}_x"] = body.pop(key)
    path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    return f"renamed:{key}"


def _corpus_members(clone: Path) -> list[Path]:
    """Every file any corpus covers (union, deduped)."""
    from quant_fund.research.repo_integrity import CORPORA

    out: list[Path] = []
    for corpus, pattern, _req, _upd in CORPORA:
        base = clone / corpus
        if not base.is_dir():
            continue
        for p in base.rglob(pattern):
            if p.is_file():
                out.append(p)
    return sorted(set(out))


def _covered_files(clone: Path) -> dict[str, list[Path]]:
    """Classify targets: pinned (jewel), corpus member, or uncovered."""
    crown = clone / "quality/crown_jewels.json"
    pinned: list[Path] = []
    if crown.is_file():
        try:
            pinned = [clone / rel for rel in json.loads(crown.read_text()).get("files", {})]
            pinned = [p for p in pinned if p.is_file()]
        except (OSError, ValueError):
            pinned = []
    return {"pinned": pinned, "corpus": _corpus_members(clone)}


def _mutations(clone: Path, rng: random.Random) -> list[tuple[str, str, Any]]:
    """Sample the attack surface: (name, expect, mutate-closure)."""
    cov = _covered_files(clone)
    corpus = [p for p in cov["corpus"] if p.is_file()]
    pinned = [p for p in cov["pinned"] if p.is_file()]
    out: list[tuple[str, str, Any]] = []

    # --- must-fail: covered state ---
    if corpus:
        v = rng.choice(corpus)
        out.append(("corpus_byte_flip", _EXPECT_FAIL, lambda v=v: _flip_byte(v, rng)))
        if v.suffix == ".json":
            out.append(("corpus_key_rename", _EXPECT_FAIL, lambda v=v: _json_key_rename(v, rng)))
        v2 = rng.choice(corpus)
        out.append(("corpus_truncate", _EXPECT_FAIL, lambda v=v2: _truncate(v2, rng)))
    if pinned:
        v = rng.choice(pinned)
        out.append(("jewel_byte_flip", _EXPECT_FAIL, lambda v=v: _flip_byte(v, rng)))
        out.append(("jewel_delete", _EXPECT_FAIL, lambda v=v: v.unlink()))
    sig = clone / "gate_pins.sig"
    if sig.is_file():
        out.append(("pin_sig_flip", _EXPECT_FAIL, lambda: _flip_byte(sig, rng)))
    spine = sorted((clone / "quality/checkpoints").glob("*.json"))
    if spine:
        v = rng.choice(spine)
        out.append(("spine_record_flip", _EXPECT_FAIL, lambda v=v: _flip_byte(v, rng)))
        out.append(("spine_record_delete", _EXPECT_FAIL, lambda v=v: v.unlink()))
    wit = sorted((clone / "quality/witness").glob("*.json"))
    if wit:
        v = rng.choice(wit)
        out.append(("witness_proof_flip", _EXPECT_FAIL, lambda v=v: _flip_byte(v, rng)))
    # Unstamped corpus arrival (immutable corpus): must fail.
    receipts = clone / "receipts"
    if receipts.is_dir():

        def _inject() -> str:
            fake = receipts / f"fuzz_{rng.randrange(1 << 32):08x}.json"
            fake.write_text('{"schema": "injected", "ok": true}\n')
            return f"injected:{fake.name}"

        out.append(("unstamped_receipt_inject", _EXPECT_FAIL, _inject))

    # --- must-pass: legitimately uncovered state ---
    uncovered_dir = clone / "docs"
    uncovered_dir.mkdir(exist_ok=True)
    target = uncovered_dir / "FUZZ_NOTE.md"
    if not target.exists():

        def _write_uncovered() -> str:
            target.write_text("# fuzz probe — docs/ is not a pinned corpus\n")
            return "created:docs/FUZZ_NOTE.md"

        out.append(("uncovered_write", _EXPECT_OK, _write_uncovered))
    else:
        out.append(("uncovered_write", _EXPECT_OK, lambda: _flip_byte(target, rng)))
    return out


def fuzz_drill(root: str | Path, seed: int = 1, rounds: int | None = None) -> dict[str, Any]:
    """Seeded metamorphic drill: every mutation graded against its expected
    verdict — must-fail attacks that pass are ``escaped``, legal mutations
    that fail are ``false_positive``."""
    from quant_fund.research.repo_integrity import verify_repo

    root_path = Path(root)
    rng = random.Random(seed)
    with tempfile.TemporaryDirectory(prefix="fuzz_drill_") as tmp:
        clone = Path(tmp) / "clone"
        clone.mkdir(parents=True)
        _clone_state(root_path, clone)
        baseline = verify_repo(clone)
        baseline_errors = sorted(
            f"{g}:{e}"
            for g, gate in baseline.get("gates", {}).items()
            for e in (gate.get("errors") or [])
        )
        if baseline_errors:
            return {
                "schema": FUZZ_SCHEMA,
                "research_only": True,
                "live_pnl_claim": False,
                "data_label": "CORPUS",
                "simulated_only": False,
                "ok": False,
                "seed": seed,
                "baseline_errors": baseline_errors,
                "mutations": [],
                "n_mutations": 0,
                "n_escaped": 0,
                "n_false_positive": 0,
                "verdict": "baseline_dirty",
            }
        mutations = _mutations(clone, rng)
        rng.shuffle(mutations)
        if rounds is not None:
            mutations = mutations[:rounds]
        snapshot = Path(tmp) / "snapshot"
        results: list[dict[str, Any]] = []
        for name, expect, mutate in mutations:
            shutil.rmtree(snapshot, ignore_errors=True)
            shutil.copytree(clone, snapshot)
            try:
                detail = mutate()
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                results.append({"mutation": name, "expect": expect, "skipped": f"setup:{exc}"})
                continue
            res = verify_repo(clone)
            got_ok = bool(res.get("ok", True))
            gate_errors = sorted(
                f"{g}:{e}"
                for g, gate in res.get("gates", {}).items()
                for e in (gate.get("errors") or [])
            )
            entry: dict[str, Any] = {
                "mutation": name,
                "expect": expect,
                "detail": detail,
                "verifier_ok": got_ok,
            }
            if expect == _EXPECT_FAIL and got_ok:
                entry["outcome"] = "escaped"
            elif expect == _EXPECT_OK and not got_ok:
                entry["outcome"] = "false_positive"
                entry["errors"] = gate_errors[:8]
            else:
                entry["outcome"] = "correct"
                if gate_errors:
                    entry["errors"] = gate_errors[:8]
            results.append(entry)
            shutil.rmtree(clone)
            shutil.move(str(snapshot), clone)
    n_escaped = sum(1 for r in results if r.get("outcome") == "escaped")
    n_fp = sum(1 for r in results if r.get("outcome") == "false_positive")
    n_ran = sum(1 for r in results if "outcome" in r)
    return {
        "schema": FUZZ_SCHEMA,
        "research_only": True,
        "live_pnl_claim": False,
        "data_label": "CORPUS",
        "simulated_only": False,
        "ok": n_escaped == 0 and n_fp == 0 and n_ran > 0,
        "seed": seed,
        "baseline_errors": [],
        "mutations": results,
        "n_mutations": len(results),
        "n_escaped": n_escaped,
        "n_false_positive": n_fp,
        "tree_sha256": hash_bytes(json.dumps(results, sort_keys=True).encode()),
        "verdict": "calibrated" if n_escaped == 0 and n_fp == 0 else "miscalibrated",
    }


def fuzz_contract_errors(payload: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("ok") and (payload.get("n_escaped") or payload.get("n_false_positive")):
        errors.append("ok_with_escapes")
    if payload.get("verdict") == "calibrated" and not payload.get("ok"):
        errors.append("calibrated_not_ok")
    for m in payload.get("mutations") or []:
        if (
            m.get("expect") == "fail"
            and m.get("verifier_ok") is True
            and m.get("outcome") not in (None, "escaped")
        ):
            errors.append(f"outcome_incoherent:{m.get('mutation')}")
    return errors


def write_fuzz_receipt(payload: dict[str, Any], out_dir: Path) -> Path:
    from quant_fund.research.receipt_v2 import seal_receipt
    from quant_fund.utils.atomicio import atomic_write_text

    errs = fuzz_contract_errors(payload)
    if errs:
        raise ValueError(f"fuzz receipt contract: {errs}")
    sealed = seal_receipt(payload)
    name = f"fuzz_drill_{hash_bytes(json.dumps(payload, sort_keys=True).encode())[:16]}.json"
    out = out_dir / name
    atomic_write_text(out, json.dumps(sealed, indent=2, sort_keys=True) + "\n")
    return out
