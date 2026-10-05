"""Corpus-inference drill — one sealed meta-receipt over committed receipts.

Runs ``corpus_audit`` over a receipts directory: every p-value key at any
depth pools into a single Benjamini-Hochberg family at level ``q``, and
every e-value merges by product (Shafer 2021). The sealed
``corpus_inference.v1`` receipt derives ``data_label`` from the input
payloads — one distinct label, ``MIXED``, or ``UNKNOWN`` — and pins each
input's digest so family membership is reproducible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from quant_fund.research.corpus_inference import corpus_audit
from quant_fund.research.receipt_v2 import seal_receipt, verify_receipt_payload


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--receipts", type=Path, default=Path("receipts"))
    ap.add_argument("--out", type=Path, default=Path("receipts/corpus_real_drill.json"))
    ap.add_argument("--q", type=float, default=0.05)
    ap.add_argument("--glob", default="*.json")
    ap.add_argument(
        "--membership",
        type=Path,
        help=(
            "JSON list of receipt basenames pinning the audit's input set; "
            "without it the glob audits whatever the dir holds at run time "
            "(byte-reproducible replays require this pin)."
        ),
    )
    args = ap.parse_args()

    members = None
    membership_sha256 = None
    if args.membership is not None:
        raw_membership = args.membership.read_bytes()
        loaded = json.loads(raw_membership)
        if not isinstance(loaded, list) or not all(isinstance(m, str) for m in loaded):
            raise SystemExit("membership file must be a JSON list of basenames")
        members = set(loaded)
        membership_sha256 = hashlib.sha256(raw_membership).hexdigest()

    receipts_dir = args.receipts.resolve()
    out_path = args.out.resolve()
    # A previous output inside the scanned dir would be harvested as an
    # input: its corpus_evalue re-entering the product would double-count
    # evidence (the product bound needs independence). It is regenerated
    # below, so drop it from the input set.
    if out_path.parent == receipts_dir and out_path.is_file():
        out_path.unlink()

    report = corpus_audit(receipts_dir, q=args.q, glob=args.glob, members=members)
    # Volatile provenance must not enter the sealed artifact — the replay
    # carrier re-runs this script at a different commit and compares bytes.
    for stamp in (
        "code_revision",
        "meta",
        "generated_at",
        "generated_at_commit",
        "git_revision",
    ):
        report.pop(stamp, None)
    if membership_sha256 is not None:
        report["params"]["membership"] = str(args.membership)
        report["params"]["membership_sha256"] = membership_sha256
    sealed = seal_receipt(report)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(sealed, indent=2, sort_keys=True) + "\n")

    check = verify_receipt_payload(sealed, out_path)
    for name, label in sorted(report["params"]["input_labels"].items()):
        print(f"  {name}: {label}")
    print(
        f"receipts={report['n_receipts']} p_findings={report['n_p_findings']} "
        f"e_findings={report['n_e_findings']} survivors={report['n_survivors']} "
        f"corpus_evalue={report['corpus_evalue']:.6g} "
        f"reject_at_alpha={report['corpus_reject_at_alpha']} "
        f"label={report['data_label']} parse_errors={report['n_parse_errors']} "
        f"seal_valid={check['valid']}"
    )
    if not check["valid"]:
        print(f"seal verification failed: {check['errors']}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
