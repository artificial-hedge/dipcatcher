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
    args = ap.parse_args()

    receipts_dir = args.receipts.resolve()
    out_path = args.out.resolve()
    # A previous output inside the scanned dir would be harvested as an
    # input: its corpus_evalue re-entering the product would double-count
    # evidence (the product bound needs independence). It is regenerated
    # below, so drop it from the input set.
    if out_path.parent == receipts_dir and out_path.is_file():
        out_path.unlink()

    report = corpus_audit(receipts_dir, q=args.q, glob=args.glob)
    report.pop("code_revision", None)
    report.pop("meta", None)
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
