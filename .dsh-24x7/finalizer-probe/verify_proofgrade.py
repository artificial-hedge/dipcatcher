"""Disposable wrapper: run the real finalization verifier in --proof-grade mode.

Used to deterministically induce a post-verification failure: proof-grade mode
always exits 1 on a diagnostic (UNPROVEN) manifest, even when fully valid.
"""
import sys

sys.path.insert(0, r"D:\dipcatcher\scripts")
import verify_sota_finalization as _v

if __name__ == "__main__":
    raise SystemExit(_v.main([sys.argv[1], "--proof-grade"]))
