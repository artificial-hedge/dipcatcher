"""Tests for commit_reveal — hash commitment binding/hiding."""

from __future__ import annotations

from quant_fund.models.commit_reveal import bench_commit_reveal, commit, open_ok


def test_binding_same_nonce():
    # adversarial: open a commitment with the SAME nonce but a different
    # message — this is the case a weak (message-independent) commit fails.
    c = commit(0xBEEF, 0xDEAD)
    assert open_ok(c, 0xBEEF, 0xDEAD)
    assert not open_ok(c, 0xBEEF ^ 1, 0xDEAD)


def test_binding_many_messages_same_nonce():
    for m in range(1, 40):
        c = commit(m, 777)
        assert not open_ok(c, m + 1, 777)


def test_message_influences_commitment():
    # commit() must depend on m: two messages, same nonce -> different c.
    assert commit(1, 42) != commit(2, 42)


def test_bench():
    out = bench_commit_reveal()
    assert out["synthetic_open_verifies"] == 1.0
    assert out["synthetic_binding"] == 1.0
    assert out["synthetic_randomized_hiding"] == 1.0
