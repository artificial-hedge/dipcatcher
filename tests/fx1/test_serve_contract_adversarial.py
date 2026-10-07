"""SYNTHETIC independent checks of the wire-version contract leaf."""

from __future__ import annotations

import inspect

import fx1.serve.contract as contract


def test_api_version_is_a_stable_nonnegative_int_string() -> None:
    assert isinstance(contract.API_VERSION, str)
    assert contract.API_VERSION.strip() == contract.API_VERSION
    assert contract.API_VERSION  # non-empty
    assert int(contract.API_VERSION) >= 1


def test_module_exports_only_the_version() -> None:
    assert contract.__all__ == ["API_VERSION"]


def test_contract_leaf_carries_no_state_or_env_reads() -> None:
    # the contract leaf is one constant — a future version must not grow
    # env mutation, hidden state, or network seams in this file
    src = inspect.getsource(contract)
    for token in ("environ", "open(", "socket", "requests.", "urllib"):
        assert token not in src
