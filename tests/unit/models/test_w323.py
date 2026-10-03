"""Wave-323 PL-7 module unit tests."""

from __future__ import annotations

import pytest

from quant_fund.models.alg_effects import do, handle, ret
from quant_fund.models.free_monad import bind, op, pure, run
from quant_fund.models.gradual_types import DYN, Blame, cast, ceq
from quant_fund.models.row_types import subtype, unify_row
from quant_fund.models.session_types import SessionError, check, dual
from quant_fund.models.shift_reset import eval_reset


def test_free_writer() -> None:
    log: list[str] = []
    v = run(bind(op(("tell", "a")), lambda _: pure(1)), lambda f: log.append(f[1]) or f[1])
    assert v == 1 and log == ["a"]


def test_effects_state() -> None:
    st = {"v": 1}

    def h(eff, resume):  # type: ignore[no-untyped-def]
        if eff[0] == "get":
            return resume(st["v"])
        st["v"] = eff[1]
        return resume(None)

    out = handle(do(("put", 9), lambda _: do(("get",), lambda n: ret(n))), {"get": h, "put": h})
    assert out == 9


def test_shift() -> None:
    assert eval_reset(("add", ("lit", 2), ("reset", ("lit", 3)))) == 5


def test_row_subtype() -> None:
    i = ("base", "int")
    assert subtype(("rec", {"a": i, "b": i}, None), ("rec", {"a": i}, None))
    assert unify_row(("rec", {"x": i}, None), "y", {}) is None


def test_session_dual() -> None:
    n = ("base", "int")
    s = ("send", n, ("recv", n, ("end",)))
    assert dual(s) == ("recv", n, ("send", n, ("end",)))
    check(("send", "c", 1, ("recv", "c", "r", ("end",))), s)
    with pytest.raises(SessionError):
        check(("recv", "c", "x", ("end",)), s)


def test_gradual_cast() -> None:
    assert ceq(DYN, ("int",))
    assert cast(5, DYN, ("int",)) == 5
    with pytest.raises(Blame):
        cast(True, DYN, ("int",), "L")
