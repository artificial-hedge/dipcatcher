"""Wave-246 language-runtime canon tests."""

from quant_fund.models.bytecode_vm import bench_bytecode_vm, compile_expr, run
from quant_fund.models.closure_conv import bench_closure_conv, eval_cc
from quant_fund.models.inline_cache import Klass, Site, bench_inline_cache
from quant_fund.models.nan_tagging import bench_nan_tagging, decode, encode
from quant_fund.models.tail_call_tramp import _sum_tail, bench_tail_call_tramp, tramp
from quant_fund.models.threaded_interp import bench_threaded_interp, run_switch, run_threaded


def test_vm_basic():
    code = []
    compile_expr(("add", ("lit", 1), ("mul", ("lit", 2), ("lit", 3))), code)
    assert run(code) == 7


def test_vm_bench():
    assert bench_bytecode_vm()["synthetic_vm_matches_eval"] == 1.0


def test_threaded_basic():
    prog = [("push", 1), ("push", 2), ("add",)]
    assert run_switch(prog) == run_threaded(prog) == [3]


def test_threaded_bench():
    assert bench_threaded_interp()["synthetic_dispatch_equivalent"] == 1.0


def test_closure_basic():
    lam = ("lam", "x", ("add", ("var", "x"), ("var", "c")))
    assert eval_cc(("app", lam, ("lit", 5)), {"c": 10}) == 15


def test_closure_bench():
    assert bench_closure_conv()["synthetic_converted_eval_correct"] == 1.0


def test_tramp_basic():
    assert tramp(_sum_tail, 100) == 5050


def test_tramp_bench():
    assert bench_tail_call_tramp()["synthetic_deep_recursion"] == 1.0


def test_cache_basic():
    k = Klass("A", {"m": 7})
    s = Site()
    assert s.call(k, "m") == 7 and s.call(k, "m") == 7 and s.hits == 1


def test_cache_bench():
    assert bench_inline_cache()["synthetic_monomorphic_hitrate"] == 1.0


def test_tagging_basic():
    assert decode(encode(42)) == 42
    assert decode(encode(-7)) == -7
    assert decode(encode(None)) is None
    assert decode(encode(("ptr", 99))) == ("ptr", 99)


def test_tagging_bench():
    assert bench_nan_tagging()["synthetic_roundtrip"] == 1.0
