"""Wave-252 interpreters-3 canon tests."""

from quant_fund.models.anf_cps import _eval, _is_anf, anf, bench_anf_cps
from quant_fund.models.compacting_gc import CopyGC, bench_compacting_gc
from quant_fund.models.dispatch_table import Instance, Klass, bench_dispatch_table
from quant_fund.models.gen_gc import GenGC, bench_gen_gc
from quant_fund.models.poly_inline_cache import PIC, bench_poly_inline_cache
from quant_fund.models.trampoline_tc import _even, _sum_to, bench_trampoline_tc, trampoline


def test_gen_gc_minor():
    gc = GenGC()
    gc.alloc(1)
    gc.roots.add(1)
    gc.minor_collect()
    assert gc.objs[1].gen == 1


def test_gen_gc_bench():
    assert bench_gen_gc()["synthetic_preserved"] == 1.0


def test_compacting_basic():
    gc = CopyGC(4)
    for i in range(4):
        gc.alloc(i)
    gc.roots = [0]
    gc.from_space[0][0].append(1)
    heap = gc.collect()
    assert len(heap) == 2


def test_compacting_bench():
    assert bench_compacting_gc()["synthetic_topology_exact"] == 1.0


def test_dispatch_basic():
    a = Klass("A", None)
    a.define("m", "x")
    b = Klass("B", a)
    b.define("m", "y")
    assert Instance(b).call("m") == "y@B"


def test_dispatch_bench():
    assert bench_dispatch_table()["synthetic_dispatch_exact"] == 1.0


def test_pic_states():
    pic = PIC()
    vt = {"A": {"m": "1"}, "B": {"m": "2"}}
    pic.dispatch("A", "m", vt)
    assert pic.state == "mono"
    pic.dispatch("B", "m", vt)
    assert pic.state == "poly"


def test_pic_bench():
    assert bench_poly_inline_cache()["synthetic_dispatch_correct"] == 1.0


def test_anf_form():
    t = ("add", ("mul", ("var", "x"), ("lit", 2.0)), ("lit", 1.0))
    a = anf(t)
    assert _is_anf(a)
    assert _eval(a, {"x": 3.0}) == _eval(t, {"x": 3.0})


def test_anf_bench():
    assert bench_anf_cps()["synthetic_eval_preserved"] == 1.0


def test_trampoline_basic():
    assert trampoline(_even, 10) is True
    assert trampoline(_sum_to, 5) == 15


def test_trampoline_bench():
    assert bench_trampoline_tc()["synthetic_tail_exact"] == 1.0
