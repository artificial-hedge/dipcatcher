from quant_fund.models.braid_rep import bench_braid_rep
from quant_fund.models.quantum_double import bench_quantum_double
from quant_fund.models.ribbon_cat import bench_ribbon_cat
from quant_fund.models.rtt_formalism import bench_rtt_formalism
from quant_fund.models.yang_baxter import bench_yang_baxter
from quant_fund.models.yangian import bench_yangian


def test_yang_baxter():
    assert bench_yang_baxter()["synthetic_yang_baxter"] == 1.0


def test_braid_rep():
    assert bench_braid_rep()["synthetic_braid_rep"] == 1.0


def test_yangian():
    assert bench_yangian()["synthetic_yangian"] == 1.0


def test_rtt_formalism():
    assert bench_rtt_formalism()["synthetic_rtt_formalism"] == 1.0


def test_quantum_double():
    assert bench_quantum_double()["synthetic_quantum_double"] == 1.0


def test_ribbon_cat():
    assert bench_ribbon_cat()["synthetic_ribbon_cat"] == 1.0
