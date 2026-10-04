import pytest

from quant_fund.research import benches_w1480


@pytest.mark.parametrize(
    "fam",
    [
        "bench_adder_qa_studies_family",
        "bench_boa_qa_studies_family",
        "bench_krait_qa_studies_family",
        "bench_mamba_qa_studies_family",
        "bench_monitor_qa_studies_family",
        "bench_taipan_qa_studies_family",
    ],
)
def test_benches_w1480(fam):
    out = getattr(benches_w1480, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
