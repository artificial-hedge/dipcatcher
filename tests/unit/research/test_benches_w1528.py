import pytest

from quant_fund.research import benches_w1528


@pytest.mark.parametrize(
    "fam",
    [
        "bench_calcite_qa_studies_family",
        "bench_feldspar_qa_studies_family",
        "bench_fluorite_qa_studies_family",
        "bench_gypsum_qa_studies_family",
        "bench_olivine_qa_studies_family",
        "bench_quartz_qa_studies_family",
    ],
)
def test_benches_w1528(fam):
    out = getattr(benches_w1528, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
