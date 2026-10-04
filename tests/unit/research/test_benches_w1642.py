import pytest

from quant_fund.research import benches_w1642


@pytest.mark.parametrize(
    "fam",
    [
        "bench_basilisk_2_qa_studies_family",
        "bench_chimera_2_qa_studies_family",
        "bench_cockatrice_qa_studies_family",
        "bench_manticore_2_qa_studies_family",
        "bench_sphinx_2_qa_studies_family",
        "bench_wyvern_2_qa_studies_family",
    ],
)
def test_benches_w1642(fam):
    out = getattr(benches_w1642, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
