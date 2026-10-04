import pytest

from quant_fund.research import benches_w1751


@pytest.mark.parametrize(
    "fam",
    [
        "bench_anubis_qa_studies_family",
        "bench_bastet_qa_studies_family",
        "bench_khonsu_qa_studies_family",
        "bench_min_qa_studies_family",
        "bench_neith_qa_studies_family",
        "bench_sobek_qa_studies_family",
    ],
)
def test_benches_w1751(fam):
    out = getattr(benches_w1751, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
