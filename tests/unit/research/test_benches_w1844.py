import pytest

from quant_fund.research import benches_w1844


@pytest.mark.parametrize(
    "fam",
    [
        "bench_aram2_qa_studies_family",
        "bench_ashima_qa_studies_family",
        "bench_baalshamin_qa_studies_family",
        "bench_resheph2_qa_studies_family",
        "bench_rimmon_qa_studies_family",
        "bench_sahr_qa_studies_family",
    ],
)
def test_benches_w1844(fam):
    out = getattr(benches_w1844, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
