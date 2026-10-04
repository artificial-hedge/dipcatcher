import pytest

from quant_fund.research import benches_w1578


@pytest.mark.parametrize(
    "fam",
    [
        "bench_gerenuk_qa_studies_family",
        "bench_markhor_qa_studies_family",
        "bench_nilgai_qa_studies_family",
        "bench_okapi_qa_studies_family",
        "bench_saiga_qa_studies_family",
        "bench_takin_qa_studies_family",
    ],
)
def test_benches_w1578(fam):
    out = getattr(benches_w1578, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
