import pytest

from quant_fund.research import benches_w1804


@pytest.mark.parametrize(
    "fam",
    [
        "bench_enki2_qa_studies_family",
        "bench_enlil2_qa_studies_family",
        "bench_gelal2_qa_studies_family",
        "bench_namtar2_qa_studies_family",
        "bench_ninurta2_qa_studies_family",
        "bench_zababa2_qa_studies_family",
    ],
)
def test_benches_w1804(fam):
    out = getattr(benches_w1804, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
