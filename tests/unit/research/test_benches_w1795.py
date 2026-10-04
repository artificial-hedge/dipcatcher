import pytest

from quant_fund.research import benches_w1795


@pytest.mark.parametrize(
    "fam",
    [
        "bench_anubis2_qa_studies_family",
        "bench_isis2_qa_studies_family",
        "bench_khonsu2_qa_studies_family",
        "bench_osiris2_qa_studies_family",
        "bench_ra2_qa_studies_family",
        "bench_sobek2_qa_studies_family",
    ],
)
def test_benches_w1795(fam):
    out = getattr(benches_w1795, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
