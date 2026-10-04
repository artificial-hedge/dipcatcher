import pytest

from quant_fund.research import benches_w1720


@pytest.mark.parametrize(
    "fam",
    [
        "bench_akka_qa_studies_family",
        "bench_juksakka_qa_studies_family",
        "bench_lieaibolmmai_qa_studies_family",
        "bench_radien_qa_studies_family",
        "bench_sarakka_qa_studies_family",
        "bench_ukso_qa_studies_family",
    ],
)
def test_benches_w1720(fam):
    out = getattr(benches_w1720, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
