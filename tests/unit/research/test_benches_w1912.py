import pytest

from quant_fund.research import benches_w1912


@pytest.mark.parametrize(
    "fam",
    [
        "bench_belun_qa_studies_family",
        "bench_indrik_qa_studies_family",
        "bench_koshchey_qa_studies_family",
        "bench_mavka_qa_studies_family",
        "bench_psoglav_qa_studies_family",
        "bench_triglav_qa_studies_family",
    ],
)
def test_benches_w1912(fam):
    out = getattr(benches_w1912, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
