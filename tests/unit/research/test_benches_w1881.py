import pytest

from quant_fund.research import benches_w1881


@pytest.mark.parametrize(
    "fam",
    [
        "bench_juba_qa_studies_family",
        "bench_jugurtha_qa_studies_family",
        "bench_massinissa_qa_studies_family",
        "bench_micipsa_qa_studies_family",
        "bench_naravas_qa_studies_family",
        "bench_syphax_qa_studies_family",
    ],
)
def test_benches_w1881(fam):
    out = getattr(benches_w1881, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
