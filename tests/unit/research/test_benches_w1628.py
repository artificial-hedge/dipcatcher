import pytest

from quant_fund.research import benches_w1628


@pytest.mark.parametrize(
    "fam",
    [
        "bench_colugo_qa_studies_family",
        "bench_geoffroy_qa_studies_family",
        "bench_mandarin_qa_studies_family",
        "bench_moray_eel_qa_studies_family",
        "bench_pangolin_2_qa_studies_family",
        "bench_satyr_qa_studies_family",
    ],
)
def test_benches_w1628(fam):
    out = getattr(benches_w1628, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
