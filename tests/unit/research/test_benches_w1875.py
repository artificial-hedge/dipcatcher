import pytest

from quant_fund.research import benches_w1875


@pytest.mark.parametrize(
    "fam",
    [
        "bench_abdastartus_qa_studies_family",
        "bench_bariha_qa_studies_family",
        "bench_bodastart_qa_studies_family",
        "bench_mider_qa_studies_family",
        "bench_reshef_qa_studies_family",
        "bench_safon_qa_studies_family",
    ],
)
def test_benches_w1875(fam):
    out = getattr(benches_w1875, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
