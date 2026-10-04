import pytest

from quant_fund.research import benches_w1740


@pytest.mark.parametrize(
    "fam",
    [
        "bench_anat_qa_studies_family",
        "bench_astarte_qa_studies_family",
        "bench_kothar_qa_studies_family",
        "bench_mot_qa_studies_family",
        "bench_resheph_qa_studies_family",
        "bench_yam_qa_studies_family",
    ],
)
def test_benches_w1740(fam):
    out = getattr(benches_w1740, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
