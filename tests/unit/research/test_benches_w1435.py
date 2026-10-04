import pytest

from quant_fund.research import benches_w1435


@pytest.mark.parametrize(
    "fam",
    [
        "bench_blood_qa_studies_family",
        "bench_bone_qa_studies_family",
        "bench_brain_qa_studies_family",
        "bench_heart_qa_studies_family",
        "bench_muscle_qa_studies_family",
        "bench_nerve_qa_studies_family",
    ],
)
def test_benches_w1435(fam):
    out = getattr(benches_w1435, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
