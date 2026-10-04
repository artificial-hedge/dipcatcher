import pytest

from quant_fund.research import benches_w1710


@pytest.mark.parametrize(
    "fam",
    [
        "bench_anat_qa_studies_family",
        "bench_asherah_qa_studies_family",
        "bench_baal_qa_studies_family",
        "bench_lotan_qa_studies_family",
        "bench_mot_qa_studies_family",
        "bench_yam_qa_studies_family",
    ],
)
def test_benches_w1710(fam):
    out = getattr(benches_w1710, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
