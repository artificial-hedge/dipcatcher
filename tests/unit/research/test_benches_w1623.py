import pytest

from quant_fund.research import benches_w1623


@pytest.mark.parametrize(
    "fam",
    [
        "bench_altai_qa_studies_family",
        "bench_blood_pheasant_qa_studies_family",
        "bench_chukar_qa_studies_family",
        "bench_monal_qa_studies_family",
        "bench_snow_partridge_qa_studies_family",
        "bench_wallcreeper_qa_studies_family",
    ],
)
def test_benches_w1623(fam):
    out = getattr(benches_w1623, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
