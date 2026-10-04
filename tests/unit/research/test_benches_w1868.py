import pytest

from quant_fund.research import benches_w1868


@pytest.mark.parametrize(
    "fam",
    [
        "bench_anat_punic_qa_studies_family",
        "bench_carthage_punic_qa_studies_family",
        "bench_el_punic_qa_studies_family",
        "bench_hadad_punic_qa_studies_family",
        "bench_moloch_punic_qa_studies_family",
        "bench_reshef_punic_qa_studies_family",
    ],
)
def test_benches_w1868(fam):
    out = getattr(benches_w1868, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
