import pytest

from quant_fund.research import benches_w1505


@pytest.mark.parametrize(
    "fam",
    [
        "bench_gadwall_qa_studies_family",
        "bench_pintail_qa_studies_family",
        "bench_pochard_qa_studies_family",
        "bench_shoveler_qa_studies_family",
        "bench_teal_qa_studies_family",
        "bench_wigeon_qa_studies_family",
    ],
)
def test_benches_w1505(fam):
    out = getattr(benches_w1505, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
