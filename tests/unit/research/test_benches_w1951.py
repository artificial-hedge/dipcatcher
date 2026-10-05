import pytest

from quant_fund.research import benches_w1951


@pytest.mark.parametrize(
    "fam",
    [
        "bench_berith_qa_studies_family",
        "bench_bune_qa_studies_family",
        "bench_forneus_qa_studies_family",
        "bench_glasya_labolas_qa_studies_family",
        "bench_naberius_qa_studies_family",
        "bench_ronove_qa_studies_family",
    ],
)
def test_benches_w1951(fam):
    out = getattr(benches_w1951, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
