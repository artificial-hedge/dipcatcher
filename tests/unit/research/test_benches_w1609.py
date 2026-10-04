import pytest

from quant_fund.research import benches_w1609


@pytest.mark.parametrize(
    "fam",
    [
        "bench_black_lemur_qa_studies_family",
        "bench_brown_lemur_qa_studies_family",
        "bench_dwarf_lemur_qa_studies_family",
        "bench_mongoose_lemur_qa_studies_family",
        "bench_ruffed_qa_studies_family",
        "bench_sportive_lemur_qa_studies_family",
    ],
)
def test_benches_w1609(fam):
    out = getattr(benches_w1609, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
