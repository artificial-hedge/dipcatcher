import pytest

from quant_fund.research import benches_w1611


@pytest.mark.parametrize(
    "fam",
    [
        "bench_crowned_lemur_qa_studies_family",
        "bench_fat_tailed_qa_studies_family",
        "bench_fork_marked_qa_studies_family",
        "bench_needle_clawed_qa_studies_family",
        "bench_ringtail_qa_studies_family",
        "bench_sifaka_qa_studies_family",
    ],
)
def test_benches_w1611(fam):
    out = getattr(benches_w1611, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
