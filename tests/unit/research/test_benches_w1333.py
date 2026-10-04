import pytest

from quant_fund.research import benches_w1333


@pytest.mark.parametrize(
    "fam",
    [
        "bench_fava_studies_family",
        "bench_polyglo_tox_studies_family",
        "bench_regard_eval_studies_family",
        "bench_unqover_studies_family",
        "bench_vlur_studies_family",
        "bench_xlsum_studies_family",
    ],
)
def test_benches_w1333(fam):
    out = getattr(benches_w1333, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
