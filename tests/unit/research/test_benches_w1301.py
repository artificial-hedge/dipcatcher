import pytest

from quant_fund.research import benches_w1301


@pytest.mark.parametrize(
    "fam",
    [
        "bench_backdoor_studies_family",
        "bench_clean_label_studies_family",
        "bench_data_poison_studies_family",
        "bench_neural_cleanse_studies_family",
        "bench_spectral_signature_studies_family",
        "bench_trojan_studies_family",
    ],
)
def test_benches_w1301(fam):
    out = getattr(benches_w1301, fam)()
    assert all(k.startswith("synthetic_") for k in out)
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in out.values())
    assert len(out) >= 1
