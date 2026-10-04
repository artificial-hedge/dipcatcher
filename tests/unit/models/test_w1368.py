import pytest


@pytest.mark.parametrize(
    "name",
    [
        "chr_f_studies",
        "prism_mt_studies",
        "nist_metric_studies",
        "sacrebleu_lite_studies",
        "ter_lite_studies",
        "mover_score_studies",
    ],
)
def test_w1368_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
