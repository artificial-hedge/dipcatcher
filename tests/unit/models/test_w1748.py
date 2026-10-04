import pytest


@pytest.mark.parametrize(
    "name",
    [
        "palden_lhamo_qa_studies",
        "dorje_legpa_qa_studies",
        "beg_tse_qa_studies",
        "pehar_qa_studies",
        "tsiu_marpo_qa_studies",
        "tsen_god_qa_studies",
    ],
)
def test_w1748_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
