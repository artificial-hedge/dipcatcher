import pytest


@pytest.mark.parametrize(
    "name",
    [
        "divsalar_qa_studies",
        "pari_vatra_qa_studies",
        "urvan_qa_studies",
        "khrafstra_qa_studies",
        "srosh_demon_qa_studies",
        "leshenka_qa_studies",
    ],
)
def test_w1911_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
