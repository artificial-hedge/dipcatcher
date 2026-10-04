import pytest


@pytest.mark.parametrize(
    "name",
    [
        "aracari_qa_studies",
        "honeyguide_qa_studies",
        "barbet_qa_studies",
        "trogon_qa_studies",
        "quetzal_qa_studies",
        "hornbill_qa_studies",
    ],
)
def test_w1533_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
