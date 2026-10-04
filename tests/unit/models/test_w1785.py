import pytest


@pytest.mark.parametrize(
    "name",
    [
        "kushinadahime_qa_studies",
        "yamatotakeru_qa_studies",
        "hoori_qa_studies",
        "toyotamahime_qa_studies",
        "benzaiten_qa_studies",
        "jurojin_qa_studies",
    ],
)
def test_w1785_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
