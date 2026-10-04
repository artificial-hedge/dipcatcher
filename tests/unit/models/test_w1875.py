import pytest


@pytest.mark.parametrize(
    "name",
    [
        "bariha_qa_studies",
        "reshef_qa_studies",
        "abdastartus_qa_studies",
        "bodastart_qa_studies",
        "safon_qa_studies",
        "mider_qa_studies",
    ],
)
def test_w1875_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
