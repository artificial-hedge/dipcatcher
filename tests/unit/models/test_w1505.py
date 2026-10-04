import pytest


@pytest.mark.parametrize(
    "name",
    [
        "teal_qa_studies",
        "pochard_qa_studies",
        "wigeon_qa_studies",
        "shoveler_qa_studies",
        "pintail_qa_studies",
        "gadwall_qa_studies",
    ],
)
def test_w1505_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
