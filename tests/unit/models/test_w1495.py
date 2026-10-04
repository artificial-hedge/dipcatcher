import pytest


@pytest.mark.parametrize(
    "name",
    [
        "weasel_qa_studies",
        "grison_qa_studies",
        "sable_qa_studies",
        "tayra_qa_studies",
        "zorilla_qa_studies",
        "stoat_qa_studies",
    ],
)
def test_w1495_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
