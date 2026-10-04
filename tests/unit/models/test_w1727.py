import pytest


@pytest.mark.parametrize(
    "name",
    [
        "erlug_qa_studies",
        "tenger_qa_studies",
        "ulgan_qa_studies",
        "etseg_qa_studies",
        "manzan_qa_studies",
        "otgon_qa_studies",
    ],
)
def test_w1727_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
