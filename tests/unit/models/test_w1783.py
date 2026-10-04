import pytest


@pytest.mark.parametrize(
    "name",
    [
        "baldur_qa_studies",
        "hodr_qa_studies",
        "hermodr_qa_studies",
        "njord_qa_studies",
        "skadi_qa_studies",
        "freyr_qa_studies",
    ],
)
def test_w1783_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
