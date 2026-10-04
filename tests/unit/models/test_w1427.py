import pytest


@pytest.mark.parametrize(
    "name",
    [
        "coast_qa_studies",
        "field_qa_studies",
        "desert_qa_studies",
        "forest_qa_studies",
        "glacier_qa_studies",
        "canyon_qa_studies",
    ],
)
def test_w1427_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
