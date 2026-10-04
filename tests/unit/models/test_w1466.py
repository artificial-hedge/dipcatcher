import pytest


@pytest.mark.parametrize(
    "name",
    [
        "cathedral_qa_studies",
        "crag_qa_studies",
        "chasm_qa_studies",
        "plateau_qa_studies",
        "ravine_qa_studies",
        "basalt_qa_studies",
    ],
)
def test_w1466_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
