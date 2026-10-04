import pytest


@pytest.mark.parametrize(
    "name",
    [
        "daffodil_qa_studies",
        "foxglove_qa_studies",
        "daisy_qa_studies",
        "iris_qa_studies",
        "poppy_qa_studies",
        "crocus_qa_studies",
    ],
)
def test_w1471_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
