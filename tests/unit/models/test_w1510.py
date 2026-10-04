import pytest


@pytest.mark.parametrize(
    "name",
    [
        "magpie_qa_studies",
        "rook_qa_studies",
        "jay_qa_studies",
        "jackdaw_qa_studies",
        "chough_qa_studies",
        "crow_qa_studies",
    ],
)
def test_w1510_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
