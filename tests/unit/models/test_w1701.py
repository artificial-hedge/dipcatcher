import pytest


@pytest.mark.parametrize(
    "name",
    [
        "viracocha_qa_studies",
        "inti_qa_studies",
        "mamaquilla_qa_studies",
        "pachamama_qa_studies",
        "supay_qa_studies",
        "illapa_qa_studies",
    ],
)
def test_w1701_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
