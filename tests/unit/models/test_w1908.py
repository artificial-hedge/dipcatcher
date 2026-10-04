import pytest


@pytest.mark.parametrize(
    "name",
    [
        "mula_sem_cabeca_qa_studies",
        "lobisomem_qa_studies",
        "anhanga_qa_studies",
        "jurupari_qa_studies",
        "dvorovoy_qa_studies",
        "bolotnik_qa_studies",
    ],
)
def test_w1908_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
