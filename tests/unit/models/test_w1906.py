import pytest


@pytest.mark.parametrize(
    "name",
    [
        "yama_waro_qa_studies",
        "nakisawame_qa_studies",
        "amefuri_kozo_qa_studies",
        "dosanjin_qa_studies",
        "fuon_qa_studies",
        "kejoro_qa_studies",
    ],
)
def test_w1906_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
