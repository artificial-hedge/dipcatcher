import pytest


@pytest.mark.parametrize(
    "name",
    [
        "penanggalan_qa_studies",
        "pontianak_qa_studies",
        "toyol_qa_studies",
        "kum_kum_qa_studies",
        "pelesit_qa_studies",
        "bajang_qa_studies",
    ],
)
def test_w1898_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
