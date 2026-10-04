import pytest


@pytest.mark.parametrize(
    "name",
    [
        "medghassen_qa_studies",
        "arzew_qa_studies",
        "cilteni_qa_studies",
        "essuf_qa_studies",
        "tanezruft_qa_studies",
        "argemm_qa_studies",
    ],
)
def test_w1884_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
