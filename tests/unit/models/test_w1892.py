import pytest


@pytest.mark.parametrize(
    "name",
    [
        "ananke_libya_qa_studies",
        "mithra_iran_qa_studies",
        "mitra_persian_qa_studies",
        "perangal_qa_studies",
        "al_basti_qa_studies",
        "encantado_qa_studies",
    ],
)
def test_w1892_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
