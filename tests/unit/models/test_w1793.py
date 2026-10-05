import pytest


@pytest.mark.parametrize(
    "name",
    [
        "quirinus_qa_studies",
        "faunus_qa_studies",
        "tellus_qa_studies",
        "larunda_qa_studies",
        "mutina_qa_studies",
        "cluentia_qa_studies",
    ],
)
def test_w1793_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
