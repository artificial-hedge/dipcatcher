import pytest


@pytest.mark.parametrize(
    "name",
    [
        "dellingr_qa_studies",
        "gna_qa_studies",
        "jord_qa_studies",
        "sol_qa_studies",
        "mani_qa_studies",
        "sigyn_qa_studies",
    ],
)
def test_w1755_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
