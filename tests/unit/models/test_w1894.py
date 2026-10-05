import pytest


@pytest.mark.parametrize(
    "name",
    [
        "ekimmu_qa_studies",
        "gidim_qa_studies",
        "maskim_qa_studies",
        "asakku_qa_studies",
        "etemmu_qa_studies",
        "sebettu_qa_studies",
    ],
)
def test_w1894_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
