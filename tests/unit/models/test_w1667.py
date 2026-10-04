import pytest


@pytest.mark.parametrize(
    "name",
    [
        "nymph_qa_studies",
        "dryad_qa_studies",
        "faun_qa_studies",
        "centauride_qa_studies",
        "nereid_qa_studies",
        "hamadryad_qa_studies",
    ],
)
def test_w1667_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
