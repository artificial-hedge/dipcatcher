import pytest


@pytest.mark.parametrize(
    "name",
    [
        "cat_sith_qa_studies",
        "cwn_annwn_qa_studies",
        "barghest_qa_studies",
        "black_dog_qa_studies",
        "church_grim_qa_studies",
        "grimalkin_qa_studies",
    ],
)
def test_w1663_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
