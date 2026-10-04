import pytest


@pytest.mark.parametrize(
    "name",
    [
        "bone_qa_studies",
        "heart_qa_studies",
        "brain_qa_studies",
        "muscle_qa_studies",
        "nerve_qa_studies",
        "blood_qa_studies",
    ],
)
def test_w1435_contract(name):
    m = __import__(f"quant_fund.models.{name}", fromlist=[name])
    assert getattr(m, f"{name}_ok")(True, True) is True
    assert getattr(m, f"{name}_ok")(False, True) is False
    aux = object()
    assert getattr(m, f"{name}_aux")(aux) is aux
