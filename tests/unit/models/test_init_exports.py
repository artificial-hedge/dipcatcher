"""Pin: every lazy export in quant_fund.models.__init__ must resolve."""


def test_lazy_exports_resolve() -> None:
    import quant_fund.models as m

    assert sorted(m.__all__) == sorted(m._EXPORTS)
    for name in m._EXPORTS:
        assert getattr(m, name) is not None, name


def test_unknown_attr_raises() -> None:
    import pytest

    import quant_fund.models as m

    with pytest.raises(AttributeError):
        _ = m.DefinitelyNotAModel
