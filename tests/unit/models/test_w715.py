from quant_fund.models.auslander_reiten import (
    bench_auslander_reiten,
)
from quant_fund.models.cluster_algebra import (
    bench_cluster_algebra,
)
from quant_fund.models.cluster_category import (
    bench_cluster_category,
)
from quant_fund.models.quiver_mutation import (
    bench_quiver_mutation,
)
from quant_fund.models.silting_object import (
    bench_silting_object,
)
from quant_fund.models.tilting_object import (
    bench_tilting_object,
)


def test_cluster_algebra():
    assert bench_cluster_algebra()["synthetic_cluster_algebra"] == 1.0


def test_quiver_mutation():
    assert bench_quiver_mutation()["synthetic_quiver_mutation"] == 1.0


def test_tilting_object():
    assert bench_tilting_object()["synthetic_tilting_object"] == 1.0


def test_auslander_reiten():
    assert bench_auslander_reiten()["synthetic_auslander_reiten"] == 1.0


def test_cluster_category():
    assert bench_cluster_category()["synthetic_cluster_category"] == 1.0


def test_silting_object():
    assert bench_silting_object()["synthetic_silting_object"] == 1.0
