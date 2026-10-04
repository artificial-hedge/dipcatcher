"""Wave-1083 humanities-theory canon tests."""

from __future__ import annotations

from quant_fund.models.hermeneutics import bench_hermeneutics
from quant_fund.models.narratology import bench_narratology
from quant_fund.models.phenomenology import bench_phenomenology
from quant_fund.models.poststructuralism import bench_poststructuralism
from quant_fund.models.semiotics import bench_semiotics
from quant_fund.models.structuralism import bench_structuralism


def test_semiotics():
    assert bench_semiotics()["synthetic_semiotics"] == 1.0


def test_narratology():
    assert bench_narratology()["synthetic_narratology"] == 1.0


def test_hermeneutics():
    assert bench_hermeneutics()["synthetic_hermeneutics"] == 1.0


def test_phenomenology():
    assert bench_phenomenology()["synthetic_phenomenology"] == 1.0


def test_structuralism():
    assert bench_structuralism()["synthetic_structuralism"] == 1.0


def test_poststructuralism():
    assert bench_poststructuralism()["synthetic_poststructuralism"] == 1.0
