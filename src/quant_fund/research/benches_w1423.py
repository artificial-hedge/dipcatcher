"""Wave-1423 bench adapters: education canon (SYNTHETIC only)."""

from quant_fund.models import (
    class_qa_studies,
    course_qa_studies,
    exam_qa_studies,
    homework_qa_studies,
    lecture_qa_studies,
    seminar_qa_studies,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 14230


def _finite_blob(blob):
    if not (isinstance(blob, dict) and blob):
        raise ValueError("bench blob must be a non-empty dict")
    for k, v in blob.items():
        if not k.startswith("synthetic_"):
            raise ValueError(f"non-synthetic metric key {k}")
        if k in _FORBIDDEN:
            raise ValueError(f"forbidden metric key {k}")
        if not (isinstance(v, float) and 0.0 <= v <= 1.0):
            raise ValueError(f"metric {k} is not a [0,1] float")
    return blob


def _floats(blob):
    return sorted(v for _, v in _finite_blob(blob).items())


def bench_class_qa_studies_family(seed: int = _SEED + 0):
    """class_qa_studies: synthetic correctness bench."""
    return _finite_blob(class_qa_studies.bench_class_qa_studies(seed))


def bench_course_qa_studies_family(seed: int = _SEED + 1):
    """course_qa_studies: synthetic correctness bench."""
    return _finite_blob(course_qa_studies.bench_course_qa_studies(seed))


def bench_exam_qa_studies_family(seed: int = _SEED + 2):
    """exam_qa_studies: synthetic correctness bench."""
    return _finite_blob(exam_qa_studies.bench_exam_qa_studies(seed))


def bench_homework_qa_studies_family(seed: int = _SEED + 3):
    """homework_qa_studies: synthetic correctness bench."""
    return _finite_blob(homework_qa_studies.bench_homework_qa_studies(seed))


def bench_lecture_qa_studies_family(seed: int = _SEED + 4):
    """lecture_qa_studies: synthetic correctness bench."""
    return _finite_blob(lecture_qa_studies.bench_lecture_qa_studies(seed))


def bench_seminar_qa_studies_family(seed: int = _SEED + 5):
    """seminar_qa_studies: synthetic correctness bench."""
    return _finite_blob(seminar_qa_studies.bench_seminar_qa_studies(seed))
