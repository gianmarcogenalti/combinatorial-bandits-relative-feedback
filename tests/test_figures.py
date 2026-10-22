"""Confidence intervals and their native PGFPlots export."""

from types import SimpleNamespace

import numpy as np
import pytest

from experiment_code.figures import PaperFigures


def summary(group="comparison", count=20, kind="standard_error"):
    return SimpleNamespace(
        labels={
            "group": group,
            "algorithm.name": "mnl_topk_uif" if group == "mnl" else "topk_uif",
            "data.params.m": 1,
            "data.params.theta0": 0.001,
        },
        x=np.array([1, 10]),
        mean=np.array([3.0, 5.0]),
        uncertainty=np.array([0.2, 0.3]),
        count=count,
        uncertainty_kind=kind,
    )


@pytest.mark.parametrize(
    "count,critical", [(3, 4.302652729696142), (20, 2.093024054408263)]
)
def test_student_interval_uses_actual_repetition_count(count, critical):
    half = PaperFigures()._half_width(summary(count=count))
    np.testing.assert_allclose(half, critical * np.array([0.2, 0.3]), rtol=1e-12)


def test_intervals_require_sem_and_two_independent_runs():
    with pytest.raises(ValueError, match="standard-error"):
        PaperFigures()._half_width(summary(kind="std"))
    assert np.isnan(PaperFigures()._half_width(summary(count=1))).all()
    with pytest.raises(ValueError, match="confidence_level"):
        PaperFigures(confidence_level=1)


@pytest.mark.parametrize("group", ["comparison", "feedback", "mnl"])
def test_export_preserves_means_and_uses_confidence_half_widths(group, tmp_path):
    value = summary(group=group)
    mean_before = value.mean.copy()
    sem_before = value.uncertainty.copy()
    [path] = PaperFigures().plot([value], None, tmp_path)
    text = path.read_text()
    assert "Pointwise 95% Student-t confidence intervals" in text
    if group == "comparison":
        assert "(1,2.58139518912)" in text  # 3 - t(.975,19)*.2
        assert "(10,5.62790721632)" in text  # 5 + t(.975,19)*.3
        assert "coordinates {(1,3) (10,5)}" in text
    else:
        assert "+- (0,0.627907216322)" in text
    np.testing.assert_array_equal(value.mean, mean_before)
    np.testing.assert_array_equal(value.uncertainty, sem_before)
