"""Tests for the remember_data config of BaseDetector."""

__author__ = ["Deep-Axe"]

import warnings

import numpy as np
import pandas as pd
import pytest

from sktime.detection.base import BaseDetector
from sktime.tests.test_switch import run_test_module_changed


class _RecordingDetector(BaseDetector):
    """Detector with default _update, records calls of _fit."""

    _tags = {
        "task": "anomaly_detection",
        "learning_type": "unsupervised",
        "fit_is_empty": False,
    }

    def __init__(self):
        super().__init__()

    def _fit(self, X, y=None):
        self.n_fit_calls_ = getattr(self, "n_fit_calls_", 0) + 1
        self.n_fit_timepoints_ = len(X)
        return self

    def _predict(self, X):
        return pd.DataFrame({"ilocs": pd.Series([], dtype=int)})


def _make_X(n_timepoints=20):
    return pd.DataFrame({"a": np.arange(n_timepoints, dtype=float)})


def _remember_data_warnings(fun):
    """Run fun and return the FutureWarnings about remember_data it raised."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        fun()
    return [
        x
        for x in w
        if issubclass(x.category, FutureWarning) and "remember_data" in str(x.message)
    ]


@pytest.mark.skipif(
    not run_test_module_changed("sktime.detection"),
    reason="module not changed",
)
def test_remember_data_true_stores_and_refits():
    """Test that with remember_data=True, data is stored, pooled and refit to."""
    X = _make_X()
    detector = _RecordingDetector().set_config(remember_data=True)
    detector.fit(X.iloc[:10])

    assert len(detector._X) == 10

    detector.update(X.iloc[10:15])

    assert len(detector._X) == 15
    assert detector.n_fit_calls_ == 2
    assert detector.n_fit_timepoints_ == 15


@pytest.mark.skipif(
    not run_test_module_changed("sktime.detection"),
    reason="module not changed",
)
def test_remember_data_false_stores_nothing_and_does_not_refit():
    """Test that with remember_data=False, no data is stored and update warns."""
    X = _make_X()
    detector = _RecordingDetector().set_config(remember_data=False)
    detector.fit(X.iloc[:10])

    assert detector._X is None
    assert detector._y is None

    with pytest.warns(UserWarning, match="remember_data=False"):
        detector.update(X.iloc[10:15])

    assert detector._X is None
    assert detector._y is None
    assert detector.n_fit_calls_ == 1
    assert detector.n_fit_timepoints_ == 10


@pytest.mark.skipif(
    not run_test_module_changed("sktime.detection"),
    reason="module not changed",
)
def test_remember_data_false_fit_removes_stale_data():
    """Test that fit with remember_data=False removes data from an earlier fit."""
    X = _make_X()
    detector = _RecordingDetector().fit(X.iloc[:10])
    assert detector._X is not None

    detector.set_config(remember_data=False)
    detector.fit(X.iloc[10:20])

    assert detector._X is None
    assert detector._y is None


@pytest.mark.skipif(
    not run_test_module_changed("sktime.detection"),
    reason="module not changed",
)
def test_remember_data_switched_on_after_fit():
    """Test that update works if remember_data is set to True after fit."""
    X = _make_X()
    detector = _RecordingDetector().set_config(remember_data=False)
    detector.fit(X.iloc[:10])

    detector.set_config(remember_data=True)
    detector.update(X.iloc[10:15])

    # only data seen after the config change can be stored
    assert len(detector._X) == 5


@pytest.mark.skipif(
    not run_test_module_changed("sktime.detection"),
    reason="module not changed",
)
def test_remember_data_false_native_update():
    """Test that a native _update still receives every batch with False."""
    from sktime.detection.dummy import DummyPretrainedAnomalies

    X = _make_X()
    detector = DummyPretrainedAnomalies.create_test_instance()
    detector.set_config(remember_data=False)
    detector.fit(X.iloc[:6])

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        detector.update(X.iloc[6:12])

    assert detector.n_timepoints_seen_ == 12
    assert detector._X is None
    assert not any("remember_data=False" in str(x.message) for x in w)


@pytest.mark.skipif(
    not run_test_module_changed("sktime.detection"),
    reason="module not changed",
)
def test_remember_data_fit_is_empty_unaffected():
    """Test that detectors with empty fit neither store data nor warn."""
    from sktime.detection.naive import ThresholdDetector

    X = _make_X()
    detector = ThresholdDetector()
    assert detector.get_tag("fit_is_empty")

    future_warnings = _remember_data_warnings(
        lambda: detector.fit(X.iloc[:10]).update(X.iloc[10:15])
    )

    assert future_warnings == []
    assert not hasattr(detector, "_X")


@pytest.mark.skipif(
    not run_test_module_changed("sktime.detection"),
    reason="module not changed",
)
def test_remember_data_future_warning():
    """Test the deprecation warning is raised in fit if data is stored only."""
    X = _make_X()

    # no warning at construction, or on clone
    assert _remember_data_warnings(lambda: _RecordingDetector().clone()) == []

    # warning in fit with the default
    assert len(_remember_data_warnings(lambda: _RecordingDetector().fit(X))) == 1

    # no warning in fit after adopting the new default
    detector = _RecordingDetector().set_config(remember_data=False)
    assert _remember_data_warnings(lambda: detector.fit(X)) == []
    assert _remember_data_warnings(lambda: detector.clone().fit(X)) == []


@pytest.mark.skipif(
    not run_test_module_changed("sktime.detection"),
    reason="module not changed",
)
def test_remember_data_set_config_doc():
    """Test that set_config of detectors documents remember_data."""
    from sktime.detection.dummy import DummyRegularAnomalies

    assert "remember_data" in BaseDetector.set_config.__doc__
    assert "remember_data" in DummyRegularAnomalies.set_config.__doc__


@pytest.mark.skipif(
    not run_test_module_changed("sktime.detection"),
    reason="module not changed",
)
@pytest.mark.parametrize("child_remember_data", [True, False])
def test_remember_data_detector_pipeline(child_remember_data):
    """Test that remember_data applies to the pipeline only, not to components."""
    from sktime.detection.compose import DetectorPipeline
    from sktime.transformations.exponent import ExponentTransformer

    X = _make_X()
    child = _RecordingDetector().set_config(remember_data=child_remember_data)
    pipe = DetectorPipeline([ExponentTransformer(), child])
    pipe.set_config(remember_data=False)
    pipe.fit(X)

    fitted_child = pipe.steps_[-1][1]

    assert pipe._X is None
    assert (fitted_child._X is not None) == child_remember_data
