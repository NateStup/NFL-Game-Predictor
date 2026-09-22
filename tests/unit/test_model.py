"""Unit tests for the logistic-regression training module (synthetic data)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nfl_predictor.training.model import (
    build_pipeline,
    build_random_forest,
    evaluate_pipeline,
    train_pipeline,
)

FEATURES = ["rolling_win_pct_diff", "rolling_point_diff", "elo_diff"]

# Training fixture: every column has three rows at (mean - s) and three at
# (mean + s), so the population mean and std (ddof=0, as StandardScaler uses)
# are exact by construction. Home wins exactly when elo/point diffs are high,
# so the pattern is linearly separable.
#
#   rolling_win_pct_diff: {-0.2 x3, 0.4 x3}  -> mean 0.1,  std 0.3
#   rolling_point_diff:   {-3 x3,   5 x3}    -> mean 1.0,  std 4.0
#   elo_diff:             {-10 x3,  30 x3}   -> mean 10.0, std 20.0
# e.g. win_pct: mean = (3*-0.2 + 3*0.4) / 6 = 0.6 / 6 = 0.1
#               var  = (3*0.3^2 + 3*0.3^2) / 6 = 0.09 -> std 0.3
TRAIN_MEAN = [0.1, 1.0, 10.0]
TRAIN_STD = [0.3, 4.0, 20.0]


@pytest.fixture
def X_train():
    return pd.DataFrame(
        {
            "rolling_win_pct_diff": [0.4, -0.2, -0.2, 0.4, -0.2, 0.4],
            "rolling_point_diff": [-3, -3, -3, 5, 5, 5],
            "elo_diff": [-10, -10, -10, 30, 30, 30],
        },
        dtype=float,
    )


@pytest.fixture
def y_train():
    return pd.Series([0, 0, 0, 1, 1, 1], name="home_win")


# --- build_pipeline ----------------------------------------------------------


def test_build_pipeline_has_named_scaler_and_classifier_steps():
    pipeline = build_pipeline()

    assert isinstance(pipeline, Pipeline)
    assert list(pipeline.named_steps) == ["scaler", "classifier"]
    assert isinstance(pipeline.named_steps["scaler"], StandardScaler)
    assert isinstance(pipeline.named_steps["classifier"], LogisticRegression)


def test_build_pipeline_passes_random_state_to_classifier():
    assert build_pipeline().named_steps["classifier"].random_state == 42
    assert build_pipeline(random_state=7).named_steps["classifier"].random_state == 7


def test_build_pipeline_is_unfitted(X_train):
    with pytest.raises(NotFittedError):
        build_pipeline().predict(X_train)


def test_build_pipeline_returns_fresh_instances():
    assert build_pipeline() is not build_pipeline()


# --- train_pipeline: scaler fitted on X_train only (leakage guard) -----------


def test_train_pipeline_fits_scaler_on_train_data_only(X_train, y_train):
    fitted = train_pipeline(build_pipeline(), X_train, y_train)

    scaler = fitted.named_steps["scaler"]
    np.testing.assert_allclose(scaler.mean_, TRAIN_MEAN)
    np.testing.assert_allclose(scaler.scale_, TRAIN_STD)


def test_train_pipeline_scaler_does_not_see_the_full_dataset(X_train, y_train):
    # A larger "full dataset" = train + four held-out rows far from the train
    # distribution. Only the train slice is passed to train_pipeline.
    held_out = pd.DataFrame(
        {
            "rolling_win_pct_diff": [0.9, 0.9, -0.9, -0.9],
            "rolling_point_diff": [20, 20, -20, -20],
            "elo_diff": [200, 200, 100, 100],
        },
        dtype=float,
    )
    full = pd.concat([X_train, held_out], ignore_index=True)

    fitted = train_pipeline(build_pipeline(), X_train, y_train)

    scaler_mean = fitted.named_steps["scaler"].mean_
    assert not np.allclose(scaler_mean, full.mean().to_numpy())
    np.testing.assert_allclose(scaler_mean, TRAIN_MEAN)
    # Control: the full-data means really are different (elo 10 vs 66).
    assert full["elo_diff"].mean() == pytest.approx(66.0)


def test_train_pipeline_returns_a_fitted_pipeline(X_train, y_train):
    fitted = train_pipeline(build_pipeline(), X_train, y_train)

    assert isinstance(fitted, Pipeline)
    assert list(fitted.predict(X_train)) == list(y_train)


# --- evaluate_pipeline -------------------------------------------------------

# Test rows are either far on the "home win" side of the training pattern
# (all features high) or far on the "home loss" side (all low), so the fitted
# model's predictions are unambiguous. Actual outcomes are chosen to give:
#
#   row  features  predicted  actual  cell
#   t0   high      1          1       TP
#   t1   high      1          0       FP
#   t2   low       0          0       TN
#   t3   low       0          1       FN
#   t4   high      1          1       TP
#
# TP=2, FP=1, TN=1, FN=1
# accuracy  = (TP + TN) / 5 = 3/5 = 0.6
# precision = TP / (TP + FP) = 2/3
# recall    = TP / (TP + FN) = 2/3
# f1        = 2PR / (P + R) = 2 * (4/9) / (4/3) = 2/3
# confusion_matrix (rows actual [0, 1], cols predicted [0, 1]):
#   [[TN, FP],   = [[1, 1],
#    [FN, TP]]      [1, 2]]
HIGH = (0.6, 8.0, 50.0)
LOW = (-0.5, -6.0, -40.0)


@pytest.fixture
def X_test():
    return pd.DataFrame([HIGH, HIGH, LOW, LOW, HIGH], columns=FEATURES)


@pytest.fixture
def y_test():
    return pd.Series([1, 0, 0, 1, 1], name="home_win")


@pytest.fixture
def fitted(X_train, y_train):
    return train_pipeline(build_pipeline(), X_train, y_train)


def test_evaluate_fixture_premise_predictions(fitted, X_test):
    # Guards the hand calculation below: it assumes these predictions.
    assert list(fitted.predict(X_test)) == [1, 1, 0, 0, 1]


def test_evaluate_pipeline_returns_expected_keys(fitted, X_test, y_test):
    result = evaluate_pipeline(fitted, X_test, y_test)

    assert set(result) == {"accuracy", "precision", "recall", "f1", "confusion_matrix"}


def test_evaluate_pipeline_matches_hand_computed_metrics(fitted, X_test, y_test):
    result = evaluate_pipeline(fitted, X_test, y_test)

    assert result["accuracy"] == pytest.approx(0.6)
    assert result["precision"] == pytest.approx(2 / 3)
    assert result["recall"] == pytest.approx(2 / 3)
    assert result["f1"] == pytest.approx(2 / 3)
    assert result["confusion_matrix"] == [[1, 1], [1, 2]]


def test_evaluate_confusion_matrix_is_2x2_even_if_one_class_absent(fitted):
    # All-HIGH rows, all actually home wins: only TP cells are populated.
    X = pd.DataFrame([HIGH, HIGH], columns=FEATURES)
    y = pd.Series([1, 1])

    result = evaluate_pipeline(fitted, X, y)

    assert result["confusion_matrix"] == [[0, 0], [0, 2]]


# --- build_random_forest -----------------------------------------------------


def test_build_random_forest_is_a_random_forest_not_a_pipeline():
    model = build_random_forest()

    assert isinstance(model, RandomForestClassifier)
    # No scaler: trees split on raw thresholds, so there is no Pipeline wrapper.
    assert not isinstance(model, Pipeline)


def test_build_random_forest_uses_100_trees_and_random_state():
    assert build_random_forest().n_estimators == 100
    assert build_random_forest().random_state == 42
    assert build_random_forest(random_state=7).random_state == 7


def test_build_random_forest_leaves_all_other_params_at_defaults():
    # Deliberately unconstrained (max_depth=None, min_samples_leaf=1, ...) so
    # any overfitting is visible before choosing constraints.
    expected = RandomForestClassifier(n_estimators=100, random_state=42).get_params()

    assert build_random_forest().get_params() == expected


def test_build_random_forest_is_unfitted(X_train):
    with pytest.raises(NotFittedError):
        build_random_forest().predict(X_train)
