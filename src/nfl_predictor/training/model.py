"""Logistic-regression pipeline: build, train, evaluate."""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_pipeline(random_state: int = 42) -> Pipeline:
    """Unfitted StandardScaler -> LogisticRegression pipeline."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(random_state=random_state)),
        ]
    )


def train_pipeline(
    pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series
) -> Pipeline:
    """Fit the pipeline (in place) on training data only and return it.

    The scaler learns its mean/std from X_train alone, so no test-set
    statistics reach the model.
    """
    return pipeline.fit(X_train, y_train)


def evaluate_pipeline(
    fitted_pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series
) -> dict:
    """Score a fitted pipeline on held-out data.

    The positive class is 1 (home win). Returns floats for accuracy,
    precision, recall and f1 (precision/recall are 0.0 when undefined), and
    confusion_matrix as a 2x2 nested list of ints, always in this layout:

        rows = actual [0, 1], columns = predicted [0, 1]
        [[TN, FP],
         [FN, TP]]
    """
    y_pred = fitted_pipeline.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist(),
    }


def build_random_forest(random_state: int = 42):
    raise NotImplementedError
