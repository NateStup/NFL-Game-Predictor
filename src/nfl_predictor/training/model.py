"""Logistic-regression pipeline: build, train, evaluate."""

import pandas as pd
from sklearn.pipeline import Pipeline


def build_pipeline(random_state: int = 42) -> Pipeline:
    raise NotImplementedError


def train_pipeline(
    pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series
) -> Pipeline:
    raise NotImplementedError


def evaluate_pipeline(
    fitted_pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series
) -> dict:
    raise NotImplementedError
