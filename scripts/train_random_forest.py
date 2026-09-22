"""Train and evaluate an unconstrained random forest; save the fitted model.

Uses the exact feature splits as train_logistic_regression.py, so results
are directly comparable.

Usage: python scripts/train_random_forest.py
"""

from pathlib import Path

import joblib

from build_features import build_feature_splits  # sibling script in scripts/
from nfl_predictor.data.transforms import home_baseline_accuracy
from nfl_predictor.predictor.features import FEATURE_COLUMNS, TARGET_COLUMN
from nfl_predictor.training.model import (
    build_random_forest,
    evaluate_pipeline,
    train_pipeline,
)

EXPECTED_BASELINE = 0.5331  # 2024 regular-season home win rate, 4 dp
LOGREG_TEST_ACCURACY = 0.6875  # from prior run, for reference
ARTIFACT_PATH = Path(__file__).resolve().parents[1] / "models" / "random_forest.joblib"


def print_metrics(label: str, metrics: dict) -> None:
    (tn, fp), (fn, tp) = metrics["confusion_matrix"]
    print(f"{label} metrics (positive class = home win):")
    print(f"  accuracy:  {metrics['accuracy']:.4f}")
    print(f"  precision: {metrics['precision']:.4f}")
    print(f"  recall:    {metrics['recall']:.4f}")
    print(f"  f1:        {metrics['f1']:.4f}")
    print(f"  confusion matrix [[TN, FP], [FN, TP]]: {metrics['confusion_matrix']}")
    print(f"    TN={tn} FP={fp} FN={fn} TP={tp}")


def main() -> None:
    # (a) same feature splits as the logistic regression run
    splits = build_feature_splits()
    X_train, y_train = splits.train[FEATURE_COLUMNS], splits.train[TARGET_COLUMN]
    X_test, y_test = splits.test[FEATURE_COLUMNS], splits.test[TARGET_COLUMN]
    print(f"Train: X {X_train.shape}, y {y_train.shape} | Test: X {X_test.shape}, y {y_test.shape}")
    print(f"HFA used in Elo (train-derived): {splits.hfa:.2f}")

    # (b) fit on train only
    model = train_pipeline(build_random_forest(random_state=42), X_train, y_train)

    # (c) baseline regression guard
    baseline = home_baseline_accuracy(splits.test_games)
    if round(baseline, 4) != EXPECTED_BASELINE:
        raise RuntimeError(
            f"Test baseline changed: got {baseline:.6f}, expected {EXPECTED_BASELINE}"
        )
    print(f"Test baseline (home team always wins): {baseline:.4f} [matches {EXPECTED_BASELINE}]")

    # (d) train AND test metrics: the overfitting diagnostic
    train_metrics = evaluate_pipeline(model, X_train, y_train)
    test_metrics = evaluate_pipeline(model, X_test, y_test)
    print_metrics("Train", train_metrics)
    print_metrics("Test", test_metrics)

    # (e) comparison block
    print("Accuracy comparison (2024 test set unless noted):")
    print(f"  Baseline (always home):            {baseline:.4f}")
    print(f"  Logistic regression test:          {LOGREG_TEST_ACCURACY:.4f}  (from prior run, for reference)")
    print(f"  Random forest train (2015-2023):   {train_metrics['accuracy']:.4f}")
    print(f"  Random forest test:                {test_metrics['accuracy']:.4f}")
    print(f"  RF train - test gap:               {train_metrics['accuracy'] - test_metrics['accuracy']:+.4f}")
    print(f"  RF test vs baseline:               {test_metrics['accuracy'] - baseline:+.4f}")
    print(f"  RF test vs logistic regression:    {test_metrics['accuracy'] - LOGREG_TEST_ACCURACY:+.4f}")

    # (f) feature importances
    print("Feature importances (mean decrease in impurity):")
    for name, importance in zip(FEATURE_COLUMNS, model.feature_importances_):
        print(f"  {name:22s} {importance:.4f}")

    # (g) save artifact (models/ is gitignored)
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, ARTIFACT_PATH)
    print(f"Saved fitted model to {ARTIFACT_PATH.relative_to(ARTIFACT_PATH.parents[1])}")


if __name__ == "__main__":
    main()
