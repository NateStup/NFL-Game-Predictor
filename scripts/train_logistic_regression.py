"""Train and evaluate the scaled logistic regression; save the fitted pipeline.

Usage: python scripts/train_logistic_regression.py
"""

from pathlib import Path

import joblib

from build_features import build_feature_splits  # sibling script in scripts/
from nfl_predictor.data.transforms import home_baseline_accuracy
from nfl_predictor.predictor.features import FEATURE_COLUMNS, TARGET_COLUMN
from nfl_predictor.training.model import (
    build_pipeline,
    evaluate_pipeline,
    train_pipeline,
)

EXPECTED_BASELINE = 0.5331  # 2024 regular-season home win rate, 4 dp
ARTIFACT_PATH = Path(__file__).resolve().parents[1] / "models" / "logistic_regression.joblib"


def main() -> None:
    # (a) full feature pipeline
    splits = build_feature_splits()

    # (b) features / target
    X_train, y_train = splits.train[FEATURE_COLUMNS], splits.train[TARGET_COLUMN]
    X_test, y_test = splits.test[FEATURE_COLUMNS], splits.test[TARGET_COLUMN]
    print(f"Train: X {X_train.shape}, y {y_train.shape} | Test: X {X_test.shape}, y {y_test.shape}")
    print(f"HFA used in Elo (train-derived): {splits.hfa:.2f}")

    # (c) fit on train only
    pipeline = train_pipeline(build_pipeline(random_state=42), X_train, y_train)

    # (d) independent baseline recomputation, as a regression check
    baseline = home_baseline_accuracy(splits.test_games)
    if round(baseline, 4) != EXPECTED_BASELINE:
        raise RuntimeError(
            f"Test baseline changed: got {baseline:.6f}, expected {EXPECTED_BASELINE}"
        )
    if abs(y_test.mean() - baseline) > 1e-12:
        raise RuntimeError(
            f"home_win target mean {y_test.mean():.6f} != baseline {baseline:.6f}"
        )
    print(f"Test baseline (home team always wins): {baseline:.4f} [matches {EXPECTED_BASELINE}]")

    # (e) evaluate on test
    metrics = evaluate_pipeline(pipeline, X_test, y_test)
    (tn, fp), (fn, tp) = metrics["confusion_matrix"]
    print("Test metrics (positive class = home win):")
    print(f"  accuracy:  {metrics['accuracy']:.4f}")
    print(f"  precision: {metrics['precision']:.4f}")
    print(f"  recall:    {metrics['recall']:.4f}")
    print(f"  f1:        {metrics['f1']:.4f}")
    print("  confusion matrix (rows actual, cols predicted; [[TN, FP], [FN, TP]]):")
    print(f"    {metrics['confusion_matrix']}")
    print(f"    TN={tn} FP={fp} FN={fn} TP={tp}")

    coefs = pipeline.named_steps["classifier"].coef_[0]
    intercept = pipeline.named_steps["classifier"].intercept_[0]
    print("Coefficients (on standardized features):")
    for name, coef in zip(FEATURE_COLUMNS, coefs):
        print(f"  {name:22s} {coef:+.4f}")
    print(f"  {'intercept':22s} {intercept:+.4f}")

    # (f) comparison
    delta = metrics["accuracy"] - baseline
    print(f"Model accuracy: {metrics['accuracy']:.4f} vs baseline {baseline:.4f} ({delta:+.4f})")

    # (g) save artifact (models/ is gitignored)
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, ARTIFACT_PATH)
    print(f"Saved fitted pipeline to {ARTIFACT_PATH.relative_to(ARTIFACT_PATH.parents[1])}")


if __name__ == "__main__":
    main()
