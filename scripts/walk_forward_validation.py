"""Walk-forward validation of the logistic regression over the training era.

For each season N+1 in 2019-2023: train on 2015..N, test on N+1. Only the
training-era seasons are downloaded; the held-out test season is never loaded,
so this estimate cannot depend on how predictable that season was.

Usage: python scripts/walk_forward_validation.py
"""

import statistics

from build_features import (  # sibling script in scripts/
    INITIAL_RATING,
    K_FACTOR,
    TRAIN_SEASONS,
    WINDOW,
)
from nfl_predictor.data.fetch import fetch_schedules
from nfl_predictor.data.transforms import (
    drop_ties,
    home_baseline_accuracy,
    normalize_franchises,
    regular_season_only,
)
from nfl_predictor.training.validation import (
    generate_walk_forward_folds,
    score_walk_forward_fold,
)

MIN_TRAIN_SEASONS = 4


def main() -> None:
    # (a) training-era games only, processed exactly as in build_features
    games = drop_ties(regular_season_only(normalize_franchises(fetch_schedules(TRAIN_SEASONS))))
    games = games.sort_values("gameday", kind="mergesort")
    loaded = sorted(games["season"].unique())
    if loaded != TRAIN_SEASONS:
        raise RuntimeError(f"expected seasons {TRAIN_SEASONS}, loaded {loaded}")
    print(f"Seasons loaded:   {loaded[0]}-{loaded[-1]} ({len(games)} regular-season, tie-free games)")

    # (b) score each fold
    folds = generate_walk_forward_folds(TRAIN_SEASONS, min_train_seasons=MIN_TRAIN_SEASONS)
    accuracies, baselines = [], []
    print(f"{'train':>9}  {'test':>4}  {'games':>5}  {'baseline':>8}  {'accuracy':>8}  {'vs base':>7}")
    for train_seasons, test_season in folds:
        accuracy = score_walk_forward_fold(
            games,
            train_seasons,
            test_season,
            k_factor=K_FACTOR,
            window=WINDOW,
            initial_rating=INITIAL_RATING,
        )
        test_games = games[games["season"] == test_season]
        baseline = home_baseline_accuracy(test_games)
        accuracies.append(accuracy)
        baselines.append(baseline)
        print(
            f"{train_seasons[0]}-{train_seasons[-1]}  {test_season}  {len(test_games):>5}  "
            f"{baseline:>8.4f}  {accuracy:>8.4f}  {accuracy - baseline:>+7.4f}"
        )

    # (c) summary; std is the sample std (ddof=1) across folds
    mean_acc, std_acc = statistics.mean(accuracies), statistics.stdev(accuracies)
    mean_base, std_base = statistics.mean(baselines), statistics.stdev(baselines)
    print(f"Folds: {len(folds)}")
    print(f"Accuracy:           mean {mean_acc:.4f} +/- {std_acc:.4f} (sample std)")
    print(f"Home baseline:      mean {mean_base:.4f} +/- {std_base:.4f} (sample std)")
    print(f"Mean margin over baseline: {mean_acc - mean_base:+.4f}")
    print(f"Folds beating baseline:    {sum(a > b for a, b in zip(accuracies, baselines))}/{len(folds)}")


if __name__ == "__main__":
    main()
