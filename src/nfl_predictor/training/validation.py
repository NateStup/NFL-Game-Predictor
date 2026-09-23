"""Walk-forward validation: expanding-window folds over seasons."""

import pandas as pd

from nfl_predictor.data.transforms import chronological_split, home_baseline_accuracy
from nfl_predictor.predictor.elo import derive_home_field_advantage
from nfl_predictor.predictor.features import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_feature_table,
)
from nfl_predictor.training.model import build_pipeline, evaluate_pipeline, train_pipeline


def generate_walk_forward_folds(
    seasons: list[int], min_train_seasons: int = 4
) -> list[tuple[list[int], int]]:
    """Expanding-window folds: train on every season before the test season.

    Seasons are ordered chronologically. The first fold trains on the first
    `min_train_seasons` seasons and tests on the next one; each later fold adds
    one season to training and tests on the season after it. Returns
    (train_seasons, test_season) pairs, oldest test season first.
    """
    if min_train_seasons < 1:
        raise ValueError(f"min_train_seasons must be >= 1, got {min_train_seasons}")
    if len(set(seasons)) != len(seasons):
        raise ValueError(f"seasons contains duplicate values: {seasons}")
    if len(seasons) <= min_train_seasons:
        raise ValueError(
            f"need at least {min_train_seasons + 1} seasons for one fold, "
            f"got {len(seasons)}"
        )

    ordered = sorted(seasons)
    return [
        (ordered[:i], ordered[i]) for i in range(min_train_seasons, len(ordered))
    ]


def score_walk_forward_fold(
    games_df: pd.DataFrame,
    train_seasons: list[int],
    test_season: int,
    *,
    k_factor: float = 20.0,
    window: int = 8,
    initial_rating: float = 1500.0,
) -> float:
    """Test-season accuracy of a logistic regression fit on train_seasons only.

    Only the fold's own seasons are used: games from any other season are
    discarded before features are built. The Elo home-field advantage is
    derived from the fold's training seasons, so nothing the fold could not
    have known at the time reaches training. Training rows without a full
    rolling window are dropped; test rows must all have one.
    """
    if test_season in train_seasons:
        raise ValueError(f"test_season {test_season} is also a training season")
    if max(train_seasons) > test_season:
        raise ValueError(
            f"test_season {test_season} must come after every training season, "
            f"got {sorted(train_seasons)}"
        )

    fold_games = games_df[games_df["season"].isin([*train_seasons, test_season])]
    train_games, test_games = chronological_split(fold_games, train_seasons, [test_season])
    if test_games.empty:
        raise ValueError(f"no games found for test_season {test_season}")

    hfa = derive_home_field_advantage(home_baseline_accuracy(train_games))
    table = build_feature_table(
        fold_games,
        home_field_advantage=hfa,
        window=window,
        k_factor=k_factor,
        initial_rating=initial_rating,
    )
    train, test = chronological_split(table, train_seasons, [test_season])

    if test[FEATURE_COLUMNS].isna().any(axis=None):
        raise ValueError(
            f"test_season {test_season} has rows without full rolling history "
            f"(window={window})"
        )
    train = train.dropna(subset=FEATURE_COLUMNS)

    pipeline = train_pipeline(
        build_pipeline(), train[FEATURE_COLUMNS], train[TARGET_COLUMN]
    )
    return evaluate_pipeline(pipeline, test[FEATURE_COLUMNS], test[TARGET_COLUMN])["accuracy"]
