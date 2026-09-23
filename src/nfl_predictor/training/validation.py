"""Walk-forward validation: expanding-window folds over seasons."""


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
