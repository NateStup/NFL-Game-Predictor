"""Unit tests for walk-forward validation (fold generation)."""

import pytest

from nfl_predictor.training.validation import generate_walk_forward_folds

# --- generate_walk_forward_folds ---------------------------------------------


def test_folds_expand_training_window_one_season_at_a_time():
    # By hand: with 2 seasons minimum, the first test season is the 3rd.
    folds = generate_walk_forward_folds([1, 2, 3, 4, 5], min_train_seasons=2)

    assert folds == [
        ([1, 2], 3),
        ([1, 2, 3], 4),
        ([1, 2, 3, 4], 5),
    ]


def test_default_on_training_era_gives_five_folds_2019_to_2023():
    folds = generate_walk_forward_folds(list(range(2015, 2024)))

    assert folds == [
        ([2015, 2016, 2017, 2018], 2019),
        ([2015, 2016, 2017, 2018, 2019], 2020),
        ([2015, 2016, 2017, 2018, 2019, 2020], 2021),
        ([2015, 2016, 2017, 2018, 2019, 2020, 2021], 2022),
        ([2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022], 2023),
    ]


def test_every_training_season_precedes_its_test_season():
    for train_seasons, test_season in generate_walk_forward_folds(list(range(2015, 2024))):
        assert max(train_seasons) < test_season


def test_exactly_enough_seasons_gives_one_fold():
    assert generate_walk_forward_folds([1, 2, 3], min_train_seasons=2) == [([1, 2], 3)]


def test_unsorted_input_is_ordered_chronologically():
    folds = generate_walk_forward_folds([3, 1, 2], min_train_seasons=2)

    assert folds == [([1, 2], 3)]


def test_does_not_mutate_input():
    seasons = [3, 1, 2]
    generate_walk_forward_folds(seasons, min_train_seasons=2)

    assert seasons == [3, 1, 2]


def test_too_few_seasons_raises():
    with pytest.raises(ValueError, match="at least"):
        generate_walk_forward_folds([1, 2], min_train_seasons=2)


def test_duplicate_seasons_raise():
    with pytest.raises(ValueError, match="duplicate"):
        generate_walk_forward_folds([1, 2, 2, 3], min_train_seasons=2)


def test_min_train_seasons_below_one_raises():
    with pytest.raises(ValueError, match="min_train_seasons"):
        generate_walk_forward_folds([1, 2, 3], min_train_seasons=0)
