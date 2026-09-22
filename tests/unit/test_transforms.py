"""Unit tests for the pure schedule transforms (no I/O, inline fixtures)."""

import pandas as pd
import pytest

from nfl_predictor.data.transforms import (
    chronological_split,
    drop_ties,
    home_baseline_accuracy,
    regular_season_only,
)


@pytest.fixture
def games():
    return pd.DataFrame(
        {
            "season": [2020, 2020, 2021, 2021, 2022, 2022],
            "home_team": ["KC", "BUF", "KC", "DAL", "PHI", "SF"],
            "away_team": ["BUF", "KC", "DAL", "KC", "SF", "PHI"],
            "home_score": [24, 17, 20, 13, 31, 10],
            "away_score": [20, 17, 27, 13, 7, 14],
        }
    )


# --- drop_ties ---------------------------------------------------------------


def test_drop_ties_removes_only_tied_games(games):
    result = drop_ties(games)

    assert len(result) == 4
    assert (result["home_score"] != result["away_score"]).all()


def test_drop_ties_keeps_non_tied_rows_unchanged(games):
    result = drop_ties(games)

    expected = games.iloc[[0, 2, 4, 5]]
    pd.testing.assert_frame_equal(result, expected)


def test_drop_ties_does_not_mutate_input(games):
    before = games.copy()

    drop_ties(games)

    pd.testing.assert_frame_equal(games, before)


# --- chronological_split -----------------------------------------------------


def test_chronological_split_partitions_by_season(games):
    train, test = chronological_split(
        games, train_seasons=[2020, 2021], test_seasons=[2022]
    )

    assert set(train["season"]) == {2020, 2021}
    assert set(test["season"]) == {2022}
    assert len(train) == 4
    assert len(test) == 2


def test_chronological_split_excludes_unlisted_seasons(games):
    train, test = chronological_split(games, train_seasons=[2020], test_seasons=[2022])

    assert 2021 not in set(train["season"]) | set(test["season"])
    assert len(train) + len(test) == 4


def test_chronological_split_raises_on_overlapping_seasons(games):
    with pytest.raises(ValueError, match="2021"):
        chronological_split(
            games, train_seasons=[2020, 2021], test_seasons=[2021, 2022]
        )


# --- home_baseline_accuracy --------------------------------------------------


def test_home_baseline_accuracy_is_fraction_of_home_wins():
    df = pd.DataFrame(
        {
            "home_score": [24, 20, 31, 10],
            "away_score": [20, 27, 7, 14],
        }
    )

    assert home_baseline_accuracy(df) == pytest.approx(0.5)


def test_home_baseline_accuracy_all_home_wins():
    df = pd.DataFrame({"home_score": [21, 30], "away_score": [3, 28]})

    assert home_baseline_accuracy(df) == pytest.approx(1.0)


def test_home_baseline_accuracy_no_home_wins():
    df = pd.DataFrame({"home_score": [3, 28], "away_score": [21, 30]})

    assert home_baseline_accuracy(df) == pytest.approx(0.0)


# --- regular_season_only -----------------------------------------------------


def test_regular_season_only_keeps_only_reg_games():
    df = pd.DataFrame(
        {
            "game_id": ["g1", "g2", "g3", "g4", "g5", "g6", "g7"],
            "game_type": ["REG", "WC", "REG", "DIV", "CON", "REG", "SB"],
        }
    )

    result = regular_season_only(df)

    assert list(result["game_id"]) == ["g1", "g3", "g6"]
    assert (result["game_type"] == "REG").all()


def test_regular_season_only_does_not_mutate_input():
    df = pd.DataFrame({"game_type": ["REG", "WC"]})
    before = df.copy()

    regular_season_only(df)

    pd.testing.assert_frame_equal(df, before)
