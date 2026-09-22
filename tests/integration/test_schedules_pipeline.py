"""Integration test: real nfl_data_py fetch -> drop_ties -> chronological_split.

Hits the network. Deselect with: pytest -m "not integration"
"""

import pytest

from nfl_predictor.data.fetch import fetch_schedules
from nfl_predictor.data.transforms import chronological_split, drop_ties

pytestmark = pytest.mark.integration

TRAIN_SEASONS = list(range(2015, 2024))
TEST_SEASONS = [2024]
EXPECTED_COLUMNS = {
    "game_id",
    "season",
    "game_type",
    "week",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
}


@pytest.fixture(scope="module")
def schedules():
    return fetch_schedules(TRAIN_SEASONS + TEST_SEASONS)


@pytest.fixture(scope="module")
def split(schedules):
    return chronological_split(
        drop_ties(schedules), train_seasons=TRAIN_SEASONS, test_seasons=TEST_SEASONS
    )


def test_expected_columns_present(schedules):
    assert EXPECTED_COLUMNS <= set(schedules.columns)


def test_no_nulls_in_key_columns_after_tie_drop(schedules):
    no_ties = drop_ties(schedules)

    assert no_ties[["home_score", "away_score", "season"]].notna().all().all()


def test_split_row_counts_in_sane_range(split):
    # Schedules include playoffs (~11-13 games/season) as well as the regular
    # season (256 games/season through 2020, 272 from 2021).
    train, test = split

    assert 2400 <= len(train) <= 2500
    assert 275 <= len(test) <= 295


def test_no_season_overlap_on_real_data(split):
    train, test = split

    assert set(train["season"]) == set(TRAIN_SEASONS)
    assert set(test["season"]) == set(TEST_SEASONS)
    assert set(train["season"]).isdisjoint(set(test["season"]))
