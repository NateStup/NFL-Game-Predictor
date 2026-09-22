"""Integration test: real fetch -> regular_season_only -> drop_ties -> split.

Hits the network. Deselect with: pytest -m "not integration"
"""

import pytest

from nfl_predictor.data.fetch import fetch_schedules
from nfl_predictor.data.transforms import (
    chronological_split,
    drop_ties,
    normalize_franchises,
    regular_season_only,
)

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

CURRENT_TEAMS = [
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE",
    "DAL", "DEN", "DET", "GB", "HOU", "IND", "JAX", "KC",
    "LA", "LAC", "LV", "MIA", "MIN", "NE", "NO", "NYG",
    "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS",
]  # fmt: skip


@pytest.fixture(scope="module")
def schedules():
    return fetch_schedules(TRAIN_SEASONS + TEST_SEASONS)


@pytest.fixture(scope="module")
def split(schedules):
    return chronological_split(
        drop_ties(regular_season_only(schedules)),
        train_seasons=TRAIN_SEASONS,
        test_seasons=TEST_SEASONS,
    )


def test_expected_columns_present(schedules):
    assert EXPECTED_COLUMNS <= set(schedules.columns)


def test_no_nulls_in_key_columns_after_tie_drop(schedules):
    no_ties = drop_ties(schedules)

    assert no_ties[["home_score", "away_score", "season"]].notna().all().all()


def test_split_row_counts_in_sane_range(split):
    # Regular season: 256 games/season through 2020, 272 from 2021.
    train, test = split

    assert 2200 <= len(train) <= 2400
    assert 260 <= len(test) <= 280


def test_no_playoff_games_remain(split):
    train, test = split

    assert set(train["game_type"]) == {"REG"}
    assert set(test["game_type"]) == {"REG"}


def test_no_season_overlap_on_real_data(split):
    train, test = split

    assert set(train["season"]) == set(TRAIN_SEASONS)
    assert set(test["season"]) == set(TEST_SEASONS)
    assert set(train["season"]).isdisjoint(set(test["season"]))


def test_processed_data_has_exactly_the_32_current_franchises(schedules):
    processed = drop_ties(regular_season_only(normalize_franchises(schedules)))

    teams = sorted(set(processed["home_team"]) | set(processed["away_team"]))
    assert teams == CURRENT_TEAMS
