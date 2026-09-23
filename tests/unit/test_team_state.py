"""Unit tests for the combined current team-state snapshot."""

import pandas as pd
import pytest

from nfl_predictor.predictor.elo import latest_team_elo_ratings
from nfl_predictor.predictor.rolling_stats import latest_team_rolling_stats
from nfl_predictor.predictor.team_state import get_team_state_snapshot

COLUMNS = ["gameday", "season", "home_team", "away_team", "home_score", "away_score"]

# 4 teams, 6 games: each pair meets once, so every team plays 3 games.
# Team D appears ONLY as an away team, to check a team isn't lost when it
# never hosts.
GAMES = [
    ("2024-09-08", 2024, "A", "B", 24, 17),
    ("2024-09-08", 2024, "C", "D", 10, 20),
    ("2024-09-15", 2024, "A", "C", 27, 13),
    ("2024-09-15", 2024, "B", "D", 21, 14),
    ("2024-09-22", 2024, "C", "B", 30, 3),
    ("2024-09-22", 2024, "A", "D", 7, 35),
]


@pytest.fixture
def games_df():
    return pd.DataFrame(GAMES, columns=COLUMNS)


def snapshot(df, **kwargs):
    return get_team_state_snapshot(df, home_field_advantage=30.0, **kwargs)


def test_exactly_one_row_per_unique_team(games_df):
    snap = snapshot(games_df, window=3)

    teams_in_input = set(games_df["home_team"]) | set(games_df["away_team"])
    assert len(snap) == len(teams_in_input) == 4
    assert snap["team"].is_unique
    assert set(snap["team"]) == teams_in_input
    assert list(snap["team"]) == ["A", "B", "C", "D"]


def test_columns(games_df):
    snap = snapshot(games_df, window=3)

    assert list(snap.columns) == [
        "team",
        "rolling_win_pct",
        "rolling_point_diff",
        "elo_rating",
    ]


def test_values_come_from_the_two_snapshot_functions(games_df):
    snap = snapshot(games_df, window=3, k_factor=32.0, initial_rating=1400.0)

    rolling = latest_team_rolling_stats(games_df, window=3)
    elo = latest_team_elo_ratings(
        games_df, initial_rating=1400.0, k_factor=32.0, home_field_advantage=30.0
    )
    pd.testing.assert_frame_equal(
        snap[["team", "rolling_win_pct", "rolling_point_diff"]], rolling
    )
    pd.testing.assert_frame_equal(snap[["team", "elo_rating"]], elo)


def test_short_history_keeps_team_rows_with_nan_rolling_and_real_elo(games_df):
    # 3 games per team < default window 8: rolling NaN, Elo still defined.
    snap = snapshot(games_df)

    assert len(snap) == 4
    assert snap[["rolling_win_pct", "rolling_point_diff"]].isna().all().all()
    assert snap["elo_rating"].notna().all()


def test_home_field_advantage_is_required(games_df):
    with pytest.raises(TypeError):
        get_team_state_snapshot(games_df)
