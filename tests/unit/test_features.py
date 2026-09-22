"""Unit tests for feature-table assembly (synthetic rotation fixture).

Same 3-team, 2-season rotation as test_rolling_stats: 15 games, 10 per team,
g0-g8 in 2023, g9-g14 in 2024. Rolling windows are full only for g12-g14.
"""

import pandas as pd
import pytest

from nfl_predictor.predictor.elo import compute_elo_ratings
from nfl_predictor.predictor.features import build_feature_table
from nfl_predictor.predictor.rolling_stats import rolling_point_diff, rolling_win_pct_diff

HFA = 23.0

GAMES = [
    # (game_id, gameday, season, home, away, home_score, away_score)
    ("g0", "2023-09-10", 2023, "A", "B", 24, 17),
    ("g1", "2023-09-17", 2023, "C", "A", 10, 20),
    ("g2", "2023-09-24", 2023, "B", "C", 21, 14),
    ("g3", "2023-10-01", 2023, "B", "A", 27, 13),
    ("g4", "2023-10-08", 2023, "A", "C", 17, 20),
    ("g5", "2023-10-15", 2023, "C", "B", 30, 10),
    ("g6", "2023-10-22", 2023, "A", "B", 28, 21),
    ("g7", "2023-10-29", 2023, "C", "A", 16, 23),
    ("g8", "2023-11-05", 2023, "B", "C", 24, 10),
    ("g9", "2024-09-08", 2024, "B", "A", 20, 17),
    ("g10", "2024-09-15", 2024, "A", "C", 31, 3),
    ("g11", "2024-09-22", 2024, "C", "B", 17, 14),
    ("g12", "2024-09-29", 2024, "A", "B", 21, 10),
    ("g13", "2024-10-06", 2024, "C", "A", 7, 35),
    ("g14", "2024-10-13", 2024, "B", "C", 27, 24),
]

ID_COLUMNS = ["game_id", "season", "gameday", "home_team", "away_team"]
FEATURE_COLUMNS = ["rolling_win_pct_diff", "rolling_point_diff", "elo_diff"]


@pytest.fixture
def games_df():
    return pd.DataFrame(
        GAMES,
        columns=[
            "game_id",
            "gameday",
            "season",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
        ],
    )


def build(df, **kwargs):
    return build_feature_table(df, home_field_advantage=HFA, **kwargs)


# --- (a) columns -------------------------------------------------------------


def test_output_has_id_feature_and_target_columns(games_df):
    table = build(games_df)

    assert list(table.columns) == ID_COLUMNS + FEATURE_COLUMNS + ["home_win"]


def test_scores_are_not_carried_into_feature_table(games_df):
    # The target is derived from the scores; carrying them forward would make
    # it trivial to leak the answer into a model's inputs.
    table = build(games_df)

    assert "home_score" not in table.columns
    assert "away_score" not in table.columns


# --- (b) warmup rows: rolling NaN, Elo defined -------------------------------


def test_warmup_rows_have_nan_rolling_features_but_defined_elo(games_df):
    table = build(games_df)
    warmup = table.iloc[:12]  # g0-g11: some team has < 8 prior games

    assert warmup["rolling_win_pct_diff"].isna().all()
    assert warmup["rolling_point_diff"].isna().all()
    assert warmup["elo_diff"].notna().all()


def test_full_history_rows_have_all_features(games_df):
    table = build(games_df)

    assert table.iloc[12:][FEATURE_COLUMNS].notna().all().all()


# --- (c) target --------------------------------------------------------------


def test_home_win_target_matches_known_results(games_df):
    table = build(games_df).set_index("game_id")

    # g0 A 24-17 home win; g1 C 10-20 home loss; g3 B 27-13 home win;
    # g4 A 17-20 home loss; g13 C 7-35 home loss.
    assert table.loc["g0", "home_win"] == 1
    assert table.loc["g1", "home_win"] == 0
    assert table.loc["g3", "home_win"] == 1
    assert table.loc["g4", "home_win"] == 0
    assert table.loc["g13", "home_win"] == 0
    assert set(table["home_win"]) == {0, 1}
    assert table["home_win"].dtype == "int64"


# --- (d) no rows dropped -----------------------------------------------------


def test_row_count_and_index_match_input(games_df):
    games_df.index = games_df.index + 100
    table = build(games_df)

    assert len(table) == len(games_df)
    assert list(table.index) == list(games_df.index)


# --- wiring ------------------------------------------------------------------


def test_feature_columns_match_underlying_functions(games_df):
    table = build(games_df, window=8, k_factor=20.0, initial_rating=1500.0)

    pd.testing.assert_series_equal(
        table["rolling_win_pct_diff"], rolling_win_pct_diff(games_df, window=8)
    )
    pd.testing.assert_series_equal(
        table["rolling_point_diff"], rolling_point_diff(games_df, window=8)
    )
    pd.testing.assert_series_equal(
        table["elo_diff"],
        compute_elo_ratings(
            games_df, initial_rating=1500.0, k_factor=20.0, home_field_advantage=HFA
        ),
    )


def test_parameters_are_passed_through(games_df):
    base = build(games_df)

    # A shorter window fills rolling features earlier.
    assert build(games_df, window=2)["rolling_point_diff"].notna().sum() > (
        base["rolling_point_diff"].notna().sum()
    )
    # HFA and K change the Elo updates (g0 is 0 either way: both teams new).
    other_hfa = build_feature_table(games_df, home_field_advantage=0.0)
    assert not other_hfa["elo_diff"].iloc[1:].equals(base["elo_diff"].iloc[1:])
    assert not build(games_df, k_factor=40.0)["elo_diff"].equals(base["elo_diff"])


def test_home_field_advantage_is_required(games_df):
    with pytest.raises(TypeError):
        build_feature_table(games_df)
