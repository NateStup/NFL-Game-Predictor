"""Unit tests for walk-forward validation (fold generation, fold scoring)."""

import pandas as pd
import pytest

from nfl_predictor.predictor.elo import derive_home_field_advantage
from nfl_predictor.training import validation
from nfl_predictor.training.validation import (
    generate_walk_forward_folds,
    score_walk_forward_fold,
)

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


# --- score_walk_forward_fold -------------------------------------------------
#
# Synthetic league: 4 teams in a strict hierarchy (A > B > C > D). The
# higher-ranked team always wins, by 7 points per rank of difference, so the
# outcome is fully predictable from past form. Each season is a double round
# robin (12 games, 6 per team, one game per date): in round 1 the
# lower-ranked team hosts (home loses), in round 2 the higher-ranked team
# hosts (home wins), so both target classes appear and the home win rate is
# 6/12 = 0.5 unless games are flipped.

RANK = {"A": 0, "B": 1, "C": 2, "D": 3}
PAIRINGS = [("A", "B"), ("C", "D"), ("A", "C"), ("B", "D"), ("A", "D"), ("B", "C")]
SEASON_COLUMNS = [
    "game_id", "season", "gameday", "home_team", "away_team", "home_score", "away_score"
]


def make_season(season, flip_first=0, reverse_hierarchy=False):
    """One season of games. flip_first moves the first N round-1 games to the
    stronger team's home field, raising the home win rate to (6 + N) / 12.
    reverse_hierarchy makes the weakest team win instead."""
    rows = []
    start = pd.Timestamp(f"{season}-09-01")
    for round_no in (1, 2):
        for i, (strong, weak) in enumerate(PAIRINGS):
            if reverse_hierarchy:
                strong, weak = weak, strong
            strong_home = round_no == 2 or i < flip_first
            home, away = (strong, weak) if strong_home else (weak, strong)
            margin = 7 * abs(RANK[strong] - RANK[weak])
            home_score, away_score = (20 + margin, 20) if strong_home else (20, 20 + margin)
            n = len(rows)
            rows.append(
                (f"{season}_{n:02d}", season, (start + pd.Timedelta(days=n)).strftime("%Y-%m-%d"),
                 home, away, home_score, away_score)
            )
    return rows


def make_games(*seasons):
    """seasons: season ints, or (season, make_season kwargs) pairs."""
    rows = []
    for spec in seasons:
        season, kwargs = spec if isinstance(spec, tuple) else (spec, {})
        rows.extend(make_season(season, **kwargs))
    return pd.DataFrame(rows, columns=SEASON_COLUMNS)


TRAIN = [2001, 2002, 2003]
TEST = 2004
WINDOW = 4


def score(games_df, train_seasons=TRAIN, test_season=TEST, **kwargs):
    kwargs.setdefault("window", WINDOW)
    return score_walk_forward_fold(games_df, train_seasons, test_season, **kwargs)


def test_fully_predictable_league_scores_perfect_accuracy():
    accuracy = score(make_games(2001, 2002, 2003, 2004))

    assert isinstance(accuracy, float)
    assert accuracy == 1.0


def test_seasons_after_the_test_season_do_not_change_the_score():
    # A later season with the hierarchy reversed would wreck the features if
    # it leaked into the fold (Elo and rolling form run over every input row).
    base = make_games(2001, 2002, 2003, 2004)
    with_future = make_games(2001, 2002, 2003, 2004, (2005, {"reverse_hierarchy": True}))

    assert score(with_future) == score(base)


def test_seasons_outside_the_fold_are_not_passed_to_feature_building(monkeypatch):
    captured = {}
    real = validation.build_feature_table

    def spy(games_df, **kwargs):
        captured["seasons"] = set(games_df["season"])
        return real(games_df, **kwargs)

    monkeypatch.setattr(validation, "build_feature_table", spy)
    score(make_games(2000, 2001, 2002, 2003, 2004, 2005))

    assert captured["seasons"] == {2001, 2002, 2003, 2004}


def test_home_field_advantage_is_derived_from_the_fold_train_seasons_only(monkeypatch):
    # Train seasons: home win rate 7/12. Test season: 12/12 (every game
    # flipped), so any HFA that saw the test season would be larger.
    games = make_games(
        (2001, {"flip_first": 1}),
        (2002, {"flip_first": 1}),
        (2003, {"flip_first": 1}),
        (2004, {"flip_first": 6}),
    )
    captured = {}
    real = validation.build_feature_table

    def spy(games_df, **kwargs):
        captured["hfa"] = kwargs["home_field_advantage"]
        return real(games_df, **kwargs)

    monkeypatch.setattr(validation, "build_feature_table", spy)
    score(games)

    assert captured["hfa"] == pytest.approx(derive_home_field_advantage(7 / 12))


def test_passes_k_factor_window_and_initial_rating_to_feature_building(monkeypatch):
    captured = {}
    real = validation.build_feature_table

    def spy(games_df, **kwargs):
        captured.update(kwargs)
        return real(games_df, **kwargs)

    monkeypatch.setattr(validation, "build_feature_table", spy)
    score(make_games(2001, 2002, 2003, 2004), k_factor=32.0, window=3, initial_rating=1400.0)

    assert captured["k_factor"] == 32.0
    assert captured["window"] == 3
    assert captured["initial_rating"] == 1400.0


def test_test_season_inside_training_seasons_raises():
    with pytest.raises(ValueError):
        score(make_games(2001, 2002, 2003, 2004), train_seasons=[2001, 2002, 2003], test_season=2003)


def test_test_season_before_a_training_season_raises():
    with pytest.raises(ValueError, match="after"):
        score(make_games(2001, 2002, 2003, 2004), train_seasons=[2001, 2002, 2004], test_season=2003)


def test_test_season_without_games_raises():
    with pytest.raises(ValueError, match="no games"):
        score(make_games(2001, 2002, 2003), test_season=2004)


def test_test_rows_without_full_rolling_history_raise():
    # 3 train seasons x 6 games = 18 prior games per team; window 20 leaves
    # test rows without a full window.
    with pytest.raises(ValueError, match="rolling history"):
        score(make_games(2001, 2002, 2003, 2004), window=20)
