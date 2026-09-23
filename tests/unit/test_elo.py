"""Unit tests for the Elo rating module (synthetic, hand-computable data).

Test constants (not production values): INITIAL = 1500, K = 20, HFA = 65.
"""

import numpy as np
import pandas as pd
import pytest

from nfl_predictor.predictor.elo import (
    compute_elo_ratings,
    derive_home_field_advantage,
    expected_home_win_prob,
    latest_team_elo_ratings,
    update_ratings,
)

INITIAL = 1500.0
K = 20.0
HFA = 65.0

# Rotation fixture: 3 teams, 12 games, 8 per team, g0-g5 in 2023 and g6-g11
# in 2024. A wins every game it plays; B and C split their games.
#
#  game  gameday     home-away  score  winner
#  g0    2023-09-10  A - B      24-17  A
#  g1    2023-09-17  C - A      10-20  A
#  g2    2023-09-24  B - C      21-14  B
#  g3    2023-10-01  B - A      13-27  A
#  g4    2023-10-08  A - C      30-3   A
#  g5    2023-10-15  C - B      20-17  C
#  ---- season boundary ----
#  g6    2024-09-08  A - B      28-21  A
#  g7    2024-09-15  C - A      16-23  A
#  g8    2024-09-22  B - C      24-10  B
#  g9    2024-09-29  B - A      17-20  A
#  g10   2024-10-06  A - C      31-3   A
#  g11   2024-10-13  C - B      17-14  C
GAMES = [
    # (gameday, season, home, away, home_score, away_score)
    ("2023-09-10", 2023, "A", "B", 24, 17),
    ("2023-09-17", 2023, "C", "A", 10, 20),
    ("2023-09-24", 2023, "B", "C", 21, 14),
    ("2023-10-01", 2023, "B", "A", 13, 27),
    ("2023-10-08", 2023, "A", "C", 30, 3),
    ("2023-10-15", 2023, "C", "B", 20, 17),
    ("2024-09-08", 2024, "A", "B", 28, 21),
    ("2024-09-15", 2024, "C", "A", 16, 23),
    ("2024-09-22", 2024, "B", "C", 24, 10),
    ("2024-09-29", 2024, "B", "A", 17, 20),
    ("2024-10-06", 2024, "A", "C", 31, 3),
    ("2024-10-13", 2024, "C", "B", 17, 14),
]
COLUMNS = ["gameday", "season", "home_team", "away_team", "home_score", "away_score"]


@pytest.fixture
def games_df():
    return pd.DataFrame(GAMES, columns=COLUMNS)


def elo(df, hfa=HFA):
    return compute_elo_ratings(df, initial_rating=INITIAL, k_factor=K, home_field_advantage=hfa)


# =============================================================================
# PART 1: home-field advantage derivation
#   HFA = -400 * log10((1 / home_win_rate) - 1)
# =============================================================================


def test_hfa_rate_ten_elevenths_is_400():
    # 1 / (10/11) = 1.1;  1.1 - 1 = 0.1;  log10(0.1) = -1;  -400 * -1 = 400
    assert derive_home_field_advantage(10 / 11) == pytest.approx(400.0)


def test_hfa_rate_one_eleventh_is_minus_400():
    # 1 / (1/11) = 11;  11 - 1 = 10;  log10(10) = 1;  -400 * 1 = -400
    assert derive_home_field_advantage(1 / 11) == pytest.approx(-400.0)


def test_hfa_rate_sixty_percent():
    # 1 / 0.6 = 5/3;  5/3 - 1 = 2/3;  log10(2/3) = log10(2) - log10(3)
    #   = 0.30103 - 0.47712 = -0.17609;  -400 * -0.17609 = 70.4365
    assert derive_home_field_advantage(0.6) == pytest.approx(70.4365, abs=1e-4)


def test_hfa_coin_flip_is_exactly_zero():
    # 1 / 0.5 = 2;  2 - 1 = 1;  log10(1) = 0;  -400 * 0 = 0
    assert derive_home_field_advantage(0.5) == 0.0


@pytest.mark.parametrize("rate", [0.0, 1.0, -0.1, 1.5])
def test_hfa_rejects_rates_outside_open_unit_interval(rate):
    with pytest.raises(ValueError):
        derive_home_field_advantage(rate)


# =============================================================================
# PART 2: win probability
#   P(home) = 1 / (1 + 10^(-((home + hfa) - away) / 400))
# =============================================================================


def test_win_prob_equal_ratings_no_hfa_is_one_half():
    # exponent = -(1500 - 1500) / 400 = 0;  10^0 = 1;  1 / (1 + 1) = 0.5
    assert expected_home_win_prob(1500, 1500, 0) == 0.5


def test_win_prob_equal_ratings_hfa_65():
    # exponent = -(1565 - 1500) / 400 = -0.1625;  10^-0.1625 = 0.687860
    # 1 / (1 + 0.687860) = 1 / 1.687860 = 0.592466
    assert expected_home_win_prob(1500, 1500, 65) == pytest.approx(0.592466, abs=1e-6)


def test_win_prob_400_point_edge_is_ten_elevenths():
    # exponent = -(1900 - 1500) / 400 = -1;  10^-1 = 0.1;  1 / 1.1 = 10/11
    assert expected_home_win_prob(1900, 1500, 0) == pytest.approx(10 / 11)


def test_win_prob_increases_with_home_rating():
    probs = [expected_home_win_prob(r, 1500, 65) for r in range(1300, 1800, 50)]

    assert all(later > earlier for earlier, later in zip(probs, probs[1:]))
    assert all(0 < p < 1 for p in probs)


# =============================================================================
# PART 2: single-game update
# =============================================================================


@pytest.mark.parametrize("home_won", [True, False])
def test_update_winner_gains_loser_drops(home_won):
    new_home, new_away = update_ratings(1500, 1500, home_won, HFA, K)

    if home_won:
        assert new_home > 1500 and new_away < 1500
    else:
        assert new_home < 1500 and new_away > 1500


@pytest.mark.parametrize("home_won", [True, False])
def test_update_is_zero_sum(home_won):
    new_home, new_away = update_ratings(1580, 1460, home_won, HFA, K)

    home_change = new_home - 1580
    away_change = new_away - 1460
    assert home_change + away_change == pytest.approx(0.0, abs=1e-9)
    assert home_change != 0


def test_update_upset_swings_more_than_expected_result():
    # Home 1700 vs away 1400: the home team is a heavy favourite.
    fav_home, _ = update_ratings(1700, 1400, True, HFA, K)
    upset_home, _ = update_ratings(1700, 1400, False, HFA, K)

    favourite_swing = abs(fav_home - 1700)
    upset_swing = abs(upset_home - 1700)
    assert upset_swing > favourite_swing


def test_update_matches_hand_calculation():
    # P(home) at equal ratings, HFA 65 = 0.592466 (see above).
    # Home wins: delta = K * (1 - 0.592466) = 20 * 0.407534 = 8.150675
    new_home, new_away = update_ratings(1500, 1500, True, HFA, K)

    assert new_home == pytest.approx(1508.150675, abs=1e-6)
    assert new_away == pytest.approx(1491.849325, abs=1e-6)


# =============================================================================
# PART 2: sequential engine. The output is rating-only (home - away, PRE-game);
# HFA is used only inside the win probability that drives the updates.
# =============================================================================


def test_first_game_for_new_teams_is_zero_even_with_hfa(games_df):
    # g0: A and B both start at INITIAL. 1500 - 1500 = 0, and HFA is NOT
    # added to the output feature, so it stays exactly 0 with HFA = 65.
    assert elo(games_df).iloc[0] == 0.0


def test_new_team_enters_at_initial_rating(games_df):
    # g1: C (home) is new at 1500. A won g0 at home: 1500 + 8.150675.
    # diff = 1500 - 1508.150675 = -8.150675
    assert elo(games_df).iloc[1] == pytest.approx(-8.150675, abs=1e-6)


def test_flipping_a_result_propagates_forward_not_backward(games_df):
    before = elo(games_df)

    # Flip g4 (A 30-3 over C) to a C win; nothing earlier changes.
    altered = games_df.copy()
    altered.loc[4, ["home_score", "away_score"]] = [3, 30]
    after = elo(altered)

    # g0-g4 (including g4's own pre-game differential) are unchanged.
    pd.testing.assert_series_equal(after.iloc[:5], before.iloc[:5])
    # g5 (C vs B) is the next game involving A or C; it must change.
    assert after.iloc[5] != pytest.approx(before.iloc[5])


def test_rating_carries_across_season_boundary():
    # Two teams, one game per season, HFA = 0 so the arithmetic is exact.
    # 2023: A (home) beats B. P = 0.5, delta = 20 * (1 - 0.5) = 10.
    #       A -> 1510, B -> 1490.
    # 2024: B (home) vs A. Pre-game diff = 1490 - 1510 = -20.
    #       A reset to 1500 at week 1 would give 0 instead.
    df = pd.DataFrame(
        [
            ("2023-09-10", 2023, "A", "B", 24, 17),
            ("2024-09-08", 2024, "B", "A", 20, 17),
        ],
        columns=COLUMNS,
    )

    result = elo(df, hfa=0.0)

    assert result.iloc[0] == 0.0
    assert result.iloc[1] == pytest.approx(-20.0)


def test_season_one_history_drives_season_two_opener(games_df):
    # g6 is the first 2024 game. Without 2023 in the input, both teams would
    # be new (diff 0); with it, A's unbeaten 2023 shows up in the rating.
    full = elo(games_df)
    only_2024 = elo(games_df[games_df["season"] == 2024])

    assert only_2024.loc[6] == 0.0
    assert full.loc[6] > 0


def test_unbeaten_team_rating_trends_up(games_df):
    result = elo(games_df)

    # A's-perspective differential across its 8 games (sign flipped when away).
    is_a_home = games_df["home_team"] == "A"
    is_a_away = games_df["away_team"] == "A"
    a_diff = pd.concat([result[is_a_home], -result[is_a_away]]).sort_index().tolist()

    assert len(a_diff) == 8
    assert a_diff[0] == 0.0
    assert all(d > 0 for d in a_diff[1:])
    assert a_diff[-1] > a_diff[len(a_diff) // 2] > a_diff[1]


# --- defensive behaviour -----------------------------------------------------


def test_unsorted_input_is_sorted_and_aligned_to_index(games_df):
    expected = elo(games_df)

    shuffled = games_df.sample(frac=1, random_state=0)
    shuffled.index = shuffled.index + 100
    result = elo(shuffled)

    assert list(result.index) == list(shuffled.index)
    for idx in shuffled.index:
        assert result.loc[idx] == pytest.approx(expected.iloc[idx - 100])


def test_null_date_raises(games_df):
    games_df.loc[3, "gameday"] = None

    with pytest.raises(ValueError, match="gameday"):
        elo(games_df)


def test_duplicate_index_raises(games_df):
    with pytest.raises(ValueError, match="unique"):
        elo(games_df.set_axis([0] * len(games_df)))


# =============================================================================
# Current-state snapshot: each team's rating AFTER its most recent game.
# =============================================================================

# Three games, HFA = 0 so every step is hand-computable. K = 20, start 1500.
#
# g0  A (home) beats B.  Equal ratings: P(home) = 0.5
#     delta = 20 * (1 - 0.5) = 10          -> A 1510,  B 1490
# g1  B (home) loses to C.  edge = 1490 - 1500 = -10
#     P(home) = 1 / (1 + 10^(10/400)) = 1 / (1 + 10^0.025)
#             = 1 / (1 + 1.059254) = 0.485613
#     delta = 20 * (0 - 0.485613) = -9.712256
#     -> B 1490 - 9.712256 = 1480.287744,  C 1500 + 9.712256 = 1509.712256
# g2  A (home) beats C.  edge = 1510 - 1509.712256 = 0.287744
#     P(home) = 1 / (1 + 10^(-0.287744/400)) = 0.500414
#     delta = 20 * (1 - 0.500414) = 9.991718
#     -> A 1510 + 9.991718 = 1519.991718,  C 1509.712256 - 9.991718 = 1499.720538
# Final: A 1519.991718, B 1480.287744, C 1499.720538  (sum 4500: zero-sum)
SNAPSHOT_GAMES = [
    ("2024-09-08", 2024, "A", "B", 24, 17),
    ("2024-09-15", 2024, "B", "C", 10, 20),
    ("2024-09-22", 2024, "A", "C", 27, 13),
]
SNAPSHOT_FINAL = {"A": 1519.991718, "B": 1480.287744, "C": 1499.720538}


@pytest.fixture
def snapshot_games():
    return pd.DataFrame(SNAPSHOT_GAMES, columns=COLUMNS)


def latest(df, hfa=0.0):
    return latest_team_elo_ratings(
        df, initial_rating=INITIAL, k_factor=K, home_field_advantage=hfa
    )


def test_elo_snapshot_one_row_per_team_with_expected_columns(snapshot_games):
    snap = latest(snapshot_games)

    assert list(snap.columns) == ["team", "elo_rating"]
    assert list(snap["team"]) == ["A", "B", "C"]


def test_elo_snapshot_matches_hand_calculated_final_ratings(snapshot_games):
    snap = latest(snapshot_games).set_index("team")["elo_rating"]

    for team, rating in SNAPSHOT_FINAL.items():
        assert snap[team] == pytest.approx(rating, abs=1e-6)
    assert snap.sum() == pytest.approx(3 * INITIAL)


def test_elo_snapshot_is_post_update_not_pre_game(snapshot_games):
    # A's rating going INTO its last game (g2) was 1510; the snapshot must
    # include g2's update.
    snap = latest(snapshot_games).set_index("team")["elo_rating"]

    assert snap["A"] != pytest.approx(1510.0)
    assert snap["A"] == pytest.approx(1510.0 + 9.991718, abs=1e-6)


def test_elo_snapshot_equals_pre_game_diff_of_the_next_game(games_df):
    # A future A-vs-B game's training-time elo_diff is built from ratings after
    # every played game, i.e. the snapshot. Its placeholder score is never
    # read for its own differential.
    future = pd.DataFrame([("2024-10-20", 2024, "A", "B", 0, 0)], columns=COLUMNS)
    with_future = pd.concat([games_df, future], ignore_index=True)

    snap = latest(games_df, hfa=HFA).set_index("team")["elo_rating"]

    assert elo(with_future).iloc[-1] == pytest.approx(snap["A"] - snap["B"])
