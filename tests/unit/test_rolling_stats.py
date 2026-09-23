"""Unit tests for rolling team-form features (synthetic, hand-computable data).

Fixture: 3 teams (A, B, C) in a repeating round robin, 15 games, 10 per team.
Games g0-g8 are season 2023, g9-g14 are season 2024.

 game  gameday     home-away  score   result (point diff per team)
 g0    2023-09-10  A - B      24-17   A W +7    B L -7
 g1    2023-09-17  C - A      10-20   A W +10   C L -10
 g2    2023-09-24  B - C      21-14   B W +7    C L -7
 g3    2023-10-01  B - A      27-13   B W +14   A L -14
 g4    2023-10-08  A - C      17-20   C W +3    A L -3
 g5    2023-10-15  C - B      30-10   C W +20   B L -20
 g6    2023-10-22  A - B      28-21   A W +7    B L -7
 g7    2023-10-29  C - A      16-23   A W +7    C L -7
 g8    2023-11-05  B - C      24-10   B W +14   C L -14
 ---- season boundary ----
 g9    2024-09-08  B - A      20-17   B W +3    A L -3
 g10   2024-09-15  A - C      31-3    A W +28   C L -28
 g11   2024-09-22  C - B      17-14   C W +3    B L -3
 g12   2024-09-29  A - B      21-10   A W +11   B L -11
 g13   2024-10-06  C - A      7-35    A W +28   C L -28
 g14   2024-10-13  B - C      27-24   B W +3    C L -3

Every team reaches 8 prior games only at g12 (A, B) and g13 (C), so g0-g11
are warmup NaNs and g12-g14 are the only games with a full window on both
sides.
"""

import numpy as np
import pandas as pd
import pytest

from nfl_predictor.predictor.rolling_stats import (
    _rolling_team_stat,
    _to_team_games,
    latest_team_rolling_stats,
    rolling_point_diff,
    rolling_win_pct_diff,
)

GAMES = [
    # (gameday, season, home, away, home_score, away_score)
    ("2023-09-10", 2023, "A", "B", 24, 17),
    ("2023-09-17", 2023, "C", "A", 10, 20),
    ("2023-09-24", 2023, "B", "C", 21, 14),
    ("2023-10-01", 2023, "B", "A", 27, 13),
    ("2023-10-08", 2023, "A", "C", 17, 20),
    ("2023-10-15", 2023, "C", "B", 30, 10),
    ("2023-10-22", 2023, "A", "B", 28, 21),
    ("2023-10-29", 2023, "C", "A", 16, 23),
    ("2023-11-05", 2023, "B", "C", 24, 10),
    ("2024-09-08", 2024, "B", "A", 20, 17),
    ("2024-09-15", 2024, "A", "C", 31, 3),
    ("2024-09-22", 2024, "C", "B", 17, 14),
    ("2024-09-29", 2024, "A", "B", 21, 10),
    ("2024-10-06", 2024, "C", "A", 7, 35),
    ("2024-10-13", 2024, "B", "C", 27, 24),
]

# Hand-calculated features for g12. Each team's window is its 8 PRIOR games.
#
# A (home), prior 8 = g0, g1, g3, g4, g6, g7, g9, g10
#   outcomes: W W L L W W L W                 -> 5 wins / 8 = 0.625
#   diffs:    +7 +10 -14 -3 +7 +7 -3 +28 = 39 -> 39 / 8     = 4.875
# B (away), prior 8 = g0, g2, g3, g5, g6, g8, g9, g11
#   outcomes: L W W L L W W L                 -> 4 wins / 8 = 0.500
#   diffs:    -7 +7 +14 -20 -7 +14 +3 -3 = 1  -> 1 / 8      = 0.125
# win_pct_diff = 0.625 - 0.500 = 0.125
# point_diff   = 4.875 - 0.125 = 4.75
G12_WIN_PCT_DIFF = 0.125
G12_POINT_DIFF = 4.75

# g13 (A away, and A's window has slid forward one game, dropping g0).
# C (home), prior 8 = g1, g2, g4, g5, g7, g8, g10, g11
#   outcomes: L L W W L L L W                     -> 3 / 8 = 0.375
#   diffs:    -10 -7 +3 +20 -7 -14 -28 +3 = -40   -> -40 / 8 = -5.0
# A (away), prior 8 = g1, g3, g4, g6, g7, g9, g10, g12
#   outcomes: W L L W W L W W                     -> 5 / 8 = 0.625
#   diffs:    +10 -14 -3 +7 +7 -3 +28 +11 = 43    -> 43 / 8 = 5.375
# win_pct_diff = 0.375 - 0.625 = -0.25
# point_diff   = -5.0 - 5.375  = -10.375
G13_WIN_PCT_DIFF = -0.25
G13_POINT_DIFF = -10.375

# g14 (B's window drops g0, C's window drops g1).
# B (home), prior 8 = g2, g3, g5, g6, g8, g9, g11, g12
#   outcomes: W W L L W W L L                     -> 4 / 8 = 0.5
#   diffs:    +7 +14 -20 -7 +14 +3 -3 -11 = -3    -> -3 / 8 = -0.375
# C (away), prior 8 = g2, g4, g5, g7, g8, g10, g11, g13
#   outcomes: L W W L L L W L                     -> 3 / 8 = 0.375
#   diffs:    -7 +3 +20 -7 -14 -28 +3 -28 = -58   -> -58 / 8 = -7.25
# win_pct_diff = 0.5 - 0.375      = 0.125
# point_diff   = -0.375 - (-7.25) = 6.875
G14_WIN_PCT_DIFF = 0.125
G14_POINT_DIFF = 6.875


@pytest.fixture
def games_df():
    return pd.DataFrame(
        GAMES,
        columns=["gameday", "season", "home_team", "away_team", "home_score", "away_score"],
    )


# --- reshape helper ----------------------------------------------------------


def test_to_team_games_produces_two_rows_per_game(games_df):
    long_df = _to_team_games(games_df)

    assert len(long_df) == 2 * len(games_df)
    assert long_df.groupby("game_idx").size().eq(2).all()


def test_to_team_games_team_perspective_values(games_df):
    long_df = _to_team_games(games_df)

    g0 = long_df[long_df["game_idx"] == 0].set_index("team")
    assert g0.loc["A", "won"] == 1 and g0.loc["A", "point_diff"] == 7
    assert g0.loc["B", "won"] == 0 and g0.loc["B", "point_diff"] == -7


# --- shared windowing function -----------------------------------------------


def test_rolling_team_stat_is_per_team_and_shifted(games_df):
    long_df = _to_team_games(games_df)

    rolled = _rolling_team_stat(long_df, "point_diff", window=8)

    team_a = rolled[long_df["team"] == "A"].tolist()
    # A's games in order: g0 g1 g3 g4 g6 g7 g9 g10 g12 g13.
    # First 8 have < 8 prior games; 9th (g12) and 10th (g13) per the header math.
    assert all(np.isnan(v) for v in team_a[:8])
    assert team_a[8] == pytest.approx(4.875)
    assert team_a[9] == pytest.approx(5.375)


# --- (a) warmup NaNs ---------------------------------------------------------


def test_games_without_full_history_are_nan(games_df):
    win = rolling_win_pct_diff(games_df)
    pts = rolling_point_diff(games_df)

    # g0-g11: at least one team has fewer than 8 prior games.
    # g11 (C vs B) is the sharp edge: both teams have exactly 7 prior games.
    assert win.iloc[:12].isna().all()
    assert pts.iloc[:12].isna().all()
    assert win.iloc[12:].notna().all()
    assert pts.iloc[12:].notna().all()


# --- (b) exact hand-calculated values ----------------------------------------


def test_full_window_matches_hand_calculation(games_df):
    win = rolling_win_pct_diff(games_df)
    pts = rolling_point_diff(games_df)

    assert win.iloc[12] == pytest.approx(G12_WIN_PCT_DIFF)
    assert pts.iloc[12] == pytest.approx(G12_POINT_DIFF)
    # Sliding window, and away-side orientation (A is away in g13).
    assert win.iloc[13] == pytest.approx(G13_WIN_PCT_DIFF)
    assert pts.iloc[13] == pytest.approx(G13_POINT_DIFF)
    assert win.iloc[14] == pytest.approx(G14_WIN_PCT_DIFF)
    assert pts.iloc[14] == pytest.approx(G14_POINT_DIFF)


# --- (c) leakage -------------------------------------------------------------


def test_target_game_result_does_not_leak_into_its_own_feature(games_df):
    before_win = rolling_win_pct_diff(games_df)
    before_pts = rolling_point_diff(games_df)

    # Flip g12 from A 21-10 win to A 0-50 loss; no earlier game changes.
    altered = games_df.copy()
    altered.loc[12, ["home_score", "away_score"]] = [0, 50]
    after_win = rolling_win_pct_diff(altered)
    after_pts = rolling_point_diff(altered)

    assert after_win.iloc[12] == pytest.approx(before_win.iloc[12])
    assert after_pts.iloc[12] == pytest.approx(before_pts.iloc[12])
    # Positive control: later games whose windows include g12 DO change.
    assert after_pts.iloc[13] != pytest.approx(before_pts.iloc[13])
    assert after_win.iloc[13] != pytest.approx(before_win.iloc[13])


# --- (d) no reset at the season boundary -------------------------------------


def test_window_carries_across_season_boundary(games_df):
    # g12 is only the 3rd 2024 game for both A and B (A: g9, g10; B: g9, g11
    # before it), so 6 of the 8 games in each window come from 2023. A reset
    # at week 1 would leave 2 prior games -> NaN.
    pts = rolling_point_diff(games_df)
    assert pts.iloc[12] == pytest.approx(G12_POINT_DIFF)

    # Proof the 2023 games are what fill the window: remove them and g12
    # no longer has a full history.
    season_2024_only = games_df[games_df["season"] == 2024]
    assert np.isnan(rolling_point_diff(season_2024_only).loc[12])
    assert np.isnan(rolling_win_pct_diff(season_2024_only).loc[12])


# --- alignment ---------------------------------------------------------------


def test_output_aligned_to_input_index_and_order(games_df):
    shuffled = games_df.sample(frac=1, random_state=0)
    shuffled.index = shuffled.index + 100  # non-default labels

    pts = rolling_point_diff(shuffled)
    win = rolling_win_pct_diff(shuffled)

    assert list(pts.index) == list(shuffled.index)
    assert list(win.index) == list(shuffled.index)
    assert pts.loc[112] == pytest.approx(G12_POINT_DIFF)
    assert win.loc[113] == pytest.approx(G13_WIN_PCT_DIFF)


def test_duplicate_index_raises(games_df):
    duplicated = games_df.set_axis([0] * len(games_df))

    with pytest.raises(ValueError, match="unique"):
        rolling_point_diff(duplicated)


# =============================================================================
# Current-state snapshot: last `window` games PLAYED, unshifted (the most
# recent game IS included). Used for future matchups, not training rows.
# =============================================================================

# One extra game after g14 so the unshifted and shifted windows differ in
# both win% and point diff (in GAMES alone, each team's dropped game and
# newest game happen to share a result, so win% would coincide).
#   g15  2024-10-20  A - B  24-14   A W +10   B L -10
G15 = ("2024-10-20", 2024, "A", "B", 24, 14)

# A's games: g0 g1 g3 g4 g6 g7 g9 g10 g12 g13 g15 (11 games).
# Snapshot = last 8 PLAYED = g4, g6, g7, g9, g10, g12, g13, g15
#   outcomes: L W W L W W W W                       -> 6 / 8 = 0.75
#   diffs:    -3 +7 +7 -3 +28 +11 +28 +10 = 85      -> 85 / 8 = 10.625
# The shifted (training) value on A's last row, g15, excludes g15 itself:
#   g3, g4, g6, g7, g9, g10, g12, g13
#   outcomes: L L W W L W W W                       -> 5 / 8 = 0.625
#   diffs:    -14 -3 +7 +7 -3 +28 +11 +28 = 61      -> 61 / 8 = 7.625
A_SNAPSHOT = (0.75, 10.625)
A_SHIFTED_LAST_ROW = (0.625, 7.625)

# B's snapshot = g5, g6, g8, g9, g11, g12, g14, g15
#   outcomes: L L W W L L W L                       -> 3 / 8 = 0.375
#   diffs:    -20 -7 +14 +3 -3 -11 +3 -10 = -31     -> -31 / 8 = -3.875
B_SNAPSHOT = (0.375, -3.875)

# C (not in g15) snapshot = g4, g5, g7, g8, g10, g11, g13, g14
#   outcomes: W W L L L W L L                       -> 3 / 8 = 0.375
#   diffs:    +3 +20 -7 -14 -28 +3 -28 -3 = -54     -> -54 / 8 = -6.75
C_SNAPSHOT = (0.375, -6.75)


@pytest.fixture
def games_with_g15(games_df):
    extra = pd.DataFrame([G15], columns=games_df.columns)
    return pd.concat([games_df, extra], ignore_index=True)


def test_snapshot_one_row_per_team_with_expected_columns(games_with_g15):
    snap = latest_team_rolling_stats(games_with_g15)

    assert list(snap.columns) == ["team", "rolling_win_pct", "rolling_point_diff"]
    assert list(snap["team"]) == ["A", "B", "C"]


def test_snapshot_matches_hand_calculated_last_8_games(games_with_g15):
    snap = latest_team_rolling_stats(games_with_g15).set_index("team")

    for team, (win_pct, point_diff) in [
        ("A", A_SNAPSHOT),
        ("B", B_SNAPSHOT),
        ("C", C_SNAPSHOT),
    ]:
        assert snap.loc[team, "rolling_win_pct"] == pytest.approx(win_pct)
        assert snap.loc[team, "rolling_point_diff"] == pytest.approx(point_diff)


def test_snapshot_includes_most_recent_game_unlike_shifted_feature(games_with_g15):
    snap = latest_team_rolling_stats(games_with_g15).set_index("team")
    long_df = _to_team_games(games_with_g15)
    is_a = long_df["team"] == "A"
    shifted_win = _rolling_team_stat(long_df, "won")[is_a].iloc[-1]
    shifted_pts = _rolling_team_stat(long_df, "point_diff")[is_a].iloc[-1]

    # The training-time value is one game stale ...
    assert (shifted_win, shifted_pts) == pytest.approx(A_SHIFTED_LAST_ROW)
    # ... the snapshot is not.
    assert snap.loc["A", "rolling_win_pct"] != pytest.approx(shifted_win)
    assert snap.loc["A", "rolling_point_diff"] != pytest.approx(shifted_pts)


def test_snapshot_equals_training_feature_for_the_next_game(games_with_g15):
    # A future A-vs-C game: its training-time (shifted) feature is built from
    # exactly the games already played, i.e. the snapshot. Its own score is a
    # placeholder; the shifted feature never reads it.
    future = pd.DataFrame(
        [("2024-10-27", 2024, "A", "C", 0, 0)], columns=games_with_g15.columns
    )
    with_future = pd.concat([games_with_g15, future], ignore_index=True)
    snap = latest_team_rolling_stats(games_with_g15).set_index("team")

    expected_pts = snap.loc["A", "rolling_point_diff"] - snap.loc["C", "rolling_point_diff"]
    expected_win = snap.loc["A", "rolling_win_pct"] - snap.loc["C", "rolling_win_pct"]
    assert rolling_point_diff(with_future).iloc[-1] == pytest.approx(expected_pts)
    assert rolling_win_pct_diff(with_future).iloc[-1] == pytest.approx(expected_win)


def test_snapshot_is_nan_for_team_with_fewer_than_window_games():
    # Two teams, 3 games each: short of an 8-game window, so NaN (same full-
    # window rule as training). With window=3 the same data is complete.
    df = pd.DataFrame(
        [
            ("2024-09-08", 2024, "A", "B", 20, 10),
            ("2024-09-15", 2024, "B", "A", 17, 14),
            ("2024-09-22", 2024, "A", "B", 30, 3),
        ],
        columns=["gameday", "season", "home_team", "away_team", "home_score", "away_score"],
    )

    short = latest_team_rolling_stats(df, window=8)
    assert short[["rolling_win_pct", "rolling_point_diff"]].isna().all().all()
    assert list(short["team"]) == ["A", "B"]  # teams are still listed

    # A: W +10, L -3, W +27 -> win% 2/3, diff 34/3
    full = latest_team_rolling_stats(df, window=3).set_index("team")
    assert full.loc["A", "rolling_win_pct"] == pytest.approx(2 / 3)
    assert full.loc["A", "rolling_point_diff"] == pytest.approx(34 / 3)
