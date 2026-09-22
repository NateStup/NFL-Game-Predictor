"""Assemble per-game features and target into one table (pure, no I/O)."""

import pandas as pd

from nfl_predictor.predictor.elo import compute_elo_ratings
from nfl_predictor.predictor.rolling_stats import rolling_point_diff, rolling_win_pct_diff

# Enough to identify a game and re-split by season. Scores are deliberately
# left out: home_win is derived from them, so they would leak the target.
ID_COLUMNS = ["game_id", "season", "gameday", "home_team", "away_team"]
FEATURE_COLUMNS = ["rolling_win_pct_diff", "rolling_point_diff", "elo_diff"]
TARGET_COLUMN = "home_win"


def build_feature_table(
    games_df: pd.DataFrame,
    *,
    home_field_advantage: float,
    window: int = 8,
    k_factor: float = 20.0,
    initial_rating: float = 1500.0,
) -> pd.DataFrame:
    """One row per input game: id columns, the three features, and home_win.

    home_field_advantage is supplied by the caller, which knows which rows are
    training data. Every input row is kept, including warmup rows whose
    rolling features are NaN; dropping them is the caller's decision.
    Expects tie-free games (a tie would be labelled home_win = 0).
    """
    table = games_df[ID_COLUMNS].copy()
    table["rolling_win_pct_diff"] = rolling_win_pct_diff(games_df, window=window)
    table["rolling_point_diff"] = rolling_point_diff(games_df, window=window)
    table["elo_diff"] = compute_elo_ratings(
        games_df,
        initial_rating=initial_rating,
        k_factor=k_factor,
        home_field_advantage=home_field_advantage,
    )
    table[TARGET_COLUMN] = (games_df["home_score"] > games_df["away_score"]).astype("int64")
    return table
