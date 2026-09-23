"""Current per-team state for predicting a future matchup (pure, no I/O)."""

import pandas as pd

from nfl_predictor.predictor.elo import latest_team_elo_ratings
from nfl_predictor.predictor.rolling_stats import latest_team_rolling_stats


def get_team_state_snapshot(
    games_df: pd.DataFrame,
    *,
    home_field_advantage: float,
    window: int = 8,
    k_factor: float = 20.0,
    initial_rating: float = 1500.0,
) -> pd.DataFrame:
    """Each team's state after its most recent game in games_df.

    One row per team (sorted): team, rolling_win_pct, rolling_point_diff,
    elo_rating. Rolling values are NaN for a team with fewer than `window`
    games; its row (and Elo rating) is still present.
    """
    rolling = latest_team_rolling_stats(games_df, window=window)
    elo = latest_team_elo_ratings(
        games_df,
        initial_rating=initial_rating,
        k_factor=k_factor,
        home_field_advantage=home_field_advantage,
    )
    # Both sides come from the same games, so the team sets are identical;
    # validate makes a duplicate on either side fail loudly.
    return rolling.merge(elo, on="team", how="outer", validate="one_to_one")
