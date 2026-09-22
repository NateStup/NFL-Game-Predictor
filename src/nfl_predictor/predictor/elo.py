"""Sequential Elo ratings (pure functions, no I/O)."""

import pandas as pd


def derive_home_field_advantage(home_win_rate: float) -> float:
    raise NotImplementedError


def expected_home_win_prob(
    home_rating: float, away_rating: float, home_field_advantage: float
) -> float:
    raise NotImplementedError


def update_ratings(
    home_rating: float,
    away_rating: float,
    home_won: bool,
    home_field_advantage: float,
    k_factor: float,
) -> tuple[float, float]:
    raise NotImplementedError


def compute_elo_ratings(
    games_df: pd.DataFrame,
    initial_rating: float,
    k_factor: float,
    home_field_advantage: float,
) -> pd.Series:
    raise NotImplementedError
