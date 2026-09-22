"""Sequential Elo ratings (pure functions, no I/O)."""

import math

import pandas as pd


def derive_home_field_advantage(home_win_rate: float) -> float:
    """Elo points of home advantage implied by a historical home win rate.

    Inverts the Elo win-probability curve: the rating edge that makes two
    otherwise equal teams produce `home_win_rate` for the home side.
    """
    if not 0 < home_win_rate < 1:
        raise ValueError(f"home_win_rate must be in (0, 1), got {home_win_rate}")
    return -400 * math.log10((1 / home_win_rate) - 1)


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
