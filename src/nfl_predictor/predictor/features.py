"""Assemble per-game features and target into one table (pure, no I/O)."""

import pandas as pd


def build_feature_table(
    games_df: pd.DataFrame,
    *,
    home_field_advantage: float,
    window: int = 8,
    k_factor: float = 20.0,
    initial_rating: float = 1500.0,
) -> pd.DataFrame:
    raise NotImplementedError
