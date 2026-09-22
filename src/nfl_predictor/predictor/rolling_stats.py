"""Rolling team-form features (pure functions, no I/O)."""

import pandas as pd


def _to_team_games(games_df: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError


def _rolling_team_stat(
    long_df: pd.DataFrame, stat_column: str, window: int = 8
) -> pd.Series:
    raise NotImplementedError


def rolling_win_pct_diff(games_df: pd.DataFrame, window: int = 8) -> pd.Series:
    raise NotImplementedError


def rolling_point_diff(games_df: pd.DataFrame, window: int = 8) -> pd.Series:
    raise NotImplementedError
