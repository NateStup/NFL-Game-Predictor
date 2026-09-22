"""Pure transforms over schedule DataFrames. No I/O lives here."""

import pandas as pd

# Relocated franchises: historical abbreviation -> current abbreviation.
FRANCHISE_RENAMES = {"STL": "LA", "SD": "LAC", "OAK": "LV"}


def drop_ties(df: pd.DataFrame) -> pd.DataFrame:
    """Return only games with a winner (home_score != away_score)."""
    return df[df["home_score"] != df["away_score"]]


def chronological_split(
    df: pd.DataFrame, train_seasons: list[int], test_seasons: list[int]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split games into train/test sets by season.

    Raises ValueError if the two season lists overlap, so a season can never
    leak from test into train.
    """
    overlap = set(train_seasons) & set(test_seasons)
    if overlap:
        raise ValueError(
            f"train_seasons and test_seasons overlap: {sorted(overlap)}"
        )
    train = df[df["season"].isin(train_seasons)]
    test = df[df["season"].isin(test_seasons)]
    return train, test


def home_baseline_accuracy(test_df: pd.DataFrame) -> float:
    """Accuracy of always predicting the home team: fraction of home wins."""
    return float((test_df["home_score"] > test_df["away_score"]).mean())


def regular_season_only(df: pd.DataFrame) -> pd.DataFrame:
    """Return only regular-season games (game_type == "REG"), dropping playoffs."""
    return df[df["game_type"] == "REG"]


def normalize_franchises(df: pd.DataFrame) -> pd.DataFrame:
    """Map relocated franchises to their current abbreviation on both sides,
    so each franchise has one continuous history."""
    return df.replace(
        {"home_team": FRANCHISE_RENAMES, "away_team": FRANCHISE_RENAMES}
    )
