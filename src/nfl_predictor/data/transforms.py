"""Pure transforms over schedule DataFrames. No I/O lives here."""

import pandas as pd


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
