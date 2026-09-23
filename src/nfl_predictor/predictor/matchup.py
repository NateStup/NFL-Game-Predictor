"""Turn two team snapshots into one model input row (pure, no I/O)."""

from collections.abc import Mapping

import pandas as pd


def build_matchup_features(
    home_snapshot: Mapping | pd.Series, away_snapshot: Mapping | pd.Series
) -> dict:
    raise NotImplementedError
