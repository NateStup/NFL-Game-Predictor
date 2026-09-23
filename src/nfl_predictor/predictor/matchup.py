"""Turn two team snapshots into one model input row (pure, no I/O)."""

from collections.abc import Mapping

import pandas as pd

from nfl_predictor.predictor.features import FEATURE_COLUMNS

# Model feature -> snapshot field it is the home-minus-away difference of.
_SNAPSHOT_FIELDS = {
    "rolling_win_pct_diff": "rolling_win_pct",
    "rolling_point_diff": "rolling_point_diff",
    "elo_diff": "elo_rating",
}


def build_matchup_features(
    home_snapshot: Mapping | pd.Series, away_snapshot: Mapping | pd.Series
) -> dict:
    """Home-minus-away features for a future game, keyed and ordered exactly
    as the model's FEATURE_COLUMNS.

    Snapshots are rows shaped like get_team_state_snapshot's output; extra
    fields (e.g. team) are ignored.
    """
    return {
        feature: float(home_snapshot[_SNAPSHOT_FIELDS[feature]])
        - float(away_snapshot[_SNAPSHOT_FIELDS[feature]])
        for feature in FEATURE_COLUMNS
    }
