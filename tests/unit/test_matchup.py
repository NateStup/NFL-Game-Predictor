"""Unit tests for combining two team snapshots into one model input row."""

import pandas as pd
import pytest

from nfl_predictor.predictor.features import FEATURE_COLUMNS
from nfl_predictor.predictor.matchup import build_matchup_features

# Hand-computed, home minus away:
#   rolling_win_pct:    0.75   - 0.375   = 0.375
#   rolling_point_diff: 10.625 - (-3.875) = 14.5
#   elo_rating:         1580.5 - 1492.25  = 88.25
HOME = {"team": "KC", "rolling_win_pct": 0.75, "rolling_point_diff": 10.625, "elo_rating": 1580.5}
AWAY = {"team": "LV", "rolling_win_pct": 0.375, "rolling_point_diff": -3.875, "elo_rating": 1492.25}
EXPECTED = {"rolling_win_pct_diff": 0.375, "rolling_point_diff": 14.5, "elo_diff": 88.25}


def test_keys_are_exactly_the_model_feature_columns_in_order():
    features = build_matchup_features(HOME, AWAY)

    assert list(features) == FEATURE_COLUMNS


def test_values_are_home_minus_away_from_dicts():
    features = build_matchup_features(HOME, AWAY)

    for key, value in EXPECTED.items():
        assert features[key] == pytest.approx(value)


def test_accepts_snapshot_rows_as_series():
    features = build_matchup_features(pd.Series(HOME), pd.Series(AWAY))

    for key, value in EXPECTED.items():
        assert features[key] == pytest.approx(value)
    assert all(type(v) is float for v in features.values())


def test_swapping_sides_negates_every_feature():
    forward = build_matchup_features(HOME, AWAY)
    reverse = build_matchup_features(AWAY, HOME)

    for key in FEATURE_COLUMNS:
        assert reverse[key] == pytest.approx(-forward[key])
