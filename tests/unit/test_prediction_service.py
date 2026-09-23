"""Unit tests for the prediction service, using small synthetic artifacts."""

import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from nfl_predictor.predictor.features import FEATURE_COLUMNS
from nfl_predictor.predictor.matchup import build_matchup_features
from nfl_predictor.predictor.team_state import get_team_state_snapshot
from nfl_predictor.service.prediction_service import (
    InsufficientHistoryError,
    PredictionArtifacts,
    SameTeamError,
    UnknownTeamError,
    load_prediction_artifacts,
    predict_matchup,
)
from nfl_predictor.training.model import build_pipeline, train_pipeline

COLUMNS = ["game_id", "season", "week", "gameday", "home_team", "away_team", "home_score", "away_score"]

# 4 teams, double round robin (12 games, 6 per team). A wins every game, D
# loses every game, B beats C at home and loses to C away. E plays once, so
# it is known but short of the window-3 history.
GAMES = [
    ("g01", 2024, 1, "2024-09-08", "A", "B", 27, 10),
    ("g02", 2024, 1, "2024-09-08", "C", "D", 24, 3),
    ("g03", 2024, 2, "2024-09-15", "A", "C", 31, 17),
    ("g04", 2024, 2, "2024-09-15", "B", "D", 20, 13),
    ("g05", 2024, 3, "2024-09-22", "D", "A", 6, 34),
    ("g06", 2024, 3, "2024-09-22", "B", "C", 23, 20),
    ("g07", 2024, 4, "2024-09-29", "B", "A", 14, 28),
    ("g08", 2024, 4, "2024-09-29", "D", "C", 10, 21),
    ("g09", 2024, 5, "2024-10-06", "C", "A", 13, 30),
    ("g10", 2024, 5, "2024-10-06", "D", "B", 9, 24),
    ("g11", 2024, 6, "2024-10-13", "A", "D", 38, 0),
    ("g12", 2024, 6, "2024-10-13", "C", "B", 27, 17),
    ("g13", 2024, 7, "2024-10-20", "E", "D", 21, 14),
]
CONFIG = {
    "home_field_advantage": 30.0,
    "k_factor": 20.0,
    "initial_rating": 1500.0,
    "window": 3,
    "test_baseline_accuracy": 0.55,
}
RESULT_KEYS = {
    "home_team",
    "away_team",
    "home_win_probability",
    "away_win_probability",
    "predicted_winner",
    "features",
}


def fitted_toy_pipeline() -> Pipeline:
    # Home wins when the summed differentials are positive: enough to give a
    # model whose probabilities move in the obvious direction.
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(200, 3)) * [0.3, 8.0, 80.0], columns=FEATURE_COLUMNS)
    y = ((X / [0.3, 8.0, 80.0]).sum(axis=1) > 0).astype(int)
    return train_pipeline(build_pipeline(), X, y)


@pytest.fixture(scope="module")
def artifacts():
    return PredictionArtifacts(
        pipeline=fitted_toy_pipeline(),
        config=dict(CONFIG),
        games=pd.DataFrame(GAMES, columns=COLUMNS),
    )


# --- valid matchups ----------------------------------------------------------


def test_valid_matchup_returns_expected_shape(artifacts):
    result = predict_matchup(artifacts, "A", "D")

    assert set(result) == RESULT_KEYS
    assert (result["home_team"], result["away_team"]) == ("A", "D")
    assert list(result["features"]) == FEATURE_COLUMNS
    assert type(result["home_win_probability"]) is float
    assert 0.0 < result["home_win_probability"] < 1.0


def test_home_and_away_probabilities_sum_to_one(artifacts):
    result = predict_matchup(artifacts, "B", "C")

    assert result["home_win_probability"] + result["away_win_probability"] == pytest.approx(1.0)


def test_strong_home_team_is_favoured_and_named_winner(artifacts):
    result = predict_matchup(artifacts, "A", "D")

    assert result["home_win_probability"] > 0.5
    assert result["predicted_winner"] == "A"


def test_strong_away_team_is_favoured_and_named_winner(artifacts):
    # Guards the probability column orientation: the unbeaten team wins
    # whichever side it is on.
    result = predict_matchup(artifacts, "D", "A")

    assert result["home_win_probability"] < 0.5
    assert result["predicted_winner"] == "A"


def test_features_come_from_snapshot_built_with_config_values(artifacts):
    result = predict_matchup(artifacts, "B", "C")

    snap = get_team_state_snapshot(
        artifacts.games,
        home_field_advantage=CONFIG["home_field_advantage"],
        window=CONFIG["window"],
        k_factor=CONFIG["k_factor"],
        initial_rating=CONFIG["initial_rating"],
    ).set_index("team")
    expected = build_matchup_features(snap.loc["B"], snap.loc["C"])
    assert result["features"] == pytest.approx(expected)


def test_config_values_are_used_not_hardcoded(artifacts):
    base = predict_matchup(artifacts, "B", "C")
    other = PredictionArtifacts(
        pipeline=artifacts.pipeline,
        config={**CONFIG, "k_factor": 40.0, "window": 2},
        games=artifacts.games,
    )
    changed = predict_matchup(other, "B", "C")

    assert changed["features"]["elo_diff"] != pytest.approx(base["features"]["elo_diff"])
    assert changed["features"]["rolling_point_diff"] != pytest.approx(
        base["features"]["rolling_point_diff"]
    )


# --- errors ------------------------------------------------------------------


def test_unknown_home_team_raises_naming_it(artifacts):
    with pytest.raises(UnknownTeamError, match="ZZZ") as excinfo:
        predict_matchup(artifacts, "ZZZ", "A")

    assert excinfo.value.teams == ["ZZZ"]


def test_unknown_away_team_raises_naming_it(artifacts):
    with pytest.raises(UnknownTeamError, match="YYY") as excinfo:
        predict_matchup(artifacts, "A", "YYY")

    assert excinfo.value.teams == ["YYY"]


def test_both_unknown_teams_are_named(artifacts):
    with pytest.raises(UnknownTeamError) as excinfo:
        predict_matchup(artifacts, "ZZZ", "YYY")

    assert excinfo.value.teams == ["ZZZ", "YYY"]
    assert "ZZZ" in str(excinfo.value) and "YYY" in str(excinfo.value)


def test_same_team_raises(artifacts):
    with pytest.raises(SameTeamError, match="A"):
        predict_matchup(artifacts, "A", "A")


def test_team_short_of_window_raises_insufficient_history(artifacts):
    # E has 1 game; window is 3, so its rolling form is NaN and the model
    # cannot score it.
    with pytest.raises(InsufficientHistoryError, match="E") as excinfo:
        predict_matchup(artifacts, "E", "A")

    assert excinfo.value.teams == ["E"]


# --- loader ------------------------------------------------------------------


def test_load_prediction_artifacts_reads_all_three_files(tmp_path, artifacts):
    model_path = tmp_path / "model.joblib"
    config_path = tmp_path / "config.json"
    games_path = tmp_path / "games.parquet"
    joblib.dump(artifacts.pipeline, model_path)
    config_path.write_text(json.dumps(CONFIG))
    artifacts.games.to_parquet(games_path, engine="fastparquet", index=False)

    loaded = load_prediction_artifacts(model_path, config_path, games_path)

    assert isinstance(loaded, PredictionArtifacts)
    assert isinstance(loaded.pipeline, Pipeline)
    assert loaded.config == CONFIG
    pd.testing.assert_frame_equal(loaded.games, artifacts.games)
    assert predict_matchup(loaded, "A", "D") == predict_matchup(artifacts, "A", "D")
