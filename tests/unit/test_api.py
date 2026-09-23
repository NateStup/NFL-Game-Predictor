"""API schema and route tests: TestClient against an app with synthetic artifacts."""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from nfl_predictor.api.main import create_app
from nfl_predictor.api.schemas import PredictResponse
from nfl_predictor.predictor.features import FEATURE_COLUMNS
from nfl_predictor.service.prediction_service import PredictionArtifacts
from nfl_predictor.training.model import build_pipeline, train_pipeline

COLUMNS = ["game_id", "season", "week", "gameday", "home_team", "away_team", "home_score", "away_score"]
# A beats everyone, C loses to everyone; E has a single game (short of window 2).
GAMES = [
    ("g1", 2024, 1, "2024-09-08", "A", "B", 24, 10),
    ("g2", 2024, 2, "2024-09-15", "B", "C", 21, 7),
    ("g3", 2024, 3, "2024-09-22", "C", "A", 3, 30),
    ("g4", 2024, 4, "2024-09-29", "B", "A", 13, 27),
    ("g5", 2024, 5, "2024-10-06", "A", "C", 35, 0),
    ("g6", 2024, 6, "2024-10-13", "C", "B", 10, 17),
    ("g7", 2024, 7, "2024-10-20", "E", "C", 20, 17),
]
CONFIG = {"home_field_advantage": 30.0, "k_factor": 20.0, "initial_rating": 1500.0, "window": 2}


def synthetic_artifacts() -> PredictionArtifacts:
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(200, 3)) * [0.3, 8.0, 80.0], columns=FEATURE_COLUMNS)
    y = ((X / [0.3, 8.0, 80.0]).sum(axis=1) > 0).astype(int)
    return PredictionArtifacts(
        pipeline=train_pipeline(build_pipeline(), X, y),
        config=CONFIG,
        games=pd.DataFrame(GAMES, columns=COLUMNS),
    )


@pytest.fixture
def load_calls():
    return []


@pytest.fixture
def client(load_calls):
    def loader():
        load_calls.append(1)
        return synthetic_artifacts()

    # The context manager runs the app's startup (lifespan) and shutdown.
    with TestClient(create_app(load_artifacts=loader)) as test_client:
        yield test_client


# --- 200 ---------------------------------------------------------------------


def test_valid_matchup_returns_200_and_response_schema(client):
    response = client.post("/predict", json={"home_team": "A", "away_team": "C"})

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {
        "home_team",
        "away_team",
        "home_win_probability",
        "away_win_probability",
        "predicted_winner",
        "features",
    }
    assert set(body["features"]) == set(FEATURE_COLUMNS)
    PredictResponse.model_validate(body)  # raises if the shape drifts
    assert (body["home_team"], body["away_team"]) == ("A", "C")
    assert 0.0 < body["home_win_probability"] < 1.0
    assert body["home_win_probability"] + body["away_win_probability"] == pytest.approx(1.0)
    assert body["predicted_winner"] == "A"


# --- error mapping -----------------------------------------------------------


def test_unknown_team_returns_404_naming_it(client):
    response = client.post("/predict", json={"home_team": "A", "away_team": "ZZZ"})

    assert response.status_code == 404
    assert "ZZZ" in response.json()["detail"]


def test_same_team_returns_400(client):
    response = client.post("/predict", json={"home_team": "B", "away_team": "B"})

    assert response.status_code == 400
    assert "B" in response.json()["detail"]


@pytest.mark.parametrize(
    "payload, missing",
    [({"home_team": "A"}, "away_team"), ({"away_team": "A"}, "home_team"), ({}, "home_team")],
)
def test_missing_field_returns_standard_422(client, payload, missing):
    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    errors = response.json()["detail"]
    assert isinstance(errors, list)  # FastAPI's validation-error format
    assert any(err["loc"] == ["body", missing] and err["type"] == "missing" for err in errors)


def test_non_string_team_returns_422(client):
    response = client.post("/predict", json={"home_team": 123, "away_team": "A"})

    assert response.status_code == 422


def test_team_without_enough_history_returns_422_naming_it(client):
    response = client.post("/predict", json={"home_team": "E", "away_team": "A"})

    assert response.status_code == 422
    assert "E" in response.json()["detail"]


# --- startup -----------------------------------------------------------------


def test_artifacts_load_once_at_startup_into_app_state(client, load_calls):
    client.post("/predict", json={"home_team": "A", "away_team": "C"})
    client.post("/predict", json={"home_team": "B", "away_team": "C"})

    assert load_calls == [1]
    assert isinstance(client.app.state.artifacts, PredictionArtifacts)
