"""FastAPI app: loads prediction artifacts once at startup into app.state.

Run: uvicorn nfl_predictor.api.main:app
"""

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from nfl_predictor.api.routes import router
from nfl_predictor.service.prediction_service import (
    PredictionArtifacts,
    load_prediction_artifacts,
)

# Repo root, for an editable install (pip install -e .): src/nfl_predictor/api/
REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = REPO_ROOT / "models" / "logistic_regression.joblib"
CONFIG_PATH = REPO_ROOT / "models" / "training_config.json"
GAMES_PATH = REPO_ROOT / "data" / "processed_games.parquet"


def load_default_artifacts() -> PredictionArtifacts:
    return load_prediction_artifacts(MODEL_PATH, CONFIG_PATH, GAMES_PATH)


def create_app(
    load_artifacts: Callable[[], PredictionArtifacts] = load_default_artifacts,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.artifacts = load_artifacts()
        yield

    app = FastAPI(title="NFL Predictor", lifespan=lifespan)
    app.include_router(router)
    return app


app = create_app()
