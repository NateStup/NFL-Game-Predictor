"""HTTP routes: parse, delegate to the service, map errors to status codes."""

from fastapi import APIRouter, HTTPException, Request, status

from nfl_predictor.api.schemas import LandingResponse, PredictRequest, PredictResponse
from nfl_predictor.service.prediction_service import (
    InsufficientHistoryError,
    SameTeamError,
    UnknownTeamError,
    predict_matchup,
)

router = APIRouter()

# Static orientation text. The seasons must match the served artifacts
# (scripts/build_features.py TRAIN_SEASONS / TEST_SEASONS).
LANDING = LandingResponse(
    description=(
        "Predicts the winner of an NFL regular-season game from each team's "
        "recent form (last 8 games) and Elo rating."
    ),
    snapshot_notice=(
        "Predictions reflect each team's state as of the end of the 2024 "
        "regular season, not the current season. This is a frozen snapshot "
        "that does not update as new games are played."
    ),
    training=(
        "Logistic regression trained on 2015-2023 regular-season games and "
        "tested on the 2024 regular season."
    ),
    docs="/docs for interactive documentation, where you can try /predict.",
    predict='POST /predict with {"home_team": "KC", "away_team": "BUF"}.',
)


@router.get("/", response_model=LandingResponse)
def landing() -> LandingResponse:
    return LANDING


@router.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest, request: Request) -> PredictResponse:
    try:
        result = predict_matchup(request.app.state.artifacts, body.home_team, body.away_team)
    except UnknownTeamError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except SameTeamError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except InsufficientHistoryError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc
    return PredictResponse.model_validate(result)
