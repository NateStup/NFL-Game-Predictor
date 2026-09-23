"""HTTP routes: parse, delegate to the service, map errors to status codes."""

from fastapi import APIRouter, HTTPException, Request, status

from nfl_predictor.api.schemas import PredictRequest, PredictResponse
from nfl_predictor.service.prediction_service import (
    InsufficientHistoryError,
    SameTeamError,
    UnknownTeamError,
    predict_matchup,
)

router = APIRouter()


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
