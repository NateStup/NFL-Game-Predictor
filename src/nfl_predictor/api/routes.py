"""HTTP routes: parse, delegate to the service, map errors to status codes."""

from fastapi import APIRouter

from nfl_predictor.api.schemas import PredictRequest, PredictResponse

router = APIRouter()


@router.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    raise NotImplementedError
