"""Request and response models for the prediction API."""

from pydantic import BaseModel


class LandingResponse(BaseModel):
    """Orientation for someone arriving at the API root."""

    description: str
    snapshot_notice: str
    training: str
    docs: str
    predict: str


class PredictRequest(BaseModel):
    home_team: str
    away_team: str


class MatchupFeatures(BaseModel):
    """Home-minus-away model inputs the prediction was made from."""

    rolling_win_pct_diff: float
    rolling_point_diff: float
    elo_diff: float


class PredictResponse(BaseModel):
    home_team: str
    away_team: str
    home_win_probability: float
    away_win_probability: float
    predicted_winner: str
    features: MatchupFeatures
