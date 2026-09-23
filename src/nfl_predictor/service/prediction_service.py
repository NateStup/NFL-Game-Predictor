"""Prediction service: load artifacts once, score a future matchup."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline


class PredictionError(Exception):
    """Base class for request-level prediction failures."""


class UnknownTeamError(PredictionError):
    def __init__(self, teams: list[str]):
        self.teams = teams
        super().__init__(f"Unknown team(s): {', '.join(teams)}")


class SameTeamError(PredictionError):
    def __init__(self, team: str):
        self.team = team
        super().__init__(f"home_team and away_team are both {team}")


class InsufficientHistoryError(PredictionError):
    def __init__(self, teams: list[str]):
        self.teams = teams
        super().__init__(f"Not enough game history for: {', '.join(teams)}")


@dataclass(frozen=True)
class PredictionArtifacts:
    pipeline: Pipeline
    config: dict
    games: pd.DataFrame


def load_prediction_artifacts(
    model_path: Path, config_path: Path, games_path: Path
) -> PredictionArtifacts:
    raise NotImplementedError


def predict_matchup(
    artifacts: PredictionArtifacts, home_team: str, away_team: str
) -> dict:
    raise NotImplementedError
