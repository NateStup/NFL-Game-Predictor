"""Prediction service: load artifacts once, score a future matchup."""

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from nfl_predictor.predictor.features import FEATURE_COLUMNS
from nfl_predictor.predictor.matchup import build_matchup_features
from nfl_predictor.predictor.team_state import get_team_state_snapshot


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
    """Read the fitted pipeline, training config and processed game history."""
    return PredictionArtifacts(
        pipeline=joblib.load(model_path),
        config=json.loads(Path(config_path).read_text()),
        games=pd.read_parquet(games_path, engine="fastparquet"),
    )


def predict_matchup(
    artifacts: PredictionArtifacts, home_team: str, away_team: str
) -> dict:
    """Home-win probability for a future game between two known teams.

    Team state is rebuilt from the stored game history with the constants the
    model was trained with (artifacts.config), so request-time features match
    training-time features. predicted_winner is the home team on an exact
    50/50.
    """
    if home_team == away_team:
        raise SameTeamError(home_team)

    config = artifacts.config
    snapshot = get_team_state_snapshot(
        artifacts.games,
        home_field_advantage=config["home_field_advantage"],
        window=config["window"],
        k_factor=config["k_factor"],
        initial_rating=config["initial_rating"],
    ).set_index("team")

    unknown = [team for team in (home_team, away_team) if team not in snapshot.index]
    if unknown:
        raise UnknownTeamError(unknown)
    short = [team for team in (home_team, away_team) if snapshot.loc[team].isna().any()]
    if short:
        raise InsufficientHistoryError(short)

    features = build_matchup_features(snapshot.loc[home_team], snapshot.loc[away_team])
    probabilities = artifacts.pipeline.predict_proba(
        pd.DataFrame([features], columns=FEATURE_COLUMNS)
    )[0]
    classes = list(artifacts.pipeline.classes_)
    home_p = float(probabilities[classes.index(1)])
    away_p = float(probabilities[classes.index(0)])

    return {
        "home_team": home_team,
        "away_team": away_team,
        "home_win_probability": home_p,
        "away_win_probability": away_p,
        "predicted_winner": home_team if home_p >= away_p else away_team,
        "features": features,
    }
