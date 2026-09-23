"""Sequential Elo ratings (pure functions, no I/O)."""

import math

import pandas as pd


def derive_home_field_advantage(home_win_rate: float) -> float:
    """Elo points of home advantage implied by a historical home win rate.

    Inverts the Elo win-probability curve: the rating edge that makes two
    otherwise equal teams produce `home_win_rate` for the home side.
    """
    if not 0 < home_win_rate < 1:
        raise ValueError(f"home_win_rate must be in (0, 1), got {home_win_rate}")
    return -400 * math.log10((1 / home_win_rate) - 1)


def expected_home_win_prob(
    home_rating: float, away_rating: float, home_field_advantage: float
) -> float:
    """Probability the home team wins, with HFA added to the home rating."""
    edge = (home_rating + home_field_advantage) - away_rating
    return 1 / (1 + 10 ** (-edge / 400))


def update_ratings(
    home_rating: float,
    away_rating: float,
    home_won: bool,
    home_field_advantage: float,
    k_factor: float,
) -> tuple[float, float]:
    """Apply one game's result; returns (new_home_rating, new_away_rating).

    Zero-sum: the away team loses exactly what the home team gains.
    """
    expected = expected_home_win_prob(home_rating, away_rating, home_field_advantage)
    delta = k_factor * (float(home_won) - expected)
    return home_rating + delta, away_rating - delta


def _run_elo(
    games_df: pd.DataFrame,
    initial_rating: float,
    k_factor: float,
    home_field_advantage: float,
) -> tuple[dict, dict[str, float]]:
    """Walk games in date order once.

    Returns (pre-game home-minus-away diff keyed by games_df index label,
    each team's rating after its final game). Ratings are one continuous
    sequence per team across all seasons (no reset at week 1); unseen teams
    start at initial_rating. Games on the same date keep their input order
    (stable sort); a team plays at most once per date, so that order cannot
    change any rating.
    """
    if not games_df.index.is_unique:
        raise ValueError("games_df index must be unique to align features")
    if games_df["gameday"].isna().any():
        raise ValueError("gameday contains nulls; cannot order games")

    order = pd.to_datetime(games_df["gameday"]).sort_values(kind="mergesort").index
    ordered = games_df.loc[order]

    ratings: dict[str, float] = {}
    diffs = {}
    for idx, home, away, home_score, away_score in zip(
        ordered.index,
        ordered["home_team"],
        ordered["away_team"],
        ordered["home_score"],
        ordered["away_score"],
    ):
        home_rating = ratings.get(home, initial_rating)
        away_rating = ratings.get(away, initial_rating)
        diffs[idx] = home_rating - away_rating
        ratings[home], ratings[away] = update_ratings(
            home_rating,
            away_rating,
            home_score > away_score,
            home_field_advantage,
            k_factor,
        )
    return diffs, ratings


def compute_elo_ratings(
    games_df: pd.DataFrame,
    initial_rating: float,
    k_factor: float,
    home_field_advantage: float,
) -> pd.Series:
    """Pre-game Elo differential (home rating - away rating) for every game.

    Each game's differential uses ratings from BEFORE that game, then the
    result updates both teams. HFA drives the win probability inside updates
    only; it is not added to the returned differential.
    """
    diffs, _ = _run_elo(games_df, initial_rating, k_factor, home_field_advantage)
    return pd.Series(diffs, name="elo_diff", dtype=float).reindex(games_df.index)


def latest_team_elo_ratings(
    games_df: pd.DataFrame,
    initial_rating: float,
    k_factor: float,
    home_field_advantage: float,
) -> pd.DataFrame:
    """Each team's CURRENT rating: after its most recent game's update.

    Unlike compute_elo_ratings (pre-game, for training rows), this includes
    every game in games_df, for rating a future matchup. Returns one row per
    team (sorted): team, elo_rating.
    """
    _, ratings = _run_elo(games_df, initial_rating, k_factor, home_field_advantage)
    return (
        pd.Series(ratings, name="elo_rating", dtype=float)
        .sort_index()
        .rename_axis("team")
        .reset_index()
    )
