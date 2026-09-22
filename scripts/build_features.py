"""Build the train/test feature tables from 2015-2024 regular-season games.

HFA is derived from the training seasons only; the sequential features
(rolling stats, Elo) run over the full chronological 2015-2024 sequence.

Usage: python scripts/build_features.py
Importable: build_feature_splits() returns the same tables without printing.
"""

from dataclasses import dataclass

import pandas as pd

from nfl_predictor.data.fetch import fetch_schedules
from nfl_predictor.data.transforms import (
    chronological_split,
    drop_ties,
    home_baseline_accuracy,
    normalize_franchises,
    regular_season_only,
)
from nfl_predictor.predictor.elo import derive_home_field_advantage
from nfl_predictor.predictor.features import FEATURE_COLUMNS, build_feature_table

TRAIN_SEASONS = list(range(2015, 2024))
TEST_SEASONS = [2024]
WINDOW = 8
K_FACTOR = 20.0
INITIAL_RATING = 1500.0
ROLLING_COLUMNS = ["rolling_win_pct_diff", "rolling_point_diff"]


@dataclass(frozen=True)
class FeatureSplits:
    train: pd.DataFrame  # feature table, warmup rows dropped
    test: pd.DataFrame  # feature table
    train_games: pd.DataFrame  # processed games (with scores), train seasons
    test_games: pd.DataFrame  # processed games (with scores), test seasons
    hfa: float
    train_home_rate: float
    n_fetched: int
    n_games: int
    dropped_by_season: dict[int, int]


def build_feature_splits() -> FeatureSplits:
    # (a) fetch, unify relocated franchises, keep decided regular-season games
    fetched = fetch_schedules(TRAIN_SEASONS + TEST_SEASONS)
    games = drop_ties(regular_season_only(normalize_franchises(fetched)))
    games = games.sort_values("gameday", kind="mergesort")

    # (b) split the raw games to isolate training rows
    train_games, test_games = chronological_split(games, TRAIN_SEASONS, TEST_SEASONS)

    # (c) HFA from the training seasons only
    train_home_rate = home_baseline_accuracy(train_games)
    hfa = derive_home_field_advantage(train_home_rate)

    # (d) features over the full chronological sequence
    table = build_feature_table(
        games,
        home_field_advantage=hfa,
        window=WINDOW,
        k_factor=K_FACTOR,
        initial_rating=INITIAL_RATING,
    )

    # (e) re-split the feature table by season
    train, test = chronological_split(table, TRAIN_SEASONS, TEST_SEASONS)

    # (f) drop warmup rows; they must all be training rows
    train_warmup = train[ROLLING_COLUMNS].isna().any(axis=1)
    test_warmup = test[ROLLING_COLUMNS].isna().any(axis=1)
    if test_warmup.any():
        raise RuntimeError(
            f"{test_warmup.sum()} test rows lack rolling history; expected 0"
        )
    dropped_by_season = train.loc[train_warmup, "season"].value_counts().sort_index()
    train = train[~train_warmup]

    # (g) no nulls may remain anywhere
    for name, split in [("train", train), ("test", test)]:
        nulls = int(split.isna().sum().sum())
        if nulls:
            raise RuntimeError(f"{nulls} nulls remain in {name} after warmup drop")

    return FeatureSplits(
        train=train,
        test=test,
        train_games=train_games,
        test_games=test_games,
        hfa=hfa,
        train_home_rate=train_home_rate,
        n_fetched=len(fetched),
        n_games=len(games),
        dropped_by_season=dropped_by_season.to_dict(),
    )


def main() -> None:
    s = build_feature_splits()
    print(f"Games fetched:               {s.n_fetched}")
    print(f"Regular-season, tie-free:    {s.n_games}")
    print(f"Train home win rate:         {s.train_home_rate:.4f} (2015-2023)")
    print(f"HFA (train-derived):         {s.hfa:.2f} Elo points")
    print(f"Warmup rows dropped:         {sum(s.dropped_by_season.values())} train, 0 test")
    print(f"  by season:                 {s.dropped_by_season}")
    print("Nulls remaining:             0 train, 0 test")
    # (h) final shapes
    print(f"Train rows:                  {len(s.train)}")
    print(f"Test rows:                   {len(s.test)}")
    print(f"Feature columns:             {FEATURE_COLUMNS}")
    print(f"Train shape / test shape:    {s.train.shape} / {s.test.shape}")


if __name__ == "__main__":
    main()
