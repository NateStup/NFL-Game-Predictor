"""Compute the always-pick-home-team baseline on the 2024 regular season.

Usage: python scripts/compute_baseline.py
"""

from nfl_predictor.data.fetch import fetch_schedules
from nfl_predictor.data.transforms import (
    chronological_split,
    drop_ties,
    home_baseline_accuracy,
    regular_season_only,
)

TRAIN_SEASONS = list(range(2015, 2024))
TEST_SEASONS = [2024]


def main() -> None:
    games = fetch_schedules(TRAIN_SEASONS + TEST_SEASONS)
    regular = regular_season_only(games)
    no_ties = drop_ties(regular)
    train, test = chronological_split(
        no_ties, train_seasons=TRAIN_SEASONS, test_seasons=TEST_SEASONS
    )

    print(f"Total games fetched:    {len(games)}")
    print(f"Playoff games dropped:  {len(games) - len(regular)}")
    print(f"Ties dropped:           {len(regular) - len(no_ties)}")
    print(f"Train rows (2015-2023): {len(train)}")
    print(f"Test rows (2024):       {len(test)}")
    print(f"Home baseline accuracy: {home_baseline_accuracy(test):.4f}")


if __name__ == "__main__":
    main()
