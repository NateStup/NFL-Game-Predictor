"""Rolling team-form features (pure functions, no I/O).

Each feature is home-minus-away, computed from each team's trailing window of
PRIOR games only. A team's game log is one continuous sequence across all
seasons in the input; windows never reset at week 1. Games where either team
has fewer than `window` prior games are NaN (warmup), by design.
"""

import pandas as pd


def _to_team_games(games_df: pd.DataFrame) -> pd.DataFrame:
    """Reshape one-row-per-game into one-row-per-team-per-game.

    Returns columns: game_idx (the games_df index label), team, gameday,
    is_home, won (1.0/0.0; a tie counts as not won), point_diff (team score
    minus opponent score). Sorted by team, then date.
    """
    home_score = games_df["home_score"].to_numpy()
    away_score = games_df["away_score"].to_numpy()
    gameday = pd.to_datetime(games_df["gameday"]).to_numpy()

    home = pd.DataFrame(
        {
            "game_idx": games_df.index,
            "team": games_df["home_team"].to_numpy(),
            "gameday": gameday,
            "is_home": True,
            "won": (home_score > away_score).astype(float),
            "point_diff": home_score - away_score,
        }
    )
    away = pd.DataFrame(
        {
            "game_idx": games_df.index,
            "team": games_df["away_team"].to_numpy(),
            "gameday": gameday,
            "is_home": False,
            "won": (away_score > home_score).astype(float),
            "point_diff": away_score - home_score,
        }
    )
    long_df = pd.concat([home, away], ignore_index=True)
    return long_df.sort_values(["team", "gameday"], kind="mergesort").reset_index(
        drop=True
    )


def _rolling_team_stat(
    long_df: pd.DataFrame, stat_column: str, window: int = 8
) -> pd.Series:
    """Per-team trailing mean of stat_column over the `window` PRIOR games.

    Expects long_df sorted by team then date. shift(1) excludes the current
    game's own value; min_periods=window keeps it NaN until a full window of
    prior games exists.
    """
    return long_df.groupby("team")[stat_column].transform(
        lambda s: s.shift(1).rolling(window, min_periods=window).mean()
    )


def _home_minus_away(
    games_df: pd.DataFrame, stat_column: str, window: int
) -> pd.Series:
    if not games_df.index.is_unique:
        raise ValueError("games_df index must be unique to align features")
    long_df = _to_team_games(games_df)
    long_df["rolling"] = _rolling_team_stat(long_df, stat_column, window)
    by_side = long_df.set_index("game_idx").groupby("is_home")["rolling"]
    home = by_side.get_group(True)
    away = by_side.get_group(False)
    return (home - away).reindex(games_df.index)


def rolling_win_pct_diff(games_df: pd.DataFrame, window: int = 8) -> pd.Series:
    """Home team's trailing win% minus away team's, per game."""
    return _home_minus_away(games_df, "won", window).rename("rolling_win_pct_diff")


def rolling_point_diff(games_df: pd.DataFrame, window: int = 8) -> pd.Series:
    """Home team's trailing average point differential minus away team's."""
    return _home_minus_away(games_df, "point_diff", window).rename(
        "rolling_point_diff"
    )


def latest_team_rolling_stats(games_df: pd.DataFrame, window: int = 8) -> pd.DataFrame:
    """Each team's CURRENT form: means over its last `window` games played.

    Unlike the training features, this is unshifted: the most recent game is
    included, because it describes a team heading into a future game rather
    than a game whose result exists. A team with fewer than `window` games
    gets NaN, the same full-window rule the model was trained under.

    Returns one row per team (sorted): team, rolling_win_pct,
    rolling_point_diff.
    """
    long_df = _to_team_games(games_df)

    def last_window_mean(s: pd.Series) -> float:
        return s.iloc[-window:].mean() if len(s) >= window else float("nan")

    by_team = long_df.groupby("team")
    return pd.DataFrame(
        {
            "rolling_win_pct": by_team["won"].agg(last_window_mean),
            "rolling_point_diff": by_team["point_diff"].agg(last_window_mean),
        }
    ).rename_axis("team").reset_index()
