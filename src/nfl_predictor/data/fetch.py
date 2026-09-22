"""Thin I/O boundary: the only place that talks to nfl_data_py."""

import nfl_data_py as nfl
import pandas as pd


def fetch_schedules(seasons: list[int]) -> pd.DataFrame:
    """Download NFL schedules/results for the given seasons."""
    return nfl.import_schedules(seasons)
