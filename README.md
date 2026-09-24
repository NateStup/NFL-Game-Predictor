# nfl-predictor

Predicts the winner of an NFL regular-season game from each team's recent
results and Elo rating, and serves the prediction over a FastAPI endpoint.
It trains a logistic regression on 2015-2023 regular-season games and
evaluates it on the 2024 season, which the model never sees during training.
The project is mainly about software engineering practice: a layered
architecture (data → features → training → service → API), tests written
before the code they cover, and honest validation. It isn't meant to show off
predictive modeling.

## Live demo

The API is deployed at <https://nfl-game-predictor.onrender.com/docs>, the
interactive Swagger UI where you can try `POST /predict` directly. It runs on
Render's free tier, which shuts the service down after 15 minutes without
traffic. The first request after an idle period can take up to about a minute
while it starts back up.

## Data source

All data comes from `nfl_data_py.import_schedules()`, which returns one row per
game: teams, date, season, week, game type, site and final score. There is no
play-by-play, player or roster data. Only final scores and dates are needed to
build win/loss and point-differential features, so that's the scope. It keeps
the data layer to a single network call (`src/nfl_predictor/data/fetch.py`),
which keeps the rest of the pipeline small enough to test with hand-built
DataFrames.

## Methodology

### Train/test split

- **Train:** 2015-2023 regular seasons.
- **Test:** the 2024 regular season.

The split is by season, never random. A random split would put 2024 games in
training and 2019 games in testing, so the model would learn from games that
happen after the ones it's scored on. That future information makes the test
score look better than anything achievable in real use, where only past games
are available. `chronological_split` raises an error if the train and test
seasons overlap.

### Features

Each feature is **home team minus away team**. A positive value means the home
team is ahead on that measure.

- **Rolling win % differential:** each team's win rate over its previous 8
  games.
- **Rolling point differential:** each team's average scoring margin over its
  previous 8 games.
- **Elo differential:** each team's Elo rating going into the game. Ratings
  start at 1500 and update after every game with K = 20. The home-field
  advantage used in the Elo win probability (33.63 Elo points) comes from the
  2015-2023 home win rate (0.5482) only.

There are three features. Home-field advantage is not a separate input. Every
row is framed from the home team's side, so a home indicator would equal 1 on
every row and duplicate the logistic regression's intercept, which already
captures the home edge. Home advantage also enters through the Elo updates, as
described above.

### Leakage guardrails

- **Shift-by-one on rolling stats.** A team's rolling window for a game covers
  only the games before it (`shift(1)` before `rolling(8)`). The game's own
  result is never part of its features. A game is also dropped from training
  until both teams have 8 prior games (131 games, all in 2015).
- **Pre-game Elo.** The Elo differential for a game is taken *before* that
  game's result updates either team's rating.
- **No season reset.** Rolling windows and Elo ratings run continuously
  across seasons. Week 1 of a season uses the end of the previous season,
  which is real, already-known information. This also means 2024 test games
  have full rolling windows from the start.
- **Train-only constants.** The Elo home-field advantage comes from training
  seasons only, and the logistic regression's feature scaler is fit on
  training rows only.
- **Relocated franchises (a real bug, caught and fixed).** In the raw data the
  Rams, Chargers and Raiders appear under their old abbreviations (`STL`,
  `SD`, `OAK`) before relocating (`LA`, `LAC`, `LV`). At first the pipeline
  treated these as different teams. Each relocated franchise's history was
  split in two, and the "new" team restarted with a 1500 Elo rating and an
  empty rolling window, as if it were an expansion team. `normalize_franchises`
  now maps the old abbreviations to the current ones before any feature is
  computed. A test asserts that the processed data contains exactly the 32
  current franchises.

### Regular season only

Playoff games are removed (120 games in 2015-2024). The Super Bowl is played at
a neutral site, so the "home team" there is just a label and breaks the
meaning of the home-minus-away framing. Filtering to `game_type == "REG"`
removes the Super Bowl and the rest of the playoffs. It does **not** remove
regular-season neutral-site games such as international games. There are 42 of
these in 2015-2024, and they are still in the data with a nominal home team
(see Known limitations).

### Ties dropped

9 of the 2,623 regular-season games in 2015-2024 ended in a tie (about 0.3%).
The target is binary (did the home team win?), and a tie is neither outcome, so
ties are removed rather than counted as home losses.

## Results

All numbers below come from `scripts/train_logistic_regression.py` and
`scripts/train_random_forest.py`. They cover 2,211 training games (2015-2023,
after dropping the 131 warmup games) and 272 test games (2024). Rerunning the
scripts reproduces them (`random_state=42`).

### Baseline

Always picking the home team is correct in **53.31%** of 2024 games. That's
the number to beat.

### Logistic regression (the model the API serves)

| Metric (2024 test, positive class = home win) | Value  |
|-----------------------------------------------|--------|
| Accuracy                                      | 0.6875 |
| Precision                                     | 0.6899 |
| Recall                                        | 0.7517 |
| F1                                            | 0.7195 |

Accuracy is 15.4 points above the baseline (187 of 272 correct, against 145).
The model and the baseline disagree on 114 games. The model is right on 78 of
them and the baseline on 36. An exact McNemar test on those counts gives
p ≈ 0.0001, so on this season the margin is unlikely to be noise.

### Random forest (rejected)

An unconstrained random forest (100 trees, default depth) was trained on the
same splits as a comparison:

| Accuracy            | Value  |
|---------------------|--------|
| Train (2015-2023)   | 1.0000 |
| Test (2024)         | 0.5956 |

The forest **overfits**. It classifies every training game correctly but only
59.6% of test games. That's 9.2 points below the logistic regression and 6.25
points above the baseline. With three noisy features and about 2,200 rows,
fully grown trees memorize the training games instead of learning a pattern
that holds up. The logistic regression is simpler, scores higher on the test
set, and has coefficients that can be read directly. It is the model the API
uses.

### How far to trust a single test season

The 2024 figure comes from one season of 272 games. Some seasons are more
predictable than others, so one season's accuracy can be higher or lower than
the model's typical accuracy. `scripts/walk_forward_validation.py` gives a
steadier estimate. For each season from 2019 to 2023, it trains a fresh model
on every earlier season and tests on that season. It uses only 2015-2023 data
and never loads 2024. Each fold also derives its own Elo home-field advantage
from its own training seasons.

| Train     | Test | Games | Home baseline | Accuracy | vs baseline |
|-----------|------|-------|---------------|----------|-------------|
| 2015-2018 | 2019 | 255   | 0.5176        | 0.6353   | +0.1176     |
| 2015-2019 | 2020 | 255   | 0.4980        | 0.6431   | +0.1451     |
| 2015-2020 | 2021 | 271   | 0.5166        | 0.6310   | +0.1144     |
| 2015-2021 | 2022 | 269   | 0.5613        | 0.6394   | +0.0781     |
| 2015-2022 | 2023 | 272   | 0.5551        | 0.6176   | +0.0625     |

Across the 5 folds (± is the sample standard deviation):

- Accuracy: **0.6333 ± 0.0099**
- Home baseline: 0.5298 ± 0.0272
- Mean margin over baseline: **+0.1035**
- **The model beats its own season's baseline in 5 of 5 folds.**

The 2024 test accuracy of 0.6875 is 4.4 to 7.0 points above every walk-forward
fold, and 5.4 points above the walk-forward mean. The 2024 home baseline
(0.5331) is in line with the other seasons, so 2024 was most likely an
easier-than-usual season to predict from recent form and Elo. For how well the
model should do in a typical season, use the walk-forward mean of about 63%,
not 69%.

## API usage

Start the server (after the setup steps below):

```bash
uvicorn nfl_predictor.api.main:app
```

The app loads the saved model, training config and processed game history
once at startup. Interactive docs are at <http://127.0.0.1:8000/docs>.

### `GET /`

The root URL returns a short JSON orientation for someone who lands on the
deployed service with no other context. It says what the API predicts, gives
the frozen-snapshot notice (predictions reflect each team as of the end of the
2024 regular season), summarizes the training and test seasons, and points to
`/docs` and `POST /predict`. Response from the running server:

```bash
curl http://127.0.0.1:8000/
```

```json
{
  "description": "Predicts the winner of an NFL regular-season game from each team's recent form (last 8 games) and Elo rating.",
  "snapshot_notice": "Predictions reflect each team's state as of the end of the 2024 regular season, not the current season. This is a frozen snapshot that does not update as new games are played.",
  "training": "Logistic regression trained on 2015-2023 regular-season games and tested on the 2024 regular season.",
  "docs": "/docs for interactive documentation, where you can try /predict.",
  "predict": "POST /predict with {\"home_team\": \"KC\", \"away_team\": \"BUF\"}."
}
```

### `POST /predict`

Request body: team abbreviations as used by `nfl_data_py` (for example `KC`,
`BUF`, `PHI`, `LV`).

```json
{"home_team": "PHI", "away_team": "DAL"}
```

Response: win probabilities for both sides, the predicted winner, and the
home-minus-away features the prediction was made from. Here is a real
response from the running server:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"home_team": "PHI", "away_team": "DAL"}'
```

```json
{
  "home_team": "PHI",
  "away_team": "DAL",
  "home_win_probability": 0.7702453491437541,
  "away_win_probability": 0.2297546508562459,
  "predicted_winner": "PHI",
  "features": {
    "rolling_win_pct_diff": 0.375,
    "rolling_point_diff": 15.5,
    "elo_diff": 101.79700089667631
  }
}
```

The predicted winner is the home team when the probability is exactly 50/50.

### Errors

| Status | When | Example `detail` |
|--------|------|------------------|
| 404 | A team abbreviation isn't in the game history. Pre-relocation codes (`STL`, `SD`, `OAK`) also get a 404, because the history uses current codes. | `Unknown team(s): XYZ` |
| 400 | `home_team` and `away_team` are the same. | `home_team and away_team are both KC` |
| 422 | A team has fewer than 8 games of history, so its rolling features can't be computed. | `Not enough game history for: ...` |

All 32 current teams have far more than 8 games in the stored history, so the
insufficient-history 422 can't be triggered with the current data. It's
covered by unit tests. FastAPI also returns 422 for a malformed body, such as a
missing field, with its standard validation-error format.

## Known limitations

- **The model only knows scores and results.** It has no data on injuries,
  weather, rosters, coaching changes or quarterback play. A team that loses its
  starting quarterback looks the same to the model as it did the week before.
- **Meaningless late-season games distort recent form.** Kansas City had
  clinched the AFC's top seed before week 18 of 2024 and rested its starters,
  losing 38-0 at Denver. That one game is 1/8 of KC's rolling window and pulls
  its 8-game average scoring margin down to +0.125. Buffalo's average is +8.75.
  Asked for KC at home against BUF, the API returns a home-win probability of
  0.490, essentially a coin flip that slightly favors Buffalo. It does so even
  though KC has the higher Elo rating (+49) and home field. The number reflects
  a game KC chose not to try to win, not KC's actual strength.
- **Predictions are a snapshot.** Team state comes from the stored game
  history, which ends with the 2024 regular season. Every prediction reflects
  each team as of its last 2024 game. Nothing updates automatically as new
  games are played.
- **Ties are dropped, not modeled.** The model can only predict a home win or
  a home loss.
- **Neutral-site regular-season games keep a nominal home team.** The 42
  international and other neutral-site regular-season games in the data are
  treated as home games for whichever team is listed as home.
- **Elo has no margin-of-victory adjustment.** A 1-point win and a 30-point
  win update ratings equally. This was a deliberate scope decision, not an
  oversight: scoring margin already enters through the rolling point
  differential feature.

## Setup

Requires Python 3.11+. Run all commands from the repository root.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows (Git Bash): source .venv/Scripts/activate
                                   # Windows (PowerShell): .venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

`pyproject.toml` splits the dependencies into three levels:

| Install | Adds | Needed for |
|---------|------|------------|
| `pip install -e .` | fastapi, uvicorn, pydantic, pandas, numpy, scikit-learn, joblib, fastparquet | Running the API from the committed artifacts. This is all the deployed service installs. |
| `pip install -e ".[train]"` | nfl_data_py | Running the scripts in `scripts/`, which download schedule data. |
| `pip install -e ".[dev]"` | Everything in `[train]`, plus pytest, pytest-cov, httpx2 | Running the test suite. |

`.[dev]` includes everything, so use it for development. The install has to be
editable (`-e`), because the API finds `models/` and `data/` relative to the
source checkout.

### Tests

```bash
pytest                        # full suite, including integration tests that download data
pytest -m "not integration"   # unit tests only, no network access
```

### Build features, train, serve

The three files the API reads (`data/processed_games.parquet`,
`models/logistic_regression.joblib`, `models/training_config.json`) are
committed, so a fresh clone can start the API straight away with the base
install. The scripts rebuild those files from scratch. They need `[train]`,
and each one downloads schedules with `nfl_data_py`. Run them in this order:

```bash
python scripts/build_features.py             # writes data/processed_games.parquet
python scripts/train_logistic_regression.py  # writes models/logistic_regression.joblib and models/training_config.json
python scripts/train_random_forest.py        # comparison only; needs the logistic regression artifact
python scripts/walk_forward_validation.py    # optional, validation only; not needed before starting the API
uvicorn nfl_predictor.api.main:app           # serves on http://127.0.0.1:8000
```

`scripts/compute_baseline.py` separately prints the game counts and the 2024
home-team baseline.

## Deployment

The live demo runs on Render's free tier. The service is defined in
[`render.yaml`](render.yaml) at the repository root. It installs only the base
dependencies (`pip install -e .`) and starts
`uvicorn nfl_predictor.api.main:app --host 0.0.0.0 --port $PORT`.
`.python-version` pins Python 3.11.9. At startup the service reads the three
committed artifacts (`data/processed_games.parquet`,
`models/logistic_regression.joblib`, `models/training_config.json`) and never
fetches data or retrains. It therefore doesn't depend on `nfl_data_py` or its
data source being reachable when the service is deployed or restarted.
