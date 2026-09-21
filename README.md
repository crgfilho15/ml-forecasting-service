# Energy Forecasting Service

A 24-hour-ahead electricity consumption forecasting model, served as a containerized REST API. Built as a public, reproducible companion to my applied forecasting work at UTAD (Vine & Wine Portugal), using an open dataset since the original industrial telemetry is proprietary.

## Context

At UTAD, I build hourly electricity-demand forecasting models (LSTM, XGBoost, hybrid architectures) from live winery telemetry, with time-aware validation and hyperparameter optimization. This project replicates that same methodology — end to end, from raw time series to a served prediction endpoint — on a public dataset, so the full approach can be inspected and reproduced by anyone.

## Dataset

- **Source**: [PJM Hourly Energy Consumption](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption) (Kaggle)
- **File used**: `PJME_hourly.csv`
- **Size**: 145,366 hourly records, January 2002 to August 2018
- **Quality**: no missing values in either column

## Exploratory Data Analysis

Before any modeling, the raw series was inspected visually at two time scales:

- **Full series (17 years)**: consumption is broadly stable over time, with no long-term upward or downward trend. Strong annual seasonality is visible (summer peaks — likely air conditioning load — and a milder winter peak). One unexplained sharp drop around 2013 was flagged but not investigated further (see *Limitations*).
- **Two-week zoom (July 2017)**: revealed a clear ~24h cycle, a visible dip around the July 4th holiday (consumption drops on a US national holiday), and an escalating pattern consistent with a building heatwave. The holiday effect was noted but not turned into a feature in this version (see *Limitations*).

## Feature Engineering

- **Cyclical encoding** (sine/cosine) for `hour`, `day_of_week`, and `month` — necessary because raw numeric encoding would treat, e.g., hour 23 and hour 0 as maximally distant, when they are in fact adjacent on the clock.
- **Lag features**: `lag_1h`, `lag_24h`, `lag_168h` (1 hour, 1 day, and 1 week prior consumption).
- **Rolling feature**: `rolling_mean_24h`, a 24-hour trailing average.
- **Missing values**: the first 168 rows (where lag features have no history yet) were dropped. They were never imputed by backfilling or by a global mean — either would leak future information into the training data.

## Validation Methodology

The train/test split is **chronological** (80/20), not random. Shuffling before splitting a time series creates data leakage: the model would end up training on data that occurred *after* some of the examples it's meant to predict. This is the same time-aware validation principle used in the UTAD forecasting work.

## Modeling and a Critical Finding

The first version of this project trained an XGBoost model to predict consumption **1 hour ahead**, using `lag_1h` as a feature. It performed extremely well (MAPE 0.98%) — too well. Inspecting feature importance revealed that `lag_1h` alone accounted for **94.4%** of the model's decisions. This is a horizon-leakage problem: a 1-hour-ahead model has no practical use for operational planning (e.g., day-ahead scheduling), and relying so heavily on the immediately preceding value meant the model was closer to sophisticated persistence than genuine forecasting for any useful horizon.

**The model was redesigned to forecast 24 hours ahead**, removing `lag_1h` entirely, since that value would not actually be available at the time a real 24h-ahead prediction would be made. `lag_24h`, `lag_168h`, and `rolling_mean_24h` remain valid, because they represent information that *would* genuinely be known 24 hours before the target timestamp.

**XGBoost hyperparameters**: `n_estimators=500`, `learning_rate=0.05`, `max_depth=6`, `random_state=42` — a moderate-depth, low-learning-rate, high-tree-count configuration chosen to favor stable generalization over aggressive fitting, with a fixed seed for reproducibility.

## Results

| Model | Horizon | MAE (MW) | RMSE (MW) | MAPE (%) |
|---|---|---|---|---|
| Naive Persistence (predict = previous hour) | 1h | 1070.05 | 1373.87 | 3.48 |
| XGBoost (1h horizon, with `lag_1h`) | 1h | 308.92 | 417.32 | 0.98 |
| Naive Seasonal (predict = same hour, previous day) | 24h | 2184.22 | 3009.11 | 6.93 |
| **XGBoost (24h horizon, corrected)** | **24h** | **1157.40** | **1548.58** | **3.67** |

The final 24h-ahead model reduces MAPE by roughly **47%** relative to the naive seasonal baseline — the correct comparison for its horizon — while the 1h-ahead experiment (kept in the notebook for transparency) shows that even the leaking model added real value over simple persistence, despite the horizon problem.

## API

A FastAPI service exposes the corrected 24h-ahead model.

**`GET /health`** — liveness check.

**`POST /predict`** — accepts:

```json
{
  "target_datetime": "2018-07-15T14:00:00",
  "lag_24h": 42000,
  "lag_168h": 40500,
  "rolling_mean_24h": 35000
}
```

and returns:

```json
{
  "predicted_mw": 40416.97
}
```

Cyclical time features (`hour_sin/cos`, `day_of_week_sin/cos`, `month_sin/cos`) are derived internally from `target_datetime`. The three lag/rolling values are validated as strictly positive via Pydantic (`Field(..., gt=0)`) and rejected with a `422` otherwise.

## Known Limitations & Next Steps

- **Lag values must be supplied by the caller.** The API does not compute `lag_24h`, `lag_168h`, or `rolling_mean_24h` itself, since it has no connection to historical telemetry. In production, these would come from a feature store or a time-series database (e.g., InfluxDB, as used in the UTAD project), not from the request payload.
- **The model is embedded in the Docker image** (`COPY model_24h.pkl`) rather than loaded from an external model registry. This means updating the model requires rebuilding the image. A more mature setup would load it at startup from something like MLflow or a versioned storage bucket.
- **Holiday effects are not modeled**, despite being visible in the EDA (e.g., the July 4th dip). A holiday-calendar feature is a natural next addition.
- **The 2013 anomaly was not investigated.** Worth a follow-up look to confirm whether it reflects a real event or a data quality issue.

## Repository Structure

```
.
├── 01_exploration.ipynb      # EDA, feature engineering, modeling, evaluation
└── api/
    ├── main.py                # FastAPI service
    ├── requirements.txt       # API-only dependencies (hand-written, not pip freeze)
    ├── Dockerfile
    └── model_24h.pkl          # Trained XGBoost model (24h-ahead)
```

`requirements.txt` inside `api/` is deliberately minimal and hand-written — it is not generated via `pip freeze` from the exploration environment, which carries Jupyter and other research-only dependencies that have no place in the served API.

## How to Run

**Locally (API only):**

```bash
cd api
python -m venv venv
source venv/bin/activate  # venv\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt
uvicorn main:app --reload --port 8002
```

**With Docker:**

```bash
cd api
docker build -t energy-forecasting-api .
docker run --rm -p 8002:8002 energy-forecasting-api
```

Then visit `http://127.0.0.1:8002/docs` for interactive API documentation, or `http://127.0.0.1:8002/health` for a liveness check.

## Tech Stack

- **Data & Modeling**: Python, Pandas, NumPy, XGBoost, scikit-learn (metrics)
- **API**: FastAPI, Pydantic, Uvicorn
- **Serving**: Docker
- **Research environment**: Jupyter Notebook
- **Model persistence**: joblib

---

**Author**: Carlos Roberto Souza Garcia Filho — [LinkedIn](https://www.linkedin.com/in/carlosrobertogarcia) · [GitHub](https://github.com/crgfilho15)
