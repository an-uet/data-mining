# Next-hour PM2.5 at Aotizhongxin (Beijing) — NYCU Data Mining HW1

<!-- TODO before publishing: replace every <PLACEHOLDER> (team ID, members, LICENSE holder). -->
| | |
|---|---|
| Course | NYCU Data Mining 2026 — HW1: Linear Regression |
| Team ID | G08 |
| Members | `<Name (student ID)>`, `<Name (student ID)>`, `<Name (student ID)>` |

## Problem
Forecast the PM2.5 concentration (µg/m³) at the Aotizhongxin station in Beijing **one hour ahead** from information available at the current hour, X(t) → PM2.5(t+1).
Data: UCI *Beijing Multi-Site Air-Quality Data* (id 501), 35,064 hourly observations, 2013-03 → 2017-02 (see [`data/README.md`](data/README.md)).
Method: linear models (OLS, Ridge, Lasso, Elastic Net) on engineered features (PM2.5 lags and rolling statistics, calendar, meteorology, interactions), chronological 80/20 split,
expanding-window cross-validation; persistence, Random Forest and Gradient Boosting are baselines only.

## Overall results
The selected model is a Lasso on the engineered feature set.

| Held-out test (µg/m³) | MAE | RMSE | R² |
|---|---:|---:|---:|
| Persistence, PM2.5(t+1) = PM2.5(t) | 10.36 | 19.28 | 0.9481 |
| **Final Lasso** | **10.06** | **18.00** | **0.9547** |

Almost all predictive power comes from the PM2.5 history; the gain over persistence is small but statistically reliable (−0.30 µg/m³ MAE). Details are in the report and in `results/`.

## Run
```
pip install -r requirements.txt          # Python 3.10
python run_all.py --until freeze         # safe check: stops before the held-out test period is opened
python run_all.py                        # full experiment (~7 min); opens the test period and overwrites results/
```

## Repository map
```
run_all.py  requirements.txt  LICENSE
data/raw/             the CSV (+ data/README.md: source, DOI, columns)
src/                  code (config, data, preprocessing, features, models, statistics, evaluation, plotting, experiment)
notebooks/            01 … 08, executed views of the experiment
results/              figures/ tables/ models/ logs/ + final_summary.txt
```

## Notes
`fetch_ucirepo(id=381)` is a different dataset; this project uses UCI id 501. Every opening of the test period is logged (`results/logs/`, `extra_experiments/outputs/logs/`).
Code: MIT (`LICENSE`); data: UCI, DOI 10.24432/C5RK5G (terms on the UCI page). Code written with the help of an AI coding assistant (Claude Code, Anthropic); see the AI-usage statement in the report.
