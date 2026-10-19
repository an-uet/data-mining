# Next-hour PM2.5 at Aotizhongxin (Beijing) — NYCU Data Mining HW1

| | |
|---|---|
| Course | NYCU Data Mining 2026 — HW1: Linear Regression |
| Team ID | G08 |
| Members | Le Thi An - 314540018, Le Quang Thang - 414540003, Pongpat Weesommai - 310540028,  Syahrizal M. N. - 314540039 |

## Problem
Forecast the PM2.5 concentration (µg/m³) at the Aotizhongxin station in Beijing **one hour ahead**, using only information available at the current hour t: X(t) → PM2.5(t+1).
Adjacent hours are almost identical, so the study uses a **chronological** split, and the baseline that matters most is **persistence**, PM2.5(t+1) = PM2.5(t).

## Data
UCI *Beijing Multi-Site Air-Quality Data* (id 501, DOI 10.24432/C5RK5G), station Aotizhongxin: 35,064 hourly rows, 2013-03-01 → 2017-02-28 (PM2.5, PM10, SO2, NO2, CO, O3, temperature, pressure, dew point, rain, wind direction, wind speed).
Train: 28,050 hours (until 2016-05-12). Test: 7,013 hours (2016-05-12 → 2017-02-28). Source and columns: [`data/README.md`](data/README.md).

## Method
- **Leakage-safe pipeline.** The target is never imputed; imputation, scaling, correlation-based feature selection and tuning are fitted inside each fold of a 5-fold expanding-window `TimeSeriesSplit`. 14 automatic leakage checks pass. The design is frozen and fingerprinted (SHA-256) before the test period is opened.
- **Features.** Set A: raw variables. Set B/B′: top-*k* correlated features. **Set C**: PM2.5 lags and shifted rolling statistics, calendar terms, meteorological differences, centred interactions (54 columns).
- **Models.** OLS, Ridge, Lasso and Elastic Net, with the family chosen from properties of the training data (collinearity, many weak predictors). Persistence, Random Forest and Gradient Boosting are baselines only. The final model is a **Lasso** (29 of 54 coefficients non-zero).

## Overall results
Held-out test period (µg/m³); CV = mean over the five training folds. Full tables: `results/tables/`.

| Method | CV MAE | Test MAE | Test RMSE | Test R² |
|---|---:|---:|---:|---:|
| Persistence | 10.69 | 10.36 | 19.28 | 0.9481 |
| Random Forest (raw features) | 11.14 | 10.20 | 18.86 | 0.9503 |
| Gradient Boosting (raw features) | 11.76 | 10.27 | 18.88 | 0.9502 |
| **Lasso, set C (final)** | **10.51** | **10.06** | **18.00** | **0.9547** |

- **The PM2.5 history carries almost everything:** removing it raises the test MAE from 10.06 to 16.25 µg/m³ (+61 %).
- **The gain over persistence is small but reliable:** −0.30 µg/m³ MAE (−2.9 %, Diebold–Mariano p ≈ 4·10⁻⁵), coming from rapid-change hours.
- **Largest errors** are abrupt one-hour jumps (226–271 µg/m³) where the forecast stays close to the last observation.
- **Limitations:** one station and one test window; one-step-ahead only. Random Forest / Gradient Boosting on the same engineered features reach a lower test MAE (9.53 / 9.71), but they are reference models only, since the final predictor must be linear.

## Run
```
pip install -r requirements.txt          # Python 3.10
python run_all.py --until freeze         # safe check: stops before the held-out test period is opened
python run_all.py                        # full experiment; opens the test period and overwrites results/
```

## Repository map
```
run_all.py  requirements.txt  LICENSE
data/raw/             the CSV (+ data/README.md: source, DOI, columns)
src/                  code (config, data, preprocessing, features, models, statistics, evaluation, plotting, experiment)
notebooks/            01 … 08, executed views of the experiment
results/              figures/ tables/ models/ logs/ + final_summary.txt
```


