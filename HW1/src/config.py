"""Central configuration of the Beijing PM2.5 (Aotizhongxin) HW1 experiment.

Everything that is a *decision* of the experiment lives here, so that it can be audited (and pre-registered) in one place.
"""
from pathlib import Path

import numpy as np

# ------------------------------------------------------------------ paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RESULTS = PROJECT_ROOT / "results"
FIGURES, TABLES, MODELS, LOGS = (RESULTS / "figures", RESULTS / "tables", RESULTS / "models", RESULTS / "logs")

# ------------------------------------------------------------------ data source (documented)
STATION = "Aotizhongxin"
RAW_FILENAME = "PRSA_Data_Aotizhongxin_20130301-20170228.csv"
UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/501/beijing+multi+site+air+quality+data.zip"
UCI_NESTED_ZIP = "PRSA2017_Data_20130301-20170228.zip"
SOURCE_INFO = {
    "name": "Beijing Multi-Site Air-Quality Data (UCI Machine Learning Repository, id 501)",
    "doi": "10.24432/C5RK5G",
    "url": "https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data",
    "paper": "Zhang, S., Guo, B., Dong, A., He, J., Xu, Z., Chen, S.X. (2017) Cautionary tales on air-quality improvement in Beijing. "
             "Proceedings of the Royal Society A 473, 20170457.",
    "note": "The snippet fetch_ucirepo(id=381) returns a DIFFERENT dataset ('Beijing PM2.5', 43,824 rows, US-embassy station, columns "
            "cbwd/Iws/Is/Ir, no PM10/SO2/NO2/CO/O3/RAIN/WSPM). The Aotizhongxin file belongs to id 501, which ucimlrepo cannot import.",
}
EXPECTED_ROWS = 35064
EXPECTED_SHA256 = "a94fbcfc71708b6ffdc033360163d91efb7f675e496f6b5860929aa96b96351b"

# ------------------------------------------------------------------ reproducibility
SEED = 42
N_JOBS = max(1, min(8, (__import__("os").cpu_count() or 1)))

# ------------------------------------------------------------------ task definition
RAW_TARGET = "PM2.5"          # PM2.5 at the current hour t (raw column)
TARGET = "target"             # PM2.5 at t+1  (= RAW_TARGET.shift(-1))
CURRENT = "PM25_t"            # current PM2.5(t): an allowed feature at prediction time t
HORIZON_HOURS = 1
UNIT = "ug/m3"

# ------------------------------------------------------------------ split / CV
TEST_FRACTION = 0.20          # latest 20 % of the hourly timeline = held-out test
N_SPLITS = 5                  # expanding-window TimeSeriesSplit inside the training part

# ------------------------------------------------------------------ missing-value policy
LOCF_LIMIT = 3                # PM2.5 *history* features: causal last-observation-carried-forward over gaps <= 3 h (never the target)

# ------------------------------------------------------------------ feature engineering windows (hours)
LAGS = (1, 3, 6, 12, 24, 48, 168)
ROLL_MEAN_W = (3, 6, 12, 24)
ROLL_STD_W = (6, 24)
WARMUP_HOURS = max(LAGS)      # the first 168 hours have no complete history

POLLUTANTS = ["PM10", "SO2", "NO2", "CO", "O3"]
METEO_NUMERIC = ["TEMP", "PRES", "DEWP", "RAIN", "WSPM"]
WIND_DIR = "wd"
WD_LEVELS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]   # compass order (reference level = N)

# a derived quantity (Group 4) is kept only if |Spearman| with the target OR with the next-hour change reaches this value (training rows only)
DERIVED_MIN_ABS_RHO = 0.10

# ------------------------------------------------------------------ hyper-parameter grids (as specified, upper edge may be extended once)
RIDGE_ALPHAS = np.logspace(-3, 3, 13)
LASSO_ALPHAS = np.logspace(-4, 1, 10)
ENET_ALPHAS = np.logspace(-4, 2, 10)
ENET_L1 = [0.1, 0.3, 0.5, 0.7, 0.9]
TOPK_VALUES = [5, 10, 15, 20]
TOPK_METHODS = ["pearson", "spearman"]
PRIMARY_METRIC = "mae"        # model selection metric (CV MAE), as required

# ------------------------------------------------------------------ pre-registered decision rules
HYP_MIN_REL_INCREASE = 0.01   # removing a group must raise mean CV-MAE by >= 1 % ...
HYP_MIN_FOLDS = 4             # ... and be worse in >= 4 of the 5 folds (predictive criterion)
HYP_ALPHA = 0.05              # joint Wald test (HAC) p-value (inferential criterion)
TIE_SE = 1.0                  # model families within 1 SE of the best CV-MAE are "tied"
# data properties (measured on the TRAINING design matrix of Set C) that drive the choice among tied families
PROP_VIF = 10.0               # max VIF above this -> strong multicollinearity
PROP_CLUSTER_RHO = 0.80       # |rho| >= this defines a correlated group
PROP_MIN_CLUSTERS = 2         # >= this many correlated groups (with >= 2 features) -> "correlated groups"
PROP_WEAK_RHO = 0.10          # |Spearman(feature, target)| below this = weak predictor
PROP_WEAK_SHARE = 0.50        # >= this share of weak predictors -> "many weak/redundant predictors"

# ------------------------------------------------------------------ inference
HAC_MAXLAGS = 48              # Newey-West lags (two days of hourly data)
N_BOOT = 400
BOOT_BLOCK = 168              # moving-block bootstrap, one-week blocks

# ------------------------------------------------------------------ decision threshold (error tolerance, ug/m3)
# PROPOSED operational tolerance, NOT a regulatory/authoritative value: roughly the mean absolute hour-to-hour change of PM2.5 in the
# training rows (~10.7 ug/m3), i.e. a forecast whose error is below this is no worse than a typical 'no change' guess.
# Must be confirmed or overridden by the author; the sensitivity table reports several alternatives.
DECISION_THRESHOLD = 10.0
THRESHOLD_GRID = [5.0, 10.0, 15.0, 20.0, 35.0]

# ------------------------------------------------------------------ slices (cut-offs come from TRAINING data or the calendar, never from the test set)
DAY_HOURS = (6, 18)           # target hour in [6, 18) = day
SEASONS = {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
           6: "summer", 7: "summer", 8: "summer", 9: "autumn", 10: "autumn", 11: "autumn"}
MAX_ERRORS_SHOWN = 3

# ------------------------------------------------------------------ external calendar knowledge (used ONLY to annotate large errors, never as a feature)
# First day of the Chinese New Year (Spring Festival) in the data period; fireworks around New Year's Eve / the first night are a documented short-lived
# PM2.5 source in Beijing (peaks around 01:00-02:00 of the first day, Atmos. Environ. 2020, "Changes in air pollution and attitude to fireworks in Beijing").
SPRING_FESTIVAL_DAY1 = ["2013-02-10", "2014-01-31", "2015-02-19", "2016-02-08", "2017-01-28"]
