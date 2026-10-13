#!/usr/bin/env python
"""Build the eight HW1 notebooks (readable views of the experiment).

    python tools/build_notebooks.py                      # (re)write notebooks/*.ipynb (not executed)
    jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb      # run them (after `python run_all.py`)

Design: `python run_all.py` runs the whole experiment once (src/experiment.py) and writes results/. The notebooks never re-open the held-out
test set and never overwrite a result: notebooks 01, 02 and 04 recompute the cheap, training-only steps live with the src/ functions
(inspection, split, feature construction, leakage checks); notebooks 03 and 05-08 present the stored results of the corresponding stages
with their computed captions. The narrative text contains no result numbers - every number on the page is read from results/.
"""
import textwrap
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"

SETUP = r'''
import io, json, re, sys
from pathlib import Path
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src" / "config.py").exists())
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from PIL import Image as PILImage
from IPython.display import Image, Markdown, display
from src import config as C

pd.set_option("display.width", 200); pd.set_option("display.max_columns", 40); pd.set_option("display.max_colwidth", 90)

def table(stem, **kw):
    """results/tables/<stem>.csv (written by run_all.py)."""
    return pd.read_csv(C.TABLES / f"{stem}.csv", **kw)

def caption(stem):
    txt = (C.FIGURES / "figure_captions.md").read_text()
    m = re.search(rf"\*\*{re.escape(stem)}\*\* - (.*?)(?:\n\n|\Z)", txt, re.S)
    return m.group(1).strip() if m else ""

def fig(stem, width=1500):
    """Show results/figures/<stem>.png (down-scaled to keep the notebook small) with the caption computed from the data."""
    im = PILImage.open(C.FIGURES / f"{stem}.png")
    if im.width > width:
        im = im.resize((width, round(im.height * width / im.width)), PILImage.LANCZOS)
    buf = io.BytesIO(); im.save(buf, format="PNG", optimize=True)
    display(Image(data=buf.getvalue())); display(Markdown(f"*{caption(stem)}*"))

def log_grep(pattern, maxlen=420):
    """Print the lines of results/logs/run_all.log that match a regular expression."""
    for line in (C.LOGS / "run_all.log").read_text().splitlines():
        if re.search(pattern, line):
            print(line[:maxlen])
'''

LIVE_TRAIN = r'''
from src import data as D, features as F, models as M
raw = D.load_raw()
df, _ = D.build_datetime(raw)
df, _ = D.drop_identifier_columns(df)
df = D.make_target(df)                                     # target(t) = PM2.5(t+1); the last hour is removed; asserted inside
train, vault, info = D.chronological_split(df)             # earliest 80 % = train, latest 20 % = locked vault
del df                                                     # only the training rows stay in memory
'''


def live(extra):
    """LIVE_TRAIN (unindented) followed by further, indented code: dedent only the added part."""
    return LIVE_TRAIN + textwrap.dedent(extra)


def md(text):
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip())


def code(text):
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip())


NOTEBOOKS = {}

# ======================================================================================================== 01
NOTEBOOKS["01_data_inspection"] = [
    md("""
    # 01 · Data inspection
    **Question.** What exactly is in the raw file, and is it a clean, regular hourly time series?
    **Scope.** The raw file only: no split, no modelling and no statistic that feeds a design decision.
    **Code.** `src/data.py` (loading, datetime, timeline validation, inspection). Live cells below; the stored tables come from `python run_all.py`.

    The file is `PRSA_Data_Aotizhongxin_20130301-20170228.csv` from the notebook the project started from. The provenance note printed below explains why it is
    attributed to the UCI *Beijing Multi-Site Air-Quality Data* set (id 501) and not to the `fetch_ucirepo(id=381)` snippet of the assignment brief.
    """),
    code(SETUP),
    md("## 1. Source, size and schema"),
    code("""
    from src import data as D
    raw = D.load_raw()                                   # asserts the expected number of rows
    prov = D.provenance()
    print(f"{prov['name']}  (DOI {prov['doi']})")
    print(f"station: {prov['station']} | file: {prov['file']} | sha256 {prov['sha256'][:16]}...")
    print("note   :", prov["note"])
    display(raw.head())
    rep = D.inspect_raw(raw)
    print(f"\\nrows: {rep['rows']:,} | columns: {rep['columns']}")
    display(rep["dtypes"].round({"missing_pct": 2}))
    print("duplicated rows:", rep["duplicate_rows"], "| duplicated rows ignoring the 'No' counter:", rep["duplicate_rows_ignoring_No"])
    print("wind-direction levels (including NaN):", len(rep["wd_levels"]))
    """),
    md("""
    ## 2. Timeline checks (asserted, not assumed)
    `year/month/day/hour` → `datetime`, sorted chronologically; `validate_timeline` asserts that the series is monotonic, has no duplicated timestamps, a single sampling
    interval of one hour and no missing timestamp. `No` (row counter) and `station` (constant) are identifiers and are dropped.
    """),
    code("""
    df, was_sorted = D.build_datetime(raw)
    tl = D.validate_timeline(df)
    print("already sorted in the file:", was_sorted)
    display(pd.Series(tl, name="value").to_frame())
    df, ids = D.drop_identifier_columns(df)
    print("identifier columns removed:", ids)
    """),
    md("""
    ## 3. Target statistics (descriptive, whole file)
    A structural summary only. Every distribution used for a modelling decision is computed on the **training rows** in notebook 03.
    """),
    code("""
    display(pd.Series(rep["target_stats"], name="PM2.5 (µg/m³), whole file").to_frame().T.round(2))
    """),
    md("## 4. Stored tables (`results/tables`)"),
    code("""
    display(table("table_1_dataset_statistics"))
    display(table("table_1b_target_statistics").round(1))
    display(table("table_S01_schema_and_missing_values").round(2))
    """),
]

# ======================================================================================================== 02
NOTEBOOKS["02_train_test_split"] = [
    md("""
    # 02 · Target construction and chronological split
    **Task.** Predict PM2.5 one hour ahead from information available at hour *t*: **X(t) → PM2.5(t+1)** (µg/m³).
    **Protocol.** The target is built from the raw series (never filled) and the last hour is removed. The earliest 80 % of the hours are the training period, the latest 20 % the
    held-out test period. No shuffling. The test rows are moved into a `HeldOutTest` vault: only their date range is public, and every `unlock()` is logged.
    **Code.** `src/data.py` (`make_target`, `chronological_split`, `HeldOutTest`).
    """),
    code(SETUP),
    md("## 1. Target construction and split (live)"),
    code(live("""
    print(f"train: {len(train):,} hours | test (locked): {vault.n_rows:,} hours")
    display(train[["datetime", "PM2.5", "target"]].head(5))
    """)),
    code("""
    display(pd.Series(info, name="value").to_frame())
    print(vault)                                                     # rows / date range only; nothing has been unlocked here
    print("gap between the last training hour and the first test hour:", vault.start - train["datetime"].iloc[-1])
    saved = json.loads((C.LOGS / "split_info.json").read_text())
    assert saved == info, "the split differs from the one used by run_all.py"
    print("identical to results/logs/split_info.json")
    """),
    md("""
    `make_target` asserts `target[:-1] == PM2.5[1:]` (the target is the next hour's reading) and that the target column is not the current PM2.5 column.
    The split asserts that every training timestamp precedes every test timestamp and that the two periods are contiguous.
    """),
    md("## 2. Figures"),
    code("""
    fig("figure_01_prediction_pipeline")
    fig("figure_S08_split_and_cv_folds")
    """),
    md("""
    ## 3. Access log of the held-out test period
    The vault is opened only after the design (features, hyper-parameters, folds, seed) has been frozen and fingerprinted (SHA-256). One entry per execution of the pipeline that reached the test stage.
    """),
    code("""
    display(pd.DataFrame(json.loads((C.LOGS / "test_access_log.json").read_text())))
    """),
]

# ======================================================================================================== 03
NOTEBOOKS["03_train_only_eda"] = [
    md("""
    # 03 · Training-only exploratory data analysis
    Every statistic and plot on this page uses the **training period only** (the test rows are locked). Pearson and Spearman correlations of every candidate
    predictor with the target and the feature–feature redundancy are computed here; nothing from the test period enters.
    **Code.** `Experiment.stage_training_data` and `Experiment.stage_eda` (`src/experiment.py`), plots in `src/plotting.py`. The figures below are the stored outputs of `python run_all.py`.

    | # | Required item | Where |
    |---|---|---|
    | 1 | PM2.5 distribution | Figure 2 |
    | 2 | PM2.5 over time | Figure S1 |
    | 3, 4 | PM2.5 by hour / by month | Figure 3 (a, b; c: weekday) |
    | 5–10 | PM2.5 vs PM10, NO2, CO, O3, temperature, dew point (humidity proxy) | Figure S2 |
    | 11–13 | correlation heatmap, Spearman, Pearson | Figure 4 (both), Figure 5, table S03 |
    | 14 | feature–feature redundancy | Figure S5 (VIF, clustered correlations), log line *raw-variable redundancy* |
    | + | wind effects, autocorrelation, missing-value policy | Figures S3, S6, S4 |
    """),
    code(SETUP),
    md("## 1. Target distribution and temporal structure"),
    code("""
    fig("figure_02_pm25_distribution")
    fig("figure_S01_pm25_over_time")
    fig("figure_03_pm25_temporal_pattern")
    """),
    md("## 2. Relation to pollutants, meteorology and wind"),
    code("""
    fig("figure_S02_target_vs_pollutants_meteorology")
    fig("figure_S03_wind_effects")
    """),
    md("## 3. Correlations with the target (Pearson and Spearman, training rows only)"),
    code("""
    fig("figure_04_correlation_heatmap_train")
    fig("figure_05_feature_target_correlation_ranking")
    display(table("table_S03_feature_target_association").head(20).round(3))
    """),
    md("## 4. Memory of the series, missing values and redundancy"),
    code("""
    fig("figure_S06_autocorrelation")
    fig("figure_S04_missing_values_policy")
    display(table("table_S02_missing_value_policy").round(1))
    fig("figure_S05_feature_redundancy")
    """),
    md("## 5. Numbers logged by the EDA stage"),
    code("""
    log_grep(r"^training    :|^POLICY|^usable rows|^target \\(next-hour|^data quality|^autocorrelation of|^top-12|^strongest raw|^raw-variable|^daily cycle|^median target by wind|^ventilation")
    """),
    md("""
    **Data-quality observation.** `PM10` equals `PM2.5(t)` exactly in a large share of the hours and is even smaller in a few, which is physically inconsistent (PM10 contains PM2.5).
    The share is printed above (*data quality*). PM10-based features and the PM2.5/PM10 ratio therefore have to be read with care.
    """),
]

# ======================================================================================================== 04
NOTEBOOKS["04_feature_engineering"] = [
    md("""
    # 04 · Feature engineering, feature sets and leakage checks
    **Feature sets.** **A** raw predictors (wind direction one-hot, reference level N); **B** the *k* most correlated features, with the ranking recomputed **inside every training fold**
    (k ∈ {5, 10, 15, 20}; Pearson or Spearman); **C** engineered features in five motivated groups (temporal, PM2.5 history, shifted rolling statistics, environmental differences, interactions).
    **Missing values.** The target is never filled; the current PM2.5 must be observed; PM2.5 *history* features use a causal last-observation-carried-forward over gaps of at most 3 hours;
    every other variable is imputed inside each training fold (`SimpleImputer`).
    **Code.** `src/features.py` (`build_candidate_features`, `build_specs`, `check_no_data_leakage`), `src/preprocessing.py` (leakage-safe transformers and `build_pipeline`).
    """),
    code(SETUP),
    md("## 1. Feature groups, rationale and the hypothesis each group tests (Table 2)"),
    code("""
    display(table("table_2_feature_groups_and_rationale"))
    display(table("table_S04_derived_feature_candidates").round(3))
    """),
    md("""
    ## 2. Build the training matrices and run `check_no_data_leakage()` (live)
    The checks perturb the future and verify that no feature changes (rolling windows and lags use only the past), that the target never enters the inputs, that the test vault is untouched,
    and that imputer medians, scaler statistics and the top-k ranking depend on the **training fold only**.
    """),
    code(live("""
    feats = F.build_candidate_features(train)                  # features at hour t from information up to t only
    elig = F.eligibility(train, feats)                         # the same usability rule for every model
    X = feats.loc[elig].reset_index(drop=True)
    y = train.loc[elig, C.TARGET].reset_index(drop=True)
    dtab = F.derived_candidate_table(X, y)
    specs = F.build_specs(dtab.loc[dtab["kept"], "candidate"].tolist())
    print(f"{len(X):,} usable training rows of {len(train):,} | candidate columns: {X.shape[1]}")
    chk = F.check_no_data_leakage(train_frame=train, X_train=X, y_train=y, spec=specs["C"], cv=M.make_cv(), vault=vault, verbose=False)
    display(chk)
    assert chk["passed"].all(), "a leakage check failed"
    print(f"all {len(chk)} checks passed")
    """)),
    md("## 3. The three feature sets as used by the experiment"),
    code("""
    fl = json.loads((C.MODELS / "feature_lists.json").read_text())
    for k, v in fl.items():
        print(f"{k:22s} ({len(v):2d})", v)
    """),
    md("## 4. Missing-value policy, redundancy of the engineered design and the shared CV folds"),
    code("""
    fig("figure_S04_missing_values_policy")
    fig("figure_S05_feature_redundancy")
    display(table("table_S05_cv_folds"))
    display(table("table_S06_leakage_checks"))
    """),
]

# ======================================================================================================== 05
NOTEBOOKS["05_baselines"] = [
    md("""
    # 05 · Baselines (time-series cross-validation on the training rows)
    Baselines of increasing strength: training mean → persistence (PM2.5(t+1) = PM2.5(t)) → single-feature OLS (PM2.5(t) → PM2.5(t+1)) → Random Forest and Gradient Boosting on the raw
    features (Set A). **Random Forest and Gradient Boosting are baselines only** and can never be the final model.
    All methods use the same five expanding-window folds (`TimeSeriesSplit(5)`, figure S8) and the same eligibility rule; imputer and scaler are fitted inside each training fold.
    **Code.** `src/models.py` (`PersistenceRegressor`, `baseline_estimators`, `tune`), `Experiment.stage_baselines`.
    The held-out test results of these baselines are in notebook 08 (Table 3).
    """),
    code(SETUP),
    md("## 1. Cross-validated performance (training period, mean ± std over the five folds)"),
    code("""
    t3 = table("table_3_model_comparison")
    base = t3.iloc[:5][["Method", "Features", "CV_MAE", "CV_MAE_std", "CV_RMSE", "CV_RMSE_std", "CV_R2", "CV_R2_std"]]
    display(base.round(3))
    """),
    md("## 2. Hyper-parameters chosen for the tree ensembles (inner search on the same folds)"),
    code("""
    hp = json.loads((C.MODELS / "selected_hyperparameters.json").read_text())["baselines"]
    display(pd.DataFrame(hp).T)
    """),
    md("""
    ## 3. Supplementary: the same ensembles on the engineered Set C
    Not candidates for the final model; they show whether the engineered features help a flexible model as well (test columns are descriptive).
    """),
    code("""
    display(table("table_S10_nonlinear_reference_on_set_c"))
    """),
]

# ======================================================================================================== 06
NOTEBOOKS["06_linear_models"] = [
    md("""
    # 06 · Linear models: OLS, Ridge, Lasso, Elastic Net
    Pipeline for every family: column selection → fold-wise imputation → centred interactions → wind one-hot → `StandardScaler` → linear model, all fitted inside each training fold.
    Grids as specified (extended upward only when an optimum sits on the upper edge): Ridge α ∈ logspace(−3, 3, 13); Lasso α ∈ logspace(−4, 1, 10); Elastic Net α ∈ logspace(−4, 2, 10) × l1_ratio ∈ {0.1, 0.3, 0.5, 0.7, 0.9}.
    Primary metric: MAE (mean over the same five folds).

    **Choosing the final family from the data, not from the ranking.** Families within one standard error of the best CV-MAE are tied; among the tied families the properties of the *training design matrix*
    decide: correlated groups **and** many weak predictors → Elastic Net; strong multicollinearity → Ridge; many weak predictors → Lasso; otherwise OLS.
    A documented amendment (written after the training CV, before the freeze) skips a penalised family whose CV-optimal penalty sits at the no-shrinkage edge of its grid, because it is numerically OLS.
    **Code.** `src/models.py` (`tune_family`, `select_final_model`), `src/statistics.py` (`data_properties`), `Experiment.stage_linear_models`.
    """),
    code(SETUP),
    md("## 1. Cross-validated results and the selection outcome"),
    code("""
    display(table("table_S07_linear_models_cv_selection"))
    """),
    md("## 2. Data properties of the training design that drive the choice"),
    code("""
    display(table("table_S08_training_design_properties").round(3))
    log_grep(r"^DATA PROPERTIES|^   correlated group|^SELECTION|pre-registered rule alone")
    """),
    md("## 3. Pre-registered rule and recorded amendment"),
    code("""
    pre = json.loads((C.LOGS / "preregistration.json").read_text())
    print(json.dumps(pre["model_selection_rule"], indent=2))
    for a in pre.get("amendments", []):
        print("\\nAMENDMENT:", json.dumps(a, indent=2))
    """),
    md("## 4. Tuning curve of the selected family"),
    code("""
    fig("figure_06_tuning_curve")
    final = json.loads((C.MODELS / "selected_hyperparameters.json").read_text())["final_model"]
    print("final model:", final["family"], "| feature set:", final["feature_set"], "| hyper-parameters:", final["hyperparameters"])
    """),
]

# ======================================================================================================== 07
NOTEBOOKS["07_feature_ablation"] = [
    md("""
    # 07 · Feature ablation and the hypotheses
    Only the feature set changes: same split, same folds, same linear family, same search procedure, same metrics.
    Set A (raw), Set B (top-k correlated, ranking recomputed inside every fold; B′ draws from the full candidate pool instead of the raw columns), Set C (engineered), and
    group ablations: C − temporal, C − lag, C − rolling, C − interaction (plus further group removals and one sensitivity run).

    **Hypotheses (pre-registered before any model was compared).** H1: recent PM2.5 predicts next-hour PM2.5. H2: environmental measurements add information beyond PM2.5 history.
    H3: hourly/weekly patterns are captured by temporal features. H4: rolling statistics are more stable than individual lags.
    A hypothesis counts as *supported* only if removing the group raises the mean CV-MAE by ≥ 1 % and is worse in ≥ 4 of 5 folds **and** the joint Wald test (HAC) rejects at 5 %;
    *partly supported* if exactly one criterion holds. The verdicts use training CV and training inference only; the test columns are descriptive.
    **Code.** `Experiment.stage_ablation`, `Experiment.stage_statistics`, `src/statistics.py` (`wald_group`, `verdict`).
    """),
    code(SETUP),
    md("## 1. Ablation table (Table 4)"),
    code("""
    t4 = table("table_4_feature_ablation")
    display(t4[["Variant", "role", "p", "nonzero_coefficients", "CV_MAE", "CV_MAE_std", "CV_RMSE", "Test_MAE", "Test_RMSE", "Test_R2", "delta_CV_MAE_vs_C_pct", "delta_test_MAE_vs_C_pct"]].round(3))
    fig("figure_08_feature_ablation")
    """),
    md("## 2. Hypotheses: pre-registered verdicts and joint Wald tests"),
    code("""
    display(table("table_S17_hypothesis_verdicts").round(3))
    display(table("table_S09_group_wald_tests"))
    log_grep(r"^  H[1-4]|H4 stability|H3 decomposition")
    """),
    md("## 3. Features chosen by the leakage-safe top-k selection (fitted on the whole training period for the refit model)"),
    code("""
    log_grep(r"Set B.*: selected")
    """),
]

# ======================================================================================================== 08
NOTEBOOKS["08_final_analysis"] = [
    md("""
    # 08 · Final evaluation and analysis
    The held-out test period was opened **once** by `python run_all.py`, after the design was frozen (`results/models/frozen_design.sha256`); see the access log at the bottom.
    Nothing below changes a feature, model, hyper-parameter, preprocessing step or hypothesis. Residual-based statements describe the errors; they are not read as evidence that the model is good.
    **Code.** `Experiment.stage_final_test`, `stage_analysis`, `stage_report`; `src/evaluation.py`.
    """),
    code(SETUP),
    md("## 1. Summary printed at the end of `run_all.py`"),
    code("""
    print((C.RESULTS / "final_summary.txt").read_text())
    """),
    md("## 2. Main results (Table 3; best value per column in bold)"),
    code("""
    t3 = table("table_3_model_comparison")
    show = t3[["Method", "CV_MAE", "CV_MAE_std", "CV_RMSE", "CV_RMSE_std", "Test_MAE", "Test_RMSE", "Test_R2"]].copy()
    fmt = {c: "{:.2f}" for c in show.columns[1:7]}; fmt["Test_R2"] = "{:.4f}"
    display(show.style.format(fmt).hide(axis="index")
            .highlight_min(subset=["CV_MAE", "CV_RMSE", "Test_MAE", "Test_RMSE"], props="font-weight: bold; background-color: #cde2fb")
            .highlight_max(subset=["Test_R2"], props="font-weight: bold; background-color: #cde2fb"))
    fig("figure_07_model_comparison")
    display(table("table_S11_diebold_mariano"))
    """),
    md("""
    ## 3. Decision threshold
    `DECISION_THRESHOLD` (`src/config.py`) is a **proposed** operational tolerance in µg/m³, not an authoritative standard; the sensitivity table shows how the comparison depends on it.
    """),
    code("""
    print("DECISION_THRESHOLD =", C.DECISION_THRESHOLD, "µg/m³ (proposed; to be confirmed or overridden by the author)")
    display(table("table_S12_decision_threshold_sensitivity").round(3))
    fig("figure_S07_decision_threshold_sensitivity")
    """),
    md("## 4. Slice-level performance (final test predictions; cut-offs from the training data or the calendar)"),
    code("""
    display(table("table_7_slice_level_performance").round(2))
    fig("figure_12_slice_performance")
    """),
    md("## 5. Error analysis: the three largest absolute errors"),
    code("""
    pd.set_option("display.max_colwidth", None)
    display(table("table_6_three_largest_test_errors")[["case", "time", "actual", "predicted", "error", "abs_error", "hour", "month", "PM10", "NO2", "CO", "TEMP", "WSPM", "wd", "possible_explanation"]].round(1))
    pd.set_option("display.max_colwidth", 90)
    fig("figure_09_actual_vs_predicted")
    log_grep(r"SHARED PATTERN|one-hour change PM2.5")
    """),
    md("## 6. Residual analysis"),
    code("""
    fig("figure_10_residual_analysis")
    display(table("table_S13_residual_diagnostics"))
    display(table("table_S14_residual_spread_by_prediction_decile").round(2))
    """),
    md("""
    ## 7. Coefficient analysis (predictive model vs OLS statistical companion)
    The final model is penalised, so ordinary OLS *p*-values do not exist for it. Intervals for the final model come from a moving-block bootstrap (one-week blocks); the **OLS companion** is fitted on the same
    standardised Set C design with HAC (Newey–West) standard errors and supplies the *p*-values (Holm-adjusted) and VIFs. Training rows only.
    """),
    code("""
    t5 = table("table_5_final_model_coefficients")
    display(t5[["feature", "group", "coef", "boot_lo", "boot_hi", "ols", "hac_lo", "hac_hi", "p_hac", "p_holm", "vif"]].round(3))
    fig("figure_11_coefficients_confidence_intervals")
    display(table("table_S16_coefficients_by_group").round(2))
    log_grep(r"^COEFFICIENT SIGNS|^  H[1-4] (recent|environment|calendar|rolling)")
    """),
    md("## 8. Hypotheses revisited (training verdicts; test comparison for information only)"),
    code("""
    display(table("table_S17_hypothesis_verdicts").round(3))
    """),
    md("## 9. Access log of the held-out test period and the frozen design"),
    code("""
    display(pd.DataFrame(json.loads((C.LOGS / "test_access_log.json").read_text())))
    print("frozen design sha256:", (C.MODELS / "frozen_design.sha256").read_text().strip())
    """),
]


def build():
    NB_DIR.mkdir(exist_ok=True)
    for name, cells in NOTEBOOKS.items():
        nb = nbf.v4.new_notebook()
        nb.cells = cells
        nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
        nb.metadata["language_info"] = {"name": "python"}
        nbf.validate(nb)
        nbf.write(nb, NB_DIR / f"{name}.ipynb")
        print("wrote", NB_DIR / f"{name}.ipynb", f"({len(cells)} cells)")


if __name__ == "__main__":
    build()
