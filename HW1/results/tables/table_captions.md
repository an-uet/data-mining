**table_1_dataset_statistics** - Dataset statistics.

**table_1b_target_statistics** - Statistics of the target PM2.5 at t+1 (µg/m³) in the usable training and test rows.

**table_2_feature_groups_and_rationale** - Feature groups, members, rationale and the hypothesis each group tests.

**table_3_model_comparison** - Baselines and linear models: 5-fold time-series CV on the training rows (mean ± std over folds) and a single evaluation on the held-out test period (µg/m³). ★ = selected final model. Bold = best value in the column over all rows.

**table_4_feature_ablation** - Feature ablation with one linear family (Lasso), identical folds and search procedure; only the feature set changes. p = number of design columns entering the model. The last row is a sensitivity run, not a candidate. Bold = best value in the column (excluding the sensitivity row).

**table_5_final_model_coefficients** - Final model (Lasso, Set C): coefficients in µg/m³ per one standard deviation of the (centred) feature with 95% intervals (moving-block bootstrap for the penalised predictive model; HAC for the OLS statistical companion fitted on the same standardised design), HAC and Holm-adjusted p-values and VIF. Training rows only. Predictive and inferential models are distinct.

**table_6_three_largest_test_errors** - The three largest absolute errors of the final model on the test period (error = actual − predicted, µg/m³; time = target hour). Explanations are rule-based hypotheses, not established causes.

**table_7_slice_level_performance** - Test-set MAE and RMSE (µg/m³) of the final model, the same linear family on the raw features (Set A) and persistence, by slice. Slice cut-offs come from the training data or the calendar. Δ < 0 means the final model is better.

**table_S01_schema_and_missing_values** - Schema of the raw file: dtype and missing values of every column (whole file, structural check).

**table_S02_missing_value_policy** - Usable training rows as a function of the forward-fill limit applied to the PM2.5 history (the chosen limit is 3 hours).

**table_S03_feature_target_association** - Pearson and Spearman correlation of every candidate feature with the next-hour PM2.5 (training rows only).

**table_S04_derived_feature_candidates** - Group 4: candidate derived quantities and their training-only Spearman correlation with the target and with the next-hour change; kept if the larger |rho| >= 0.10.

**table_S05_cv_folds** - Expanding-window TimeSeriesSplit folds on the usable training rows (every validation block lies strictly after its training block).

**table_S06_leakage_checks** - Results of check_no_data_leakage(): perturbation, identity, prefix-invariance and fold-purity checks.

**table_S07_linear_models_cv_selection** - Linear families on Set C: tuned hyper-parameters, time-series CV metrics and the outcome of the pre-registered selection rule.

**table_S08_training_design_properties** - Data properties of the Set C training design matrix that drive the pre-registered choice among tied linear families.

**table_S09_group_wald_tests** - Joint Wald tests (HAC covariance) that all coefficients of a group are zero in the OLS companion on the standardised Set C design.

**table_S10_nonlinear_reference_on_set_c** - Supplementary: Random Forest and Gradient Boosting on the engineered Set C features (nonlinear references; never candidates for the final model).

**table_S11_diebold_mariano** - Diebold–Mariano tests (HAC, 24 lags) on the test period; Δ = loss(final Lasso) − loss(comparator); negative means the final model is better.

**table_S12_decision_threshold_sensitivity** - Share of test predictions with |error| <= tolerance (µg/m³). The proposed tolerance is 10 µg/m³ (operational, not authoritative).

**table_S13_residual_diagnostics** - Residual diagnostics of the final model on the test period (error = actual − predicted).

**table_S15_variable_summary_train** - Summary of the raw numeric predictors on the usable training rows.

**table_S16_coefficients_by_group** - Final model: coefficients per feature group (µg/m³ per SD): number of non-zero coefficients, sum of absolute coefficients, bootstrap intervals excluding zero and the largest effects.

**table_S17_hypothesis_verdicts** - Hypotheses H1-H4: pre-registered verdicts from training CV and training inference; the last column repeats the same comparison on the test period for information only (never used for a verdict).
