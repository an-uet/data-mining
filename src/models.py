"""Baselines, the four linear model families, time-series CV tuning and the data-property-driven model-selection rule.

Random Forest / Gradient Boosting exist ONLY as baselines. The final predictor is always one of OLS / Ridge / Lasso / Elastic Net.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline

from . import config as C
from . import preprocessing as P


class PersistenceRegressor(RegressorMixin, BaseEstimator):
    """Baseline 2: PM2.5(t+1) = PM2.5(t)."""

    def __init__(self, column=C.CURRENT):
        self.column = column

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return X[self.column].to_numpy(float)


# ====================================================================== CV machinery
SCORING = {"mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error", "r2": "r2"}


def make_cv():
    return TimeSeriesSplit(n_splits=C.N_SPLITS)


def tune(estimator, grid, X, y, cv=None, n_jobs=C.N_JOBS):
    """GridSearchCV with the shared TimeSeriesSplit; refit on all training rows with the configuration of lowest mean CV-MAE."""
    gs = GridSearchCV(estimator, grid, cv=cv or make_cv(), scoring=SCORING, refit=C.PRIMARY_METRIC, n_jobs=n_jobs, error_score="raise")
    return gs.fit(X, y)


def _clean(v):
    return v.item() if hasattr(v, "item") else v


def cv_summary(gs, n_splits=C.N_SPLITS):
    i, r = gs.best_index_, gs.cv_results_
    fold = lambda m, sign: sign * np.array([r[f"split{k}_test_{m}"][i] for k in range(n_splits)])
    mae, rmse, r2 = fold("mae", -1), fold("rmse", -1), fold("r2", 1)
    return {"cv_mae": mae.mean(), "cv_mae_std": mae.std(ddof=1), "cv_rmse": rmse.mean(), "cv_rmse_std": rmse.std(ddof=1),
            "cv_r2": r2.mean(), "cv_r2_std": r2.std(ddof=1), "mae_folds": mae, "rmse_folds": rmse, "r2_folds": r2,
            "best_params": {k.split("__")[-1]: _clean(v) for k, v in gs.best_params_.items()}}


def pm(mean, std, d=2):
    return f"{mean:.{d}f} ± {std:.{d}f}"


def oof_predict(estimator, X, y, cv=None):
    """Out-of-fold predictions for the validation rows of every CV fold (NaN for rows that are never validated, i.e. the first block)."""
    from sklearn.base import clone
    cv = cv or make_cv()
    out = np.full(len(X), np.nan)
    for tr, va in cv.split(X):
        m = clone(estimator).fit(X.iloc[tr], y.iloc[tr])
        out[va] = m.predict(X.iloc[va])
    return out


# ====================================================================== linear families
FAMILIES = ["OLS", "Ridge", "Lasso", "ElasticNet"]
# precompute=True: coordinate descent on the Gram matrix (n >> p) - same solution, ~300x faster than the default solver
FAMILY_MODEL = {"OLS": lambda: LinearRegression(), "Ridge": lambda: Ridge(),
                "Lasso": lambda: Lasso(max_iter=50_000, precompute=True, random_state=C.SEED),
                "ElasticNet": lambda: ElasticNet(max_iter=50_000, precompute=True, random_state=C.SEED)}


def family_grid(name, alphas=None):
    if name == "OLS":
        return {}
    base = {"Ridge": C.RIDGE_ALPHAS, "Lasso": C.LASSO_ALPHAS, "ElasticNet": C.ENET_ALPHAS}[name]
    grid = {"model__alpha": base if alphas is None else alphas}
    if name == "ElasticNet":
        grid["model__l1_ratio"] = C.ENET_L1
    return grid


def extend_up(alphas, decades=3):
    step, top = np.log10(alphas[1] / alphas[0]), np.log10(alphas[-1])
    return np.concatenate([alphas, 10 ** np.arange(top + step, top + decades + 1e-9, step)])


def tune_family(name, make_estimator, X, y, cv=None, extra=None, tag="", edge_log=None):
    """Same grids and folds for every feature set. If the optimum lies on the UPPER edge of the alpha grid it is extended upward (once, by
    three decades) and the event is logged: an optimum on the edge would otherwise be an artefact of the grid."""
    extra = extra or {}
    grid = {**family_grid(name), **extra}
    gs = tune(make_estimator(FAMILY_MODEL[name]()), grid, X, y, cv)
    if name != "OLS" and np.isclose(gs.best_params_["model__alpha"], grid["model__alpha"].max()):
        ext = extend_up(grid["model__alpha"])
        if edge_log is not None:
            edge_log.append(f"{tag} | {name}: optimum on the upper alpha edge ({grid['model__alpha'].max():g}) -> grid extended to {ext.max():g}")
        gs = tune(make_estimator(FAMILY_MODEL[name]()), {**family_grid(name, ext), **extra}, X, y, cv)
    return gs


# ====================================================================== baselines
def _rf():
    return RandomForestRegressor(n_estimators=200, n_jobs=1, random_state=C.SEED)


def _hgb():
    return HistGradientBoostingRegressor(learning_rate=0.05, max_iter=300, early_stopping=False, l2_regularization=1.0, random_state=C.SEED)


RF_GRID = {"model__min_samples_leaf": [5, 20], "model__max_features": [0.33, 0.66]}
HGB_GRID = {"model__max_leaf_nodes": [15, 31], "model__min_samples_leaf": [20, 100]}


def baseline_estimators(specs):
    """key -> (label, features, estimator, grid). Rows of main Table 3 (training mean ... gradient boosting)."""
    A = specs["A"]
    return {
        "mean": ("Training mean", "-", DummyRegressor(strategy="mean"), {}),
        "persistence": ("Persistence: PM2.5(t+1) = PM2.5(t)", "PM2.5(t)", PersistenceRegressor(), {}),
        "ols_single": ("Single-feature OLS: PM2.5(t) -> PM2.5(t+1)", "PM2.5(t)",
                       Pipeline([("cols", P.ColumnSelector([C.CURRENT])), ("model", LinearRegression())]), {}),
        "rf_raw": ("Random Forest (raw features)", "Set A", P.build_pipeline(A, _rf()), RF_GRID),
        "hgb_raw": ("Gradient Boosting (raw features)", "Set A", P.build_pipeline(A, _hgb()), HGB_GRID),
    }


def reference_estimators(specs):
    """Supplementary nonlinear references on the engineered features (NOT candidates for the final model)."""
    Cc = specs["C"]
    return {"rf_C": ("Random Forest (Set C, reference only)", "Set C", P.build_pipeline(Cc, _rf()), RF_GRID),
            "hgb_C": ("Gradient Boosting (Set C, reference only)", "Set C", P.build_pipeline(Cc, _hgb()), HGB_GRID)}


# ====================================================================== model selection from data properties
def select_final_model(cv, props):
    """Pre-registered rule + one documented amendment.
    1) Families within 1 SE of the best mean CV-MAE are tied.
    2) The data properties of the TRAINING design decide among the tied families: correlated groups AND many weak/redundant predictors ->
       Elastic Net; strong multicollinearity -> Ridge; many weak predictors -> Lasso; otherwise OLS. The first family of the property-implied
       preference list that is tied is the 'original' (as pre-registered) choice.
    3) AMENDMENT (made after seeing training CV, before the freeze, test set locked): a penalised family whose CV-optimal penalty lies on the
       no-shrinkage edge of its grid is numerically OLS and does not regularise, so it is skipped and the next preference is used."""
    best = min(cv, key=lambda f: cv[f]["cv_mae"])
    se = cv[best]["cv_mae_std"] / np.sqrt(C.N_SPLITS)
    tied = [f for f in cv if cv[f]["cv_mae"] <= cv[best]["cv_mae"] + C.TIE_SE * se]
    if props["correlated_group_structure"] and props["many_weak_predictors"]:
        implied, reason = "ElasticNet", "correlated groups of predictors AND many weak/redundant predictors -> feature selection that keeps correlated groups together"
    elif props["high_collinearity"]:
        implied, reason = "Ridge", "strong multicollinearity -> shrink correlated coefficients jointly"
    elif props["many_weak_predictors"]:
        implied, reason = "Lasso", "many weak/redundant predictors -> sparse selection"
    else:
        implied, reason = "OLS", "no collinearity / sparsity problem -> unpenalised least squares"
    preference = [implied] + [f for f in ["ElasticNet", "Ridge", "Lasso", "OLS"] if f != implied]
    original = next(f for f in preference if f in tied)
    degenerate = [f for f in tied if f != "OLS" and cv[f].get("degenerate")]
    eligible = [f for f in tied if f not in degenerate]
    choice = next((f for f in preference if f in eligible), best)
    why = (f"Best mean CV-MAE: {best} ({cv[best]['cv_mae']:.3f}); 1 SE = {se:.3f}; families within 1 SE: {tied}. "
           f"Training-design properties: max VIF {props['max_vif']:.1f} ({props['max_vif_feature']}), {props['n_vif_gt_10']} features with VIF > {C.PROP_VIF:g}; "
           f"{props['n_correlated_groups']} correlated groups (|r| >= {C.PROP_CLUSTER_RHO}); {100 * props['share_weak']:.1f}% of features weak (|Spearman| < {C.PROP_WEAK_RHO}; threshold {100 * C.PROP_WEAK_SHARE:.0f}%). "
           f"Implied family: {implied} ({reason}); pre-registered outcome: {original}.")
    if choice != original:
        deg_txt = "; ".join("%s: %s, %d/%d non-zero" % (f, cv[f]["best_params"], cv[f]["nonzero"], cv[f]["n_coef"]) for f in degenerate)
        ch_txt = "%s, %d/%d non-zero" % (cv[choice]["best_params"], cv[choice]["nonzero"], cv[choice]["n_coef"])
        why += (f" AMENDMENT: {', '.join(degenerate)} {'has its' if len(degenerate) == 1 else 'have their'} CV-optimal penalty on the no-shrinkage edge of the grid "
                f"({deg_txt}), i.e. numerically OLS, so {'it is' if len(degenerate) == 1 else 'they are'} skipped -> chosen: {choice} ({ch_txt}).")
    else:
        why += f" Chosen: {choice}."
    return {"family": choice, "original": original, "implied": implied, "best_cv_family": best, "se": se, "tied": tied, "degenerate": degenerate, "why": why}
