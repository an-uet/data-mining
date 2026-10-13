"""Statistical (inferential) tools: VIF, collinearity / correlated-group diagnostics, OLS with HAC errors, group Wald tests,
moving-block bootstrap, Diebold-Mariano tests and the pre-registered hypothesis verdict logic.

Predictive model != inferential model: the penalised model produces the forecasts; OLS with HAC (Newey-West) errors on the *same*
standardised design is its statistical companion (ridge/lasso/elastic-net have no classical p-values)."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from joblib import Parallel, delayed
from scipy import stats
from scipy.sparse.csgraph import connected_components
from sklearn.base import clone
from statsmodels.stats.multitest import multipletests

from . import config as C


# ====================================================================== collinearity
def vif_series(D):
    """VIF_j = 1 / (1 - R2_j) from regressing column j on the others (inf for exact linear dependence)."""
    Z = np.asarray(D, float)
    Z = Z - Z.mean(0)
    out = []
    for j in range(Z.shape[1]):
        others = np.delete(Z, j, axis=1)
        beta, *_ = np.linalg.lstsq(others, Z[:, j], rcond=None)
        res = Z[:, j] - others @ beta
        tss = Z[:, j] @ Z[:, j]
        r2 = 1 - (res @ res) / tss if tss > 0 else 1.0
        out.append(np.inf if 1 - r2 < 1e-10 else 1.0 / (1.0 - r2))
    return pd.Series(out, index=list(D.columns))


def rank_of(D):
    Z = np.asarray(D, float)
    return int(np.linalg.matrix_rank(Z - Z.mean(0)))


def condition_number(D):
    Z = np.asarray(D, float)
    Z = (Z - Z.mean(0)) / np.where(Z.std(0) > 0, Z.std(0), 1.0)
    s = np.linalg.svd(Z, compute_uv=False)
    return float(s[0] / s[-1])


def correlated_groups(D, rho=C.PROP_CLUSTER_RHO):
    """Connected components of the graph |Pearson r| >= rho among the design columns (components with >= 2 features)."""
    R = np.corrcoef(np.asarray(D, float), rowvar=False)
    R = np.nan_to_num(R)
    adj = (np.abs(R) >= rho) & ~np.eye(R.shape[0], dtype=bool)
    n, lab = connected_components(adj, directed=False)
    groups = [[D.columns[i] for i in np.flatnonzero(lab == g)] for g in range(n)]
    return [g for g in groups if len(g) >= 2]


def spearman_with(D, y):
    return pd.Series({c: stats.spearmanr(D[c], y)[0] if D[c].std() > 0 else 0.0 for c in D.columns})


def data_properties(D, y):
    """Properties of the TRAINING design matrix that drive the choice among tied linear families (pre-registered thresholds)."""
    vif = vif_series(D)
    groups = correlated_groups(D)
    rho = spearman_with(D, np.asarray(y, float)).abs()
    props = {"n_features": int(D.shape[1]), "rank": rank_of(D), "condition_number": condition_number(D),
             "max_vif": float(vif.max()), "max_vif_feature": str(vif.idxmax()), "n_vif_gt_10": int((vif > C.PROP_VIF).sum()),
             "n_correlated_groups": len(groups), "correlated_groups": [g for g in groups],
             "n_weak": int((rho < C.PROP_WEAK_RHO).sum()), "share_weak": float((rho < C.PROP_WEAK_RHO).mean())}
    props["high_collinearity"] = bool(props["max_vif"] > C.PROP_VIF)
    props["correlated_group_structure"] = bool(props["n_correlated_groups"] >= C.PROP_MIN_CLUSTERS)
    props["many_weak_predictors"] = bool(props["share_weak"] >= C.PROP_WEAK_SHARE)
    return props, vif


# ====================================================================== OLS with HAC (inferential companion)
def ols_companion(y, Z, maxlags=C.HAC_MAXLAGS):
    """OLS on the standardised design: classical fit + HAC (Newey-West) covariance. Z is a DataFrame of standardised features."""
    Xc = sm.add_constant(Z, has_constant="add")
    cls = sm.OLS(np.asarray(y, float), Xc).fit()
    hac = sm.OLS(np.asarray(y, float), Xc).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return cls, hac


def wald_group(res, cols):
    """Joint Wald test (chi-square, robust covariance of `res`) that all coefficients in `cols` are zero."""
    names = list(res.model.exog_names)
    idx = [names.index(c) for c in cols if c in names]
    R = np.zeros((len(idx), len(names)))
    R[np.arange(len(idx)), idx] = 1.0
    w = res.wald_test(R, use_f=False, scalar=True)
    return float(w.statistic), float(w.pvalue), len(idx)


def holm(p):
    return multipletests(np.asarray(p, float), method="holm")[1]


# ====================================================================== moving-block bootstrap of the penalised model
def _fit_coef(estimator, X, y):
    return clone(estimator).fit(X, y).named_steps["model"].coef_.copy()


def block_bootstrap_coefs(estimator, X, y, n_boot=C.N_BOOT, block=C.BOOT_BLOCK, seed=C.SEED, n_jobs=C.N_JOBS):
    """Moving-block bootstrap of the rows (blocks keep the serial dependence); the whole pipeline is refitted on every resample."""
    rng = np.random.default_rng(seed)
    n = len(X)
    n_blocks = int(np.ceil(n / block))
    idxs = []
    for _ in range(n_boot):
        starts = rng.integers(0, n - block + 1, n_blocks)
        idxs.append(np.concatenate([np.arange(s, s + block) for s in starts])[:n])
    out = Parallel(n_jobs=n_jobs)(delayed(_fit_coef)(estimator, X.iloc[i], y.iloc[i]) for i in idxs)
    return np.asarray(out)


# ====================================================================== predictive-accuracy test
def diebold_mariano(y_true, p_a, p_b, loss="se", maxlags=24):
    """Diebold-Mariano test (HAC variance). d = loss(a) - loss(b); mean(d) < 0 means model a is more accurate. Returns (mean d, t, p)."""
    e1, e2 = np.asarray(y_true) - np.asarray(p_a), np.asarray(y_true) - np.asarray(p_b)
    d = (e1 ** 2 - e2 ** 2) if loss == "se" else (np.abs(e1) - np.abs(e2))
    r = sm.OLS(d, np.ones(len(d))).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return float(r.params[0]), float(r.tvalues[0]), float(r.pvalues[0])


# ====================================================================== pre-registered hypothesis verdicts
def verdict(rel_increase, folds_worse, wald_p, margin=C.HYP_MIN_REL_INCREASE, min_folds=C.HYP_MIN_FOLDS, alpha=C.HYP_ALPHA):
    """Supported = predictive AND inferential criterion; Partly supported = exactly one; Not supported = neither."""
    predictive = bool(rel_increase >= margin and folds_worse >= min_folds)
    inferential = bool(wald_p < alpha)
    label = "Supported" if (predictive and inferential) else ("Partly supported" if (predictive or inferential) else "Not supported")
    return label, predictive, inferential
