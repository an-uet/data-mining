"""Leakage-safe preprocessing components.

Every component that LEARNS something (imputation values, centring means, the top-k ranking, scaling constants) learns it in fit() from the
rows it is given - inside `GridSearchCV` those are the training rows of the current fold only.

Pipeline order:  ColumnSelector -> DFImputer (sklearn SimpleImputer) -> [CenteredProducts] -> [WindOneHot] -> [CorrTopK] -> StandardScaler -> model
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config as C
from . import features as F


class ColumnSelector(TransformerMixin, BaseEstimator):
    """Pick columns by name (keeps pandas) so that every feature set can share one candidate matrix."""

    def __init__(self, columns=()):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[list(self.columns)]


class DFImputer(TransformerMixin, BaseEstimator):
    """sklearn SimpleImputer (median for numeric columns, most frequent for categorical ones) that returns a DataFrame.

    The statistics are those of the rows passed to fit(): training-fold rows inside the CV, the full training part for the final refit.
    """

    def __init__(self, numeric_strategy="median", categorical_strategy="most_frequent"):
        self.numeric_strategy = numeric_strategy
        self.categorical_strategy = categorical_strategy

    @staticmethod
    def _imputer(strategy):
        try:
            return SimpleImputer(strategy=strategy, keep_empty_features=True)
        except TypeError:                                                         # older scikit-learn
            return SimpleImputer(strategy=strategy)

    def fit(self, X, y=None):
        self.columns_ = list(X.columns)
        self.num_cols_ = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        self.cat_cols_ = [c for c in X.columns if c not in self.num_cols_]
        self.statistics_ = {}
        if self.num_cols_:
            self.num_imp_ = self._imputer(self.numeric_strategy).fit(X[self.num_cols_])
            self.statistics_.update(dict(zip(self.num_cols_, self.num_imp_.statistics_)))
        if self.cat_cols_:
            self.cat_imp_ = self._imputer(self.categorical_strategy).fit(X[self.cat_cols_].astype(object))
            self.statistics_.update(dict(zip(self.cat_cols_, self.cat_imp_.statistics_)))
        return self

    def transform(self, X):
        out = pd.DataFrame(index=X.index)
        if self.num_cols_:
            out[self.num_cols_] = self.num_imp_.transform(X[self.num_cols_])
        if self.cat_cols_:
            out[self.cat_cols_] = self.cat_imp_.transform(X[self.cat_cols_].astype(object))
        return out[self.columns_]


class CenteredProducts(TransformerMixin, BaseEstimator):
    """Append products of mean-centred columns; the centring means are learned in fit(). Helper inputs listed in `drop` are removed afterwards."""

    def __init__(self, pairs=(), drop=()):
        self.pairs = pairs
        self.drop = drop

    def fit(self, X, y=None):
        cols = sorted({c for a, b, _ in self.pairs for c in (a, b)})
        self.means_ = {c: float(np.mean(X[c].to_numpy(float))) for c in cols}
        return self

    def transform(self, X):
        out = X.copy()
        for a, b, name in self.pairs:
            out[name] = (X[a].to_numpy(float) - self.means_[a]) * (X[b].to_numpy(float) - self.means_[b])
        return out.drop(columns=list(self.drop))


class WindOneHot(TransformerMixin, BaseEstimator):
    """One-hot encoding of the 16 compass directions (reference level N): no artificial ordering is imposed on the wind direction."""

    def __init__(self, column=C.WIND_DIR, levels=tuple(C.WD_LEVELS)):
        self.column = column
        self.levels = levels

    def fit(self, X, y=None):
        self.encoder_ = OneHotEncoder(categories=[list(self.levels)], drop="first", handle_unknown="error")
        self.encoder_.fit(X[[self.column]].astype(object))
        self.dummy_names_ = [f"{self.column}_{lv}" for lv in self.levels[1:]]
        return self

    def transform(self, X):
        Z = self.encoder_.transform(X[[self.column]].astype(object))
        Z = Z.toarray() if hasattr(Z, "toarray") else Z
        out = X.drop(columns=[self.column]).copy()
        for j, name in enumerate(self.dummy_names_):
            out[name] = Z[:, j]
        return out


class CorrTopK(TransformerMixin, BaseEstimator):
    """Keep the k columns with the largest |Pearson| / |Spearman| correlation with y, measured on the rows seen in fit() only."""

    def __init__(self, k=10, method="pearson"):
        self.k = k
        self.method = method

    def fit(self, X, y):
        Z, t = np.asarray(X, float), np.asarray(y, float)
        if self.method == "spearman":
            Z, t = np.apply_along_axis(stats.rankdata, 0, Z), stats.rankdata(t)
        elif self.method != "pearson":
            raise ValueError(self.method)
        Zc, tc = Z - Z.mean(0), t - t.mean()
        denom = np.sqrt((Zc ** 2).sum(0) * (tc ** 2).sum())
        r = np.divide((Zc * tc[:, None]).sum(0), denom, out=np.zeros(Z.shape[1]), where=denom > 0)
        self.scores_ = np.abs(r)
        order = np.argsort(-self.scores_, kind="stable")
        self.support_idx_ = np.sort(order[: min(self.k, Z.shape[1])])
        self.columns_ = list(X.columns[self.support_idx_]) if hasattr(X, "columns") else None
        return self

    def transform(self, X):
        return X.iloc[:, self.support_idx_] if hasattr(X, "iloc") else np.asarray(X)[:, self.support_idx_]


# ====================================================================== pipeline builders
def build_pipeline(spec, model, topk=None):
    """Leakage-safe pipeline of a feature-set spec (see features.build_specs). `topk` = {'k':..., 'method':...} adds the correlation filter."""
    steps = [("cols", ColumnSelector(F.spec_columns(spec))), ("impute", DFImputer())]
    if spec["pairs"]:
        steps.append(("products", CenteredProducts(spec["pairs"], drop=spec["helpers"])))
    if spec["categorical"]:
        steps.append(("wind", WindOneHot()))
    if topk is not None:
        steps.append(("select", CorrTopK(**topk)))
    steps += [("scale", StandardScaler()), ("model", model)]
    return Pipeline(steps)


def pipeline_until(pipe, step_name):
    """Sub-pipeline with the steps BEFORE `step_name` (e.g. the design matrix before scaling)."""
    names = [n for n, _ in pipe.steps]
    return Pipeline(pipe.steps[: names.index(step_name)])


def design_matrix(spec, X, y=None):
    """Design matrix (before scaling) of a feature-set spec, fitted on the given rows; returned as a DataFrame."""
    dummy = build_pipeline(spec, model="passthrough")
    pre = pipeline_until(dummy, "scale")
    return pre.fit_transform(X, y)
