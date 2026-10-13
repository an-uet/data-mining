"""Orchestration of the full experiment (protocol order). `Experiment().run_all()` runs every stage; stages can also be run one by one.

Protocol: split chronologically -> EDA / feature design / tuning / selection on TRAINING rows only -> freeze the design (SHA-256) ->
open the test vault once -> evaluate the frozen models -> analysis.  Nothing after the freeze can change a model, feature or hyper-parameter.
"""
import datetime as dt
import hashlib
import json
import platform
import time
import warnings

import joblib
import matplotlib as mpl
import numpy as np
import pandas as pd
import scipy
import sklearn
import statsmodels
from scipy import stats
from sklearn.base import clone
from statsmodels.tsa.stattools import acf, pacf

from . import config as C
from . import data as D
from . import evaluation as E
from . import features as F
from . import models as M
from . import plotting as G
from . import preprocessing as P
from . import reporting as R
from . import statistics as S

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

ABLATION_PLAN = [
    ("A", "Set A — raw features", "main"),
    ("B", "Set B — top-k correlated (raw pool)", "main"),
    ("Bp", "Set B′ — top-k correlated (full candidate pool)", "main"),
    ("C", "Set C — engineered (final)", "main"),
    ("C-temporal", "C − temporal", "group"),
    ("C-lag", "C − lag", "group"),
    ("C-rolling", "C − rolling", "group"),
    ("C-interaction", "C − interaction", "group"),
    ("C-history", "C − lag − rolling (keeps PM2.5(t))", "group"),
    ("C-pm25", "C − PM2.5 history (current, lag, rolling)", "group"),
    ("C-env", "C − environment (pollutants, meteorology, interactions)", "group"),
    ("C-pollutants", "C − pollutants", "group"),
    ("C-meteo", "C − meteorology / wind", "group"),
    ("C-envdiff", "C − derived differences", "group"),
    ("A-pm25", "A − PM2.5(t)  [sensitivity]", "sensitivity"),
]
GROUP_DROPS = {"C-temporal": ("temporal",), "C-lag": ("lag",), "C-rolling": ("rolling",), "C-interaction": ("interaction",), "C-history": ("history",),
               "C-pm25": ("pm25",), "C-env": ("env",), "C-pollutants": ("pollutants",), "C-meteo": ("meteo",), "C-envdiff": ("env_diff",)}


def _py(o):
    if isinstance(o, dict):
        return {str(k): _py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_py(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (pd.Timestamp, dt.datetime)):
        return str(o)
    return o


def _fp(p):
    return "< 1e-300" if p < 1e-300 else f"{p:.2g}"


def _pe(p):
    """p-value with its relation, for text of the form 'p {_pe(p)}': '< 1e-300' (float underflow) or '= 6.5e-72'."""
    return "< 1e-300" if p < 1e-300 else f"= {p:.2g}"


class Experiment:
    def __init__(self, data_path=None, fresh_log=True):
        self.data_path = data_path
        for d in (C.FIGURES, C.TABLES, C.MODELS, C.LOGS):
            d.mkdir(parents=True, exist_ok=True)
        self.t0 = time.time()
        self.st = {}
        self.log_path = C.LOGS / "run_all.log"
        if fresh_log:
            self.log_path.write_text("")

    # ---------------------------------------------------------------- utilities
    def log(self, msg=""):
        print(msg, flush=True)
        with open(self.log_path, "a") as f:
            f.write(msg + "\n")

    def header(self, title):
        self.log("\n" + "=" * 100 + f"\n{title}   [{(time.time() - self.t0) / 60:.1f} min]\n" + "=" * 100)

    def table(self, df, stem, caption, **kw):
        R.save_table(df, stem, caption, **kw)
        return df

    # ================================================================== 1. data inspection
    def stage_inspect(self):
        self.header("1. DATA INSPECTION (structure and data quality of the file as delivered)")
        raw = D.load_raw(self.data_path)
        rep = D.inspect_raw(raw)
        df, was_sorted = D.build_datetime(raw)
        tl = D.validate_timeline(df)
        df, ids = D.drop_identifier_columns(df)
        prov = D.provenance(self.data_path)
        self.log(f"source      : {prov['name']}  (DOI {prov['doi']})  | station {prov['station']} | file sha256 {prov['sha256'][:16]}...")
        self.log(f"NOTE        : {prov['note']}")
        self.log(f"rows        : {rep['rows']:,} | columns: {rep['columns']} -> {rep['column_names']}")
        self.log("dtypes      : " + ", ".join(f"{r.column}:{r.dtype}" for r in rep["dtypes"].itertuples()))
        self.log("missing     : " + ", ".join(f"{r.column}={r.missing} ({r.missing_pct:.1f}%)" for r in rep["dtypes"].itertuples() if r.missing))
        self.log(f"duplicates  : {rep['duplicate_rows']} duplicated rows (ignoring the 'No' counter: {rep['duplicate_rows_ignoring_No']}); duplicated timestamps: {tl['duplicated_timestamps']}")
        self.log(f"time range  : {tl['start']} -> {tl['end']} ({tl['span_days']} days) | already sorted in file: {was_sorted} | sampling: {tl['sampling_intervals']} | "
                 f"missing timestamps: {tl['missing_timestamps']}")
        ts = rep["target_stats"]
        self.log(f"target PM2.5 (global descriptive summary, ug/m3): n={ts['count']:,} missing={ts['missing']} mean={ts['mean']:.1f} std={ts['std']:.1f} min={ts['min']:.0f} "
                 f"median={ts['median']:.0f} max={ts['max']:.0f} skew={ts['skew']:.2f}")
        self.log(f"identifier columns removed: {ids['dropped']} (station is constant = {ids['constant_values'].get('station')}); wind direction levels: {len(rep['wd_levels']) - int(np.nan in rep['wd_levels'])}")
        self.table(rep["dtypes"], "table_S01_schema_and_missing_values", "Schema of the raw file: dtype and missing values of every column (whole file, structural check).", float_fmt="{:.2f}")
        (C.LOGS / "data_provenance.json").write_text(json.dumps(_py({**prov, "timeline": tl, "identifier_columns": ids}), indent=2))
        self.st.update(raw_report=rep, timeline=tl, df_full=df, provenance=prov)
        return df

    # ================================================================== 2. target + chronological split
    def stage_split(self):
        self.header("2. TARGET CONSTRUCTION AND CHRONOLOGICAL SPLIT")
        df = D.make_target(self.st.pop("df_full"))
        train_raw, vault, info = D.chronological_split(df)
        self.log("target      : target(t) = PM2.5(t+1) (built from the raw series, never filled); the final hour (no t+1) is removed; asserted: target[:-1] == PM2.5[1:]")
        self.log(f"rows        : {info['n_total']:,} after target construction")
        self.log(f"TRAIN       : {info['n_train']:,} hours  {info['train_start']} -> {info['train_end']}")
        self.log(f"TEST        : {info['n_test']:,} hours  {info['test_start']} -> {info['test_end']}   [locked in the vault]")
        self.log("chronology  : every training timestamp precedes every test timestamp; contiguous; shuffled = False. The full frame is deleted from memory now.")
        del df
        rep_ = self.st["raw_report"]
        G.fig_pipeline(rep_["rows"], rep_["columns"], C.TEST_FRACTION, C.N_SPLITS)
        (C.LOGS / "split_info.json").write_text(json.dumps(info, indent=2))
        self.st.update(train_raw=train_raw, vault=vault, split=info)

    # ================================================================== 3. training matrices (features + eligibility) and the missing-value policy
    def stage_training_data(self):
        self.header("3. TRAINING MATRICES AND MISSING-VALUE POLICY (training rows only)")
        tr = self.st["train_raw"]
        feats_all = F.build_candidate_features(tr)
        elig = F.eligibility(tr, feats_all)
        X = feats_all.loc[elig].reset_index(drop=True)
        y = tr.loc[elig, C.TARGET].reset_index(drop=True)
        meta = pd.DataFrame({"datetime": tr.loc[elig, "datetime"].to_numpy()})
        meta["target_time"] = meta["datetime"] + pd.Timedelta(hours=1)
        pm = tr[C.RAW_TARGET]
        runs, c = [], 0
        for v in pm.isna().to_numpy():
            if v:
                c += 1
            elif c:
                runs.append(c)
                c = 0
        if c:
            runs.append(c)
        runs = np.asarray(runs)
        limits = [0, 1, 2, 3, 6, 12, 24]
        counts = [int(F.eligibility(tr, F.build_candidate_features(tr, locf_limit=l)).sum()) for l in limits]
        pol = pd.DataFrame({"forward_fill_limit_hours": limits, "usable_training_rows": counts, "usable_share_pct": [100 * n / len(tr) for n in counts]})
        n_tgt = int(tr[C.TARGET].isna().sum())
        n_cur = int((tr[C.TARGET].notna() & pm.isna()).sum())
        self.log(f"training    : {len(tr):,} hours; PM2.5 missing {int(pm.isna().sum())} ({100 * pm.isna().mean():.2f} %) in {len(runs)} gaps (median {np.median(runs):.0f} h, 90th pct {np.percentile(runs, 90):.0f} h, max {runs.max()} h); "
                 f"{100 * (runs <= 3).mean():.0f} % of the gaps last <= 3 h")
        self.log("POLICY      : (1) the TARGET is never imputed - hours whose PM2.5(t+1) is missing are dropped; (2) the CURRENT PM2.5(t) must be observed (persistence needs it); "
                 f"(3) PM2.5 HISTORY features (lags, rolling) use a causal last-observation-carried-forward over gaps <= {C.LOCF_LIMIT} h, longer gaps make the row unusable; "
                 "(4) other pollutants / meteorology / wind direction are imputed INSIDE each training fold (SimpleImputer: median / most frequent). The rule depends on availability only "
                 "and is applied identically to every model and to the test period.")
        self.log(f"usable rows : {len(X):,} of {len(tr):,} training hours ({100 * len(X) / len(tr):.1f} %): missing target {n_tgt}, observed target but missing current PM2.5 {n_cur}, "
                 f"insufficient history {len(tr) - len(X) - n_tgt}   [first {C.WARMUP_HOURS} hours have no complete history]")
        self.table(pol, "table_S02_missing_value_policy", "Usable training rows as a function of the forward-fill limit applied to the PM2.5 history (the chosen limit is 3 hours).", float_fmt="{:.1f}")
        G.fig_missingness(runs, limits, counts, len(tr))
        pct = {l: 100 * c / len(tr) for l, c in zip(limits, counts)}
        G.set_caption("figure_S04_missing_values_policy", "Missing-value policy (training rows). (a) The PM2.5 series has %d gaps; %.0f %% last at most 3 hours, the longest lasts %d hours. (b) Causally forward-filling the PM2.5 history over gaps of up to %d hours "
                      "keeps %.1f %% of the training hours (no filling: %.1f %%); a 12-hour limit would add only %.1f points and a 24-hour limit %.1f. The target is never filled; rows with unusable history are excluded identically for every model."
                      % (len(runs), 100 * (runs <= 3).mean(), runs.max(), C.LOCF_LIMIT, pct[C.LOCF_LIMIT], pct[0], pct[12] - pct[C.LOCF_LIMIT], pct[24] - pct[C.LOCF_LIMIT]))
        self.st.update(feats_train_all=feats_all, elig_train=elig, X_train=X, y_train=y, meta_train=meta, gap_runs=runs, policy=pol, n_train_raw=len(tr))

    # ================================================================== 4. TRAIN-ONLY EDA
    def stage_eda(self):
        self.header("4. EXPLORATORY DATA ANALYSIS (training rows only)")
        X, y, meta, tr = self.st["X_train"], self.st["y_train"], self.st["meta_train"], self.st["train_raw"]
        ys = y.to_numpy(float)
        self.log(f"target (next-hour PM2.5, usable training rows n={len(y):,}): mean {ys.mean():.1f} median {np.median(ys):.0f} std {ys.std(ddof=1):.1f} min {ys.min():.0f} "
                 f"max {ys.max():.0f} skew {stats.skew(ys):.2f}; P25/P75/P95/P99 = {np.percentile(ys, [25, 75, 95, 99]).round(0).tolist()}")
        e = pd.DataFrame({"target_time": meta["target_time"], "target": ys})
        G.fig_distribution(ys, ys.mean(), np.median(ys))
        G.set_caption("figure_02_pm25_distribution", "Distribution of the target (PM2.5 one hour ahead) in the %s usable training rows. The series is right-skewed (skewness %.2f): median %.0f, mean %.0f, 99th percentile %.0f and maximum %.0f µg/m³. "
                      "The heavy tail makes squared-error metrics (RMSE) much larger than MAE." % (format(len(ys), ","), stats.skew(ys), np.median(ys), ys.mean(), np.percentile(ys, 99), ys.max()))
        th0 = pd.DatetimeIndex(e["target_time"])
        hr0, mo0 = e.groupby(th0.hour)["target"].mean(), e.groupby(th0.month)["target"].mean()
        wk0, we0 = e.loc[th0.dayofweek < 5, "target"].mean(), e.loc[th0.dayofweek >= 5, "target"].mean()
        G.fig_temporal_pattern(e)
        import calendar
        G.set_caption("figure_03_pm25_temporal_pattern", "Training-only temporal structure of the target (clock of the target hour). (a) Daily cycle: the hourly mean ranges from %.0f (%d:00) to %.0f µg/m³ (%d:00), a spread of %.0f %% of the overall mean; the band is the inter-quartile range. "
                      "(b) Seasonal cycle: monthly means from %.0f (%s) to %.0f µg/m³ (%s); dots are medians. (c) Weekly pattern: weekday mean %.1f vs weekend %.1f µg/m³. These patterns motivate the cyclical temporal features (H3)."
                      % (hr0.min(), hr0.idxmin(), hr0.max(), hr0.idxmax(), 100 * (hr0.max() - hr0.min()) / ys.mean(), mo0.min(), calendar.month_name[mo0.idxmin()], mo0.max(), calendar.month_name[mo0.idxmax()], wk0, we0))
        daily = tr.assign(d=tr["datetime"].dt.floor("D")).groupby("d")[C.RAW_TARGET].mean()
        G.fig_time_series(daily)
        G.set_caption("figure_S01_pm25_over_time", f"Daily mean PM2.5 in the training period ({self.st['split']['train_start'][:10]} to {self.st['split']['train_end'][:10]}) with a 30-day trailing mean (past data only): daily means range from "
                      f"{daily.min():.0f} to {daily.max():.0f} µg/m³ with a strong seasonal cycle and irregular multi-day episodes.")
        env = pd.concat([e.reset_index(drop=True), X[["PM10", "NO2", "CO", "O3", "TEMP", "DEWP", "WSPM", "wd"]].reset_index(drop=True)], axis=1)
        G.fig_scatter_grid(env, [("PM10", "PM10 (µg/m³)"), ("NO2", "NO2 (µg/m³)"), ("CO", "CO (µg/m³)"), ("O3", "O3 (µg/m³)"), ("TEMP", "temperature (°C)"), ("DEWP", "dew point (°C)")])
        rho = {c: stats.spearmanr(env[c], env["target"], nan_policy="omit")[0] for c in ("PM10", "NO2", "CO", "O3", "TEMP", "DEWP")}
        same10 = (X["PM10"] == X[C.CURRENT]) & X["PM10"].notna()
        lower10 = (X["PM10"] < X[C.CURRENT])
        G.set_caption("figure_S02_target_vs_pollutants_meteorology", "Next-hour PM2.5 against PM10, NO2, CO, O3, temperature and dew point (training rows; y clipped at 500 µg/m³). Hexagons: density; orange: median within 15 equal-count bins; "
                      "ρ = Spearman correlation (" + ", ".join("%s %+.2f" % (k, v) for k, v in rho.items()) + "). Data-quality caveat: PM10 equals PM2.5(t) exactly in %.1f %% of the hours and is even lower in %.1f %%, which is physically inconsistent." % (100 * same10.mean(), 100 * lower10.mean()))
        self.log(f"data quality: PM10 == PM2.5(t) exactly in {100 * same10.mean():.1f} % of the usable training hours ({int(same10.sum())} of {len(X)}) and PM10 < PM2.5(t) in {100 * lower10.mean():.2f} % - physically inconsistent (PM10 contains PM2.5); PM10-based features must be read with care")
        self.st["pm10_equal_share"], self.st["pm10_lower_share"] = float(same10.mean()), float(lower10.mean())
        G.fig_wind(env, C.WD_LEVELS)
        mw = env.groupby("wd")["target"].median().reindex(C.WD_LEVELS).sort_values()
        sp_bins = pd.cut(env["WSPM"], [0, 0.5, 1, 1.5, 2, 3, 4, 6, 20], include_lowest=True)
        sm = env.groupby(sp_bins, observed=True)["target"].mean()
        lab = lambda iv: "%g-%g m/s" % (max(iv.left, 0), iv.right)
        G.set_caption("figure_S03_wind_effects", "Training-only wind effects. (a) Median next-hour PM2.5 by wind direction: lowest for %s (%.0f) and %s (%.0f), highest for %s (%.0f) µg/m³; direction matters, hence one-hot encoding without an artificial ordering. "
                      "(b) Mean and median PM2.5 by wind speed at t: the mean is highest in the %s bin (%.0f) and lowest in the %s bin (%.0f µg/m³), i.e. ventilation lowers PM2.5, which motivates the wind x PM interactions."
                      % (mw.index[0], mw.iloc[0], mw.index[1], mw.iloc[1], mw.index[-1], mw.iloc[-1], lab(sm.idxmax()), sm.max(), lab(sm.idxmin()), sm.min()))
        cur = X[C.CURRENT].to_numpy(float)
        a = acf(tr[C.RAW_TARGET].to_numpy(float), nlags=168, missing="conservative", fft=False)
        pa = pacf(tr[C.RAW_TARGET].ffill(limit=C.LOCF_LIMIT).dropna().to_numpy(float), nlags=168, method="ldb")   # Levinson-Durbin (fast)
        G.fig_acf(a, pa)
        G.set_caption("figure_S06_autocorrelation", "Autocorrelation and partial autocorrelation of hourly PM2.5 (training rows). The autocorrelation falls from %.2f at lag 1 to %.2f at 24 h and %.3f at one week. The partial autocorrelation is dominated by lag 1 (%.2f) with a negative "
                      "lag-2 term (%.2f); beyond lag 2 no partial autocorrelation exceeds %.2f in absolute value (lag 24: %.3f, lag 168: %.3f), so the daily and weekly lags are expected to add little once the most recent hours are known."
                      % (a[1], a[24], a[168], pa[1], pa[2], np.abs(pa[3:]).max(), pa[24], pa[168]))
        self.log(f"autocorrelation of PM2.5 (training): lag1 {a[1]:.3f} lag3 {a[3]:.3f} lag24 {a[24]:.3f} lag168 {a[168]:.3f}; corr(PM2.5(t), PM2.5(t+1)) = {np.corrcoef(cur, ys)[0, 1]:.4f}")
        # --- association of every candidate with the target (Pearson / Spearman), training rows only
        pool = F.build_specs([])["pool"]
        Dp = P.design_matrix(pool, X)
        raw_names = set(F.build_specs([])["A"]["numeric"]) | {c for c in Dp.columns if c.startswith("wd_")}
        rows = []
        for c in Dp.columns:
            s = Dp[c]
            if s.std() == 0:
                continue
            rows.append({"feature": c, "set": "raw" if c in raw_names else "engineered", "group": F.group_of_design_column(c), "pearson": float(stats.pearsonr(s, ys)[0]),
                         "spearman": float(stats.spearmanr(s, ys)[0])})
        assoc = pd.DataFrame(rows)
        assoc["abs_pearson"], assoc["abs_spearman"] = assoc["pearson"].abs(), assoc["spearman"].abs()
        assoc = assoc.sort_values("abs_spearman", ascending=False).reset_index(drop=True)
        self.table(assoc.drop(columns=["abs_pearson", "abs_spearman"]), "table_S03_feature_target_association", "Pearson and Spearman correlation of every candidate feature with the next-hour PM2.5 (training rows only).", float_fmt="{:.3f}")
        G.fig_target_ranking(assoc)
        sp = assoc.set_index("feature")["spearman"]
        wd_cols = [f for f in sp.index if f.startswith("wd_")]
        G.set_caption("figure_05_feature_target_correlation_ranking", "Absolute Pearson and Spearman correlation of every candidate feature with the next-hour PM2.5 (training rows only). The current PM2.5 (Spearman ρ = %.2f), the first lag (%.2f) and the short rolling means "
                      "(ρ ≥ %.2f for the 3- and 6-hour windows) rank highest; among the other raw measurements PM10 (%.2f), CO (%.2f) and NO2 (%.2f) are strongest. Temperature (%+.2f), pressure (%+.2f) and rain (%+.2f) are only weakly related, and %d of the %d wind-direction dummies have |ρ| < 0.10."
                      % (sp[C.CURRENT], sp["PM25_lag_1"], min(sp["PM25_roll_mean_3"], sp["PM25_roll_mean_6"]), sp["PM10"], sp["CO"], sp["NO2"], sp["TEMP"], sp["PRES"], sp["RAIN"], int((sp[wd_cols].abs() < 0.10).sum()), len(wd_cols)))
        self.log("top-12 |Spearman| with the target: " + ", ".join(f"{r.feature} ({r.spearman:+.2f})" for r in assoc.head(12).itertuples()))
        env_raw = assoc[(assoc["set"] == "raw") & (~assoc["feature"].isin([C.CURRENT]))].head(6)
        self.log("strongest raw environment variables: " + ", ".join(f"{r.feature} (r={r.pearson:+.2f}, rho={r.spearman:+.2f})" for r in env_raw.itertuples()))
        Dc = pd.concat([pd.Series(ys, name="target(t+1)"), X[[C.CURRENT, "PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "RAIN", "WSPM"]].rename(columns={C.CURRENT: "PM2.5(t)"})], axis=1)
        G.fig_corr_heatmaps(Dc)
        G.set_caption("figure_04_correlation_heatmap_train", "Training-only Pearson (left) and Spearman (right) correlation matrices of the raw measurements and the target (next-hour PM2.5). Pearson correlation with the target: PM2.5(t) %.2f, PM10 %.2f, CO %.2f, NO2 %.2f; "
                      "TEMP and DEWP correlate at %.2f with each other and PM10 with PM2.5(t) at %.2f." % (Dc["target(t+1)"].corr(Dc["PM2.5(t)"]), Dc["target(t+1)"].corr(Dc["PM10"]), Dc["target(t+1)"].corr(Dc["CO"]), Dc["target(t+1)"].corr(Dc["NO2"]),
                                                                                                            Dc["TEMP"].corr(Dc["DEWP"]), Dc["PM10"].corr(Dc["PM2.5(t)"])))
        self.log("raw-variable redundancy: corr(TEMP, DEWP) = %.2f, corr(PM2.5(t), PM10) = %.2f, corr(PM10, CO) = %.2f, corr(NO2, CO) = %.2f" % (
            Dc["TEMP"].corr(Dc["DEWP"]), Dc["PM2.5(t)"].corr(Dc["PM10"]), Dc["PM10"].corr(Dc["CO"]), Dc["NO2"].corr(Dc["CO"])))
        th = pd.DatetimeIndex(meta["target_time"])
        hr = pd.Series(ys).groupby(th.hour).mean()
        mo = pd.Series(ys).groupby(th.month).mean()
        self.log(f"daily cycle (mean by target hour): max {hr.idxmax()}h ({hr.max():.0f}), min {hr.idxmin()}h ({hr.min():.0f}); seasonal cycle: max month {mo.idxmax()} ({mo.max():.0f}), min month {mo.idxmin()} ({mo.min():.0f}); "
                 f"weekday {ys[th.dayofweek < 5].mean():.1f} vs weekend {ys[th.dayofweek >= 5].mean():.1f}")
        med_wd = pd.Series(ys).groupby(env["wd"].to_numpy()).median().sort_values()
        self.log(f"median target by wind direction: cleanest {med_wd.index[0]} ({med_wd.iloc[0]:.0f}), {med_wd.index[1]} ({med_wd.iloc[1]:.0f}); dirtiest {med_wd.index[-1]} ({med_wd.iloc[-1]:.0f})")
        ws = X["WSPM"]
        slopes = {}
        for lab, m in [("low", ws <= ws.quantile(1 / 3)), ("mid", (ws > ws.quantile(1 / 3)) & (ws <= ws.quantile(2 / 3))), ("high", ws > ws.quantile(2 / 3))]:
            slopes[lab] = float(np.polyfit(cur[m.to_numpy()], ys[m.to_numpy()], 1)[0])
        self.log(f"ventilation: slope of PM2.5(t+1) on PM2.5(t) by wind-speed tercile: {slopes} (supports the WSPM x PM2.5(t) interaction)")
        self.st.update(assoc=assoc, eda_slopes=slopes, acf=a)

    # ================================================================== 5. pre-registration (before any model is compared)
    def stage_preregister(self):
        self.header("5. PRE-REGISTRATION OF HYPOTHESES AND DECISION RULES (written before any model is compared)")
        hyp = {
            "H1": {"statement": "Recent PM2.5 concentrations are strong predictors of next-hour PM2.5.", "test": "Set C vs C - PM2.5 history (current + lags + rolling); joint Wald test of the history block"},
            "H2": {"statement": "Meteorological and pollutant measurements provide additional predictive information beyond historical PM2.5.", "test": "Set C vs C - environment (pollutants, meteorology, wind direction, derived differences, interactions); joint Wald test of the environment block"},
            "H3": {"statement": "PM2.5 has systematic hourly and weekly temporal patterns that can be captured using temporal features.", "test": "Set C vs C - temporal (calendar terms and their hour interactions); joint Wald test; hourly/weekly/monthly decomposition reported"},
            "H4": {"statement": "Rolling statistics of recent PM2.5 provide more stable predictive information than individual lag values.",
                   "test": "rolling-only history (C - lag) vs lag-only history (C - rolling), both keep PM2.5(t); predictive criterion = rolling-only has the lower CV-MAE by >= 1 % in >= 4 of 5 folds; "
                           "inferential criterion = Diebold-Mariano test on out-of-fold absolute errors favours rolling-only (p < 0.05); stability (fold-to-fold std of MAE, coefficient spread) is reported alongside"}}
        prereg = {"written_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "hypotheses": hyp,
                  "decision_rule": {"min_relative_cv_mae_increase_when_group_removed": C.HYP_MIN_REL_INCREASE, "min_folds_worse_out_of_5": C.HYP_MIN_FOLDS, "alpha": C.HYP_ALPHA,
                                    "verdicts": "Supported = predictive AND inferential criterion; Partly supported = exactly one; Not supported = neither (verdicts use training CV / training inference only)"},
                  "model_selection_rule": {"metric": "mean CV-MAE (primary)", "tie_tolerance": f"{C.TIE_SE} standard error of the best model's fold MAEs",
                                           "among_tied_families": "decided by the data properties of the TRAINING design matrix: correlated groups AND many weak predictors -> ElasticNet; strong multicollinearity -> Ridge; many weak predictors -> Lasso; else OLS",
                                           "thresholds": {"max_vif": C.PROP_VIF, "cluster_abs_rho": C.PROP_CLUSTER_RHO, "min_clusters": C.PROP_MIN_CLUSTERS, "weak_abs_spearman": C.PROP_WEAK_RHO, "weak_share": C.PROP_WEAK_SHARE}},
                  "missing_value_policy": {"target": "never imputed; rows with missing PM2.5(t+1) dropped", "current_pm25": "must be observed", "pm25_history_locf_limit_hours": C.LOCF_LIMIT,
                                           "other_variables": "SimpleImputer inside each training fold (median numeric, most frequent wind direction)"},
                  "derived_feature_rule": f"a Group-4 derived quantity is kept only if |Spearman| with the target or with the next-hour change is >= {C.DERIVED_MIN_ABS_RHO} on the training rows",
                  "decision_threshold": {"value_ug_m3": C.DECISION_THRESHOLD, "status": "PROPOSED operational tolerance, not an authoritative standard; sensitivity grid reported", "grid": C.THRESHOLD_GRID},
                  "test_policy": "locked until the design is frozen; evaluated once; never used for selection"}
        (C.LOGS / "preregistration.json").write_text(json.dumps(_py(prereg), indent=2))
        for h, d in hyp.items():
            self.log(f"{h}: {d['statement']}")
        self.log(f"decision rule: group removal must raise CV-MAE by >= {100 * C.HYP_MIN_REL_INCREASE:.0f} % in >= {C.HYP_MIN_FOLDS}/5 folds AND joint HAC Wald p < {C.HYP_ALPHA}")
        self.log(f"model selection: primary metric CV-MAE, ties within {C.TIE_SE} SE, then the data properties of the training design decide (Ridge / Lasso / ElasticNet)")
        self.st["prereg"] = prereg

    # ================================================================== 6. feature design (groups, derived rule, leakage checks)
    def stage_features(self):
        self.header("6. FEATURE ENGINEERING: GROUPS, DERIVED-FEATURE RULE, FEATURE SETS A/B/C AND LEAKAGE CHECKS")
        X, y = self.st["X_train"], self.st["y_train"]
        dtab = F.derived_candidate_table(X, y)
        derived = dtab.loc[dtab["kept"], "candidate"].tolist()
        specs = F.build_specs(derived)
        self.table(dtab, "table_S04_derived_feature_candidates", "Group 4: candidate derived quantities and their training-only Spearman correlation with the target and with the next-hour change; kept if the larger |rho| >= 0.10.", float_fmt="{:.3f}")
        for r in dtab.itertuples():
            self.log(f"  derived candidate {r.candidate:16s} rho(target)={r.rho_with_target:+.3f} rho(change)={r.rho_with_next_hour_change:+.3f} -> {'KEPT' if r.kept else 'dropped'}")
        g2 = F.feature_group_table(specs, dtab, acf1=float(self.st["acf"][1]))
        self.table(g2[["group", "n_features", "features", "rationale", "tests"]], "table_2_feature_groups_and_rationale", "Feature groups, members, rationale and the hypothesis each group tests.", col_format="lrp{4.6cm}p{6.4cm}l")
        DA, DC = P.design_matrix(specs["A"], X), P.design_matrix(specs["C"], X)
        self.log(f"Set A: {DA.shape[1]} design columns (11 raw numeric + 15 wind-direction dummies); Set C: {DC.shape[1]} design columns "
                 f"({len(specs['C']['numeric'])} numeric main effects + {len(specs['C']['pairs'])} centred interactions + 15 wind dummies; helper input DEWP: {specs['C']['helpers']})")
        self.log(f"collinearity guard: rank(Set C design) = {S.rank_of(DC)} of {DC.shape[1]} (full rank; DEWP is represented through TEMP - DEWP, otherwise TEMP, DEWP and their difference would be exactly collinear)")
        cv = M.make_cv()
        cvj = []
        for k, (tr_i, va_i) in enumerate(cv.split(X), 1):
            cvj.append({"fold": k, "train_rows": [int(tr_i[0]), int(tr_i[-1])], "val_rows": [int(va_i[0]), int(va_i[-1])],
                        "train_time": [str(self.st["meta_train"]["datetime"].iloc[tr_i[0]]), str(self.st["meta_train"]["datetime"].iloc[tr_i[-1]])],
                        "val_time": [str(self.st["meta_train"]["datetime"].iloc[va_i[0]]), str(self.st["meta_train"]["datetime"].iloc[va_i[-1]])]})
        cvt = pd.DataFrame([{"fold": r["fold"], "train_rows": r["train_rows"][1] + 1, "train_to": r["train_time"][1][:10], "val_rows": r["val_rows"][1] - r["val_rows"][0] + 1,
                             "val_from": r["val_time"][0][:10], "val_to": r["val_time"][1][:10]} for r in cvj])
        self.table(cvt, "table_S05_cv_folds", "Expanding-window TimeSeriesSplit folds on the usable training rows (every validation block lies strictly after its training block).")
        G.fig_split_timeline(self.st["split"], cvj)
        G.set_caption("figure_S08_split_and_cv_folds", f"Chronological split ({self.st['split']['n_train']:,} training hours {self.st['split']['train_start'][:10]} to {self.st['split']['train_end'][:10]}; {self.st['split']['n_test']:,} test hours "
                      f"{self.st['split']['test_start'][:10]} to {self.st['split']['test_end'][:10]}, locked) and the {len(cvj)} expanding-window folds on the {len(X):,} usable training rows: fold k is fitted on all rows up to its validation block, which lies strictly after it "
                      f"({'; '.join('fold %d: %s rows, validation %s to %s' % (r['fold'], format(r['train_rows'][1] + 1, ','), r['val_time'][0][:10], r['val_time'][1][:10]) for r in cvj)}). All models, feature sets and ablations use these same folds.")
        (C.MODELS / "cv_splits.json").write_text(json.dumps({"scheme": "sklearn TimeSeriesSplit(n_splits=5), expanding window, gap=0; ranges inclusive in the usable-training-row index", "n_rows": len(X), "folds": cvj}, indent=2))
        np.savez(C.MODELS / "split_indices.npz", train_positions=np.arange(self.st["n_train_raw"]), usable_train_positions=np.flatnonzero(self.st["elig_train"]),
                 test_first_position=np.array([self.st["split"]["split_index"]]))
        self.log("running check_no_data_leakage() ...")
        chk = F.check_no_data_leakage(train_frame=self.st["train_raw"], X_train=X, y_train=y, spec=specs["C"], cv=cv, vault=self.st["vault"], verbose=False)
        for c, ok in zip(chk["check"], chk["passed"]):
            self.log(("  PASS  " if ok else "  FAIL  ") + c)
        self.table(chk, "table_S06_leakage_checks", "Results of check_no_data_leakage(): perturbation, identity, prefix-invariance and fold-purity checks.")
        self.st.update(derived=derived, derived_table=dtab, specs=specs, cv=cv, cv_json=cvj, leakage=chk, design_A=DA, design_C=DC)

    # ================================================================== 7. baselines
    def stage_baselines(self):
        self.header("7. BASELINES (time-series CV on the training rows)")
        X, y, cv = self.st["X_train"], self.st["y_train"], self.st["cv"]
        fits, summ = {}, {}
        for key, (label, feats, est, grid) in M.baseline_estimators(self.st["specs"]).items():
            t = time.time()
            gs = M.tune(est, grid, X, y, cv)
            fits[key], summ[key] = gs, {"label": label, "features": feats, **M.cv_summary(gs)}
            self.log(f"  {label:48s} CV MAE {M.pm(summ[key]['cv_mae'], summ[key]['cv_mae_std'])} | RMSE {M.pm(summ[key]['cv_rmse'], summ[key]['cv_rmse_std'])} | R2 {M.pm(summ[key]['cv_r2'], summ[key]['cv_r2_std'], 3)}   [{time.time() - t:.0f}s]")
        self.st.update(baseline_fits=fits, baseline_cv=summ)

    # ================================================================== 8. linear models + selection
    def stage_linear_models(self):
        self.header("8. LINEAR MODELS ON SET C: TUNING AND MODEL SELECTION (training CV only)")
        X, y, cv, specs = self.st["X_train"], self.st["y_train"], self.st["cv"], self.st["specs"]
        edge = []
        fits, summ = {}, {}
        for fam in M.FAMILIES:
            t = time.time()
            gs = M.tune_family(fam, lambda m: P.build_pipeline(specs["C"], m), X, y, cv, tag="Set C", edge_log=edge)
            coef = gs.best_estimator_.named_steps["model"].coef_
            fits[fam] = gs
            summ[fam] = {"model": fam, **M.cv_summary(gs), "nonzero": int((np.abs(coef) > 1e-8).sum()), "n_coef": int(coef.size)}
            summ[fam]["degenerate"] = bool(fam != "OLS" and np.isclose(summ[fam]["best_params"]["alpha"], float(np.min(gs.param_grid["model__alpha"]))))   # optimum = no shrinkage
            it = getattr(gs.best_estimator_.named_steps["model"], "n_iter_", None)
            self.log(f"  {fam:11s} CV MAE {M.pm(summ[fam]['cv_mae'], summ[fam]['cv_mae_std'])} | RMSE {M.pm(summ[fam]['cv_rmse'], summ[fam]['cv_rmse_std'])} | R2 {M.pm(summ[fam]['cv_r2'], summ[fam]['cv_r2_std'], 3)} | "
                     f"{summ[fam]['best_params']} | non-zero {summ[fam]['nonzero']}/{summ[fam]['n_coef']}" + (f" | solver iterations {it}" if it is not None else "") + f"   [{time.time() - t:.0f}s]")
        self.log("grid-edge log: " + (str(edge) if edge else "no optimum on a grid edge"))
        D_C = self.st["design_C"]
        props, vif = S.data_properties(D_C, y)
        sel = M.select_final_model(summ, props)
        fam = sel["family"]
        self.log("\nDATA PROPERTIES of the training design of Set C: " + f"max VIF {props['max_vif']:.1f} ({props['max_vif_feature']}), {props['n_vif_gt_10']} features > {C.PROP_VIF:g}; condition number {props['condition_number']:.0f}; "
                 f"{props['n_correlated_groups']} correlated groups (|r| >= {C.PROP_CLUSTER_RHO}); weak predictors {props['n_weak']}/{props['n_features']} ({100 * props['share_weak']:.0f} %)")
        for g in props["correlated_groups"]:
            self.log("   correlated group: " + ", ".join(g[:8]) + (" ..." if len(g) > 8 else ""))
        self.log("SELECTION: " + sel["why"])
        if sel["original"] != sel["family"]:
            pre = json.loads((C.LOGS / "preregistration.json").read_text())
            pre.setdefault("amendments", []).append({"written_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "stage": "after training CV of the linear families, before the design freeze (test set locked)",
                                                      "change": "a penalised family whose CV-optimal penalty is on the no-shrinkage edge of its grid is numerically OLS and is skipped in the property-implied preference list",
                                                      "reason": sel["why"], "original_outcome": sel["original"], "amended_outcome": sel["family"]})
            (C.LOGS / "preregistration.json").write_text(json.dumps(_py(pre), indent=2))
            self.log(f"  (pre-registered rule alone would have chosen {sel['original']}; the amendment is recorded in results/logs/preregistration.json)")
        mae_rank = sorted(summ, key=lambda f: summ[f]["cv_mae"])
        self.log("secondary check - families ranked by CV-RMSE: " + str(sorted([(f, round(float(summ[f]["cv_rmse"]), 3)) for f in summ], key=lambda t: t[1])))
        tab = pd.DataFrame([{"Model": f, "selected hyper-parameters": json.dumps(_py({k: (float(f"{v:.4g}") if isinstance(v, float) else v) for k, v in summ[f]["best_params"].items()})) if summ[f]["best_params"] else "-",
                             "CV MAE": M.pm(summ[f]["cv_mae"], summ[f]["cv_mae_std"]), "CV RMSE": M.pm(summ[f]["cv_rmse"], summ[f]["cv_rmse_std"]), "CV R²": M.pm(summ[f]["cv_r2"], summ[f]["cv_r2_std"], 4),
                             "non-zero coef.": f"{summ[f]['nonzero']}/{summ[f]['n_coef']}", "within 1 SE of best": f in sel["tied"], "selected": "★" if f == fam else ""} for f in M.FAMILIES])
        self.table(tab, "table_S07_linear_models_cv_selection", "Linear families on Set C: tuned hyper-parameters, time-series CV metrics and the outcome of the pre-registered selection rule.", col_format="llrrrrlc")
        pd.DataFrame(fits[fam].cv_results_).to_csv(C.TABLES / f"cv_results_{fam}.csv", index=False)
        bp = summ[fam]["best_params"]
        if fam != "OLS":
            G.fig_tuning_curve(pd.DataFrame(fits[fam].cv_results_), fam, bp["alpha"], bp.get("l1_ratio"))
            cvr = pd.DataFrame(fits[fam].cv_results_)
            mean_curve = -cvr[[c for c in cvr.columns if c.startswith("split") and c.endswith("_test_mae")]].mean(axis=1).to_numpy()
            alp = cvr["param_model__alpha"].astype(float).to_numpy()
            rows_ = np.ones(len(cvr), bool) if fam != "ElasticNet" else cvr["param_model__l1_ratio"].astype(float).to_numpy() == bp["l1_ratio"]
            al_s, mc_s = alp[rows_], mean_curve[rows_]
            flat = float(mc_s[al_s <= bp["alpha"] * (1 + 1e-9)].max() - mc_s.min())
            degen = [f for f in M.FAMILIES if f != "OLS" and summ[f]["degenerate"]]
            cap = ("Mean 5-fold time-series CV MAE of the selected family (%s) against the regularisation strength α%s. Orange: selected configuration (α = %.3g%s; %d of %d coefficients non-zero; CV-MAE %.2f µg/m³). "
                   "For every α up to the selected value the CV MAE is within %.2f µg/m³ of the minimum; at the largest α of the grid (%.3g) it is %.2f µg/m³."
                   % (fam, " (band: ±1 standard error)" if fam != "ElasticNet" else " for each l1_ratio", bp["alpha"], (", l1_ratio = %g" % bp["l1_ratio"]) if fam == "ElasticNet" else "", summ[fam]["nonzero"], summ[fam]["n_coef"], summ[fam]["cv_mae"],
                      flat, al_s.max(), float(mc_s[np.argmax(al_s)])))
            if degen:
                cap += " For %s the CV optimum lies on the no-shrinkage edge of the grid (table_S07): the penalty did not help, i.e. the fit is numerically OLS." % " and ".join(degen)
            G.set_caption("figure_06_tuning_curve", cap)
        sp = pd.DataFrame({"property": ["max VIF", "features with VIF > 10", "condition number", "correlated groups (|r|>=0.8)", "weak predictors (|Spearman|<0.10)", "share weak"],
                           "value": [props["max_vif"], props["n_vif_gt_10"], props["condition_number"], props["n_correlated_groups"], props["n_weak"], props["share_weak"]]})
        self.table(sp, "table_S08_training_design_properties", "Data properties of the Set C training design matrix that drive the pre-registered choice among tied linear families.", float_fmt="{:.3f}")
        self.st.update(linear_fits=fits, linear_cv=summ, props=props, vif_c=vif, selection=sel, final_family=fam, edge_log=edge)

    # ================================================================== 9. feature ablation
    def _variant(self, key):
        sp = self.st["specs"]
        if key == "A":
            return (lambda m: P.build_pipeline(sp["A"], m)), None
        if key == "B":
            return (lambda m: P.build_pipeline(sp["A"], m, topk={"k": 5, "method": "pearson"})), {"select__k": C.TOPK_VALUES, "select__method": C.TOPK_METHODS}
        if key == "Bp":
            return (lambda m: P.build_pipeline(sp["pool"], m, topk={"k": 5, "method": "pearson"})), {"select__k": C.TOPK_VALUES, "select__method": C.TOPK_METHODS}
        if key == "A-pm25":
            spec = F.drop_groups(sp["A"], "pm25")
        else:
            spec = F.drop_groups(sp["C"], *GROUP_DROPS[key])
        return (lambda m, s=spec: P.build_pipeline(s, m)), None

    def stage_ablation(self):
        self.header("9. FEATURE ABLATION (same folds, same linear family, same search; only the feature set changes)")
        X, y, cv, fam = self.st["X_train"], self.st["y_train"], self.st["cv"], self.st["final_family"]
        edge, abl = self.st["edge_log"], {}
        self.log(f"model family for every variant: {fam}")
        for key, label, role in ABLATION_PLAN:
            t = time.time()
            if key == "C":
                gs = self.st["linear_fits"][fam]
            else:
                mk, extra = self._variant(key)
                gs = M.tune_family(fam, mk, X, y, cv, extra=extra, tag=label, edge_log=edge)
            nf = int(gs.best_estimator_[:-1].transform(X).shape[1])
            nz = int((np.abs(gs.best_estimator_.named_steps["model"].coef_) > 1e-8).sum())
            abl[key] = {"label": label, "role": role, "gs": gs, "n_features": nf, "nonzero": nz, **M.cv_summary(gs)}
            self.log(f"  {label:58s} p={nf:2d} (non-zero {nz:2d}) | CV MAE {M.pm(abl[key]['cv_mae'], abl[key]['cv_mae_std'])} | RMSE {M.pm(abl[key]['cv_rmse'], abl[key]['cv_rmse_std'])} | {abl[key]['best_params']}   [{time.time() - t:.0f}s]")
        self.log("grid-edge log: " + (str(edge) if edge else "no optimum on a grid edge"))
        for k in ("B", "Bp"):
            self.log(f"  {abl[k]['label']}: selected {abl[k]['best_params']} -> {abl[k]['gs'].best_estimator_.named_steps['select'].columns_}")
        # out-of-fold predictions for the H4 comparison (rolling-only vs lag-only history)
        oof = {k: M.oof_predict(abl[k]["gs"].best_estimator_, X, y, cv) for k in ("C-lag", "C-rolling")}
        self.st.update(ablation=abl, oof=oof)

    # ================================================================== 10. statistical analysis (training) + hypotheses
    def stage_statistics(self):
        self.header("10. STATISTICAL ANALYSIS (training rows): predictive model vs OLS/HAC companion, VIF, bootstrap, hypotheses")
        X, y, fam = self.st["X_train"], self.st["y_train"], self.st["final_family"]
        pipe = self.st["linear_fits"][fam].best_estimator_
        Dm = P.pipeline_until(pipe, "scale").transform(X)
        names = list(Dm.columns)
        Z = pd.DataFrame(pipe.named_steps["scale"].transform(Dm), columns=names)
        coef = pd.Series(pipe.named_steps["model"].coef_, index=names)
        sd_y = float(y.std(ddof=1))
        cls, hac = S.ols_companion(y, Z)
        hac_alt = {L: S.ols_companion(y, Z, maxlags=L)[1] for L in (24, 168)}
        vif = S.vif_series(Dm)
        res = cls.resid
        self.log(f"final predictive model: {fam} {self.st['linear_cv'][fam]['best_params']} on Set C ({len(names)} design columns); OLS companion R2 (training) = {cls.rsquared:.4f}")
        from statsmodels.stats.diagnostic import acorr_ljungbox
        lb = acorr_ljungbox(res, lags=[1, 24, 168], return_df=True)
        self.log(f"OLS residuals: std {res.std():.1f}, skew {stats.skew(res):.2f}, excess kurtosis {stats.kurtosis(res):.1f}, Durbin-Watson {sm_dw(res):.3f}, Ljung-Box p (1/24/168) = {lb['lb_pvalue'].round(4).tolist()} -> classical SEs unreliable, HAC used")
        n_sig = {"classical": int((cls.pvalues[names] < 0.05).sum()), f"HAC({C.HAC_MAXLAGS})": int((hac.pvalues[names] < 0.05).sum()), "HAC(24)": int((hac_alt[24].pvalues[names] < 0.05).sum()), "HAC(168)": int((hac_alt[168].pvalues[names] < 0.05).sum())}
        self.log(f"coefficients significant at 5% (of {len(names)}): {n_sig}; median SE inflation HAC/classical = {float((hac.bse[names] / cls.bse[names]).median()):.2f}")
        t = time.time()
        boot = S.block_bootstrap_coefs(pipe, X, y)
        self.log(f"moving-block bootstrap: {boot.shape[0]} resamples x {boot.shape[1]} coefficients, blocks of {C.BOOT_BLOCK} h   [{time.time() - t:.0f}s]")
        ci = hac.conf_int()
        tab = pd.DataFrame({"feature": names, "group": [F.group_of_design_column(n) for n in names], "coef": coef.to_numpy(), "std_beta": coef.to_numpy() / sd_y,
                            "boot_lo": np.percentile(boot, 2.5, axis=0), "boot_hi": np.percentile(boot, 97.5, axis=0), "ols": hac.params[names].to_numpy(), "hac_se": hac.bse[names].to_numpy(),
                            "hac_lo": ci.loc[names, 0].to_numpy(), "hac_hi": ci.loc[names, 1].to_numpy(), "p_hac": hac.pvalues[names].to_numpy(), "p_holm": S.holm(hac.pvalues[names].to_numpy()),
                            "p_classical": cls.pvalues[names].to_numpy(), "vif": vif[names].to_numpy()})
        tab["abs_coef"] = tab["coef"].abs()
        tab = tab.sort_values("abs_coef", ascending=False).reset_index(drop=True)
        self.log(f"VIF of the final design: max {vif.max():.1f} ({vif.idxmax()}), {int((vif > 10).sum())} features > 10: {vif[vif > 10].sort_values(ascending=False).round(0).to_dict()}")
        self.log(f"shrinkage check (final model vs OLS companion): correlation of coefficients {np.corrcoef(tab['coef'], tab['ols'])[0, 1]:.4f}, max |difference| {np.abs(tab['coef'] - tab['ols']).max():.2f} ug/m3 per SD")
        self.log("top-10 coefficients (ug/m3 per SD): " + ", ".join(f"{r.feature} {r.coef:+.2f}" for r in tab.head(10).itertuples()))
        # ---- joint Wald tests and hypotheses
        hist_cols = [F.CURRENT] + F.LAG_COLS + F.ROLL_COLS
        env_cols = [c for c in names if c in F.POLLUTANTS or c in ("TEMP", "PRES", "RAIN", "WSPM") or c.startswith("wd_") or c in F.DERIVED_CANDIDATES or c in F.interaction_names()]
        hour_pairs = [n for a, b, n in F.INTERACTION_PAIRS if "hour_sin" in (a, b)]
        wald_groups = {"PM2.5 history (current + lags + rolling)": hist_cols, "environment (pollutants, meteorology, wind, derived, interactions)": env_cols,
                       "temporal (calendar terms + hour interactions)": F.TEMPORAL_COLS + hour_pairs, "interactions": F.interaction_names(),
                       "  daily terms (hour_sin, hour_cos)": ["hour_sin", "hour_cos"], "  weekly terms (dow_sin, dow_cos, is_weekend)": ["dow_sin", "dow_cos", "is_weekend"],
                       "  monthly terms (month_sin, month_cos)": ["month_sin", "month_cos"], "wind direction dummies": [c for c in names if c.startswith("wd_")]}
        wrows = []
        for g, cols in wald_groups.items():
            stat, p, q = S.wald_group(hac, [c for c in cols if c in names])
            wrows.append({"coefficient group": g, "df": q, "Wald chi2": stat, "p-value (HAC)": p})
        wald = pd.DataFrame(wrows)
        self.table(wald, "table_S09_group_wald_tests", "Joint Wald tests (HAC covariance) that all coefficients of a group are zero in the OLS companion on the standardised Set C design.", float_fmt="{:.4g}")
        abl = self.st["ablation"]
        wl = {r["coefficient group"]: r for r in wrows}
        hrows = []
        for h, (var, wg, label) in {"H1": ("C-pm25", "PM2.5 history (current + lags + rolling)", "PM2.5 history"),
                                    "H2": ("C-env", "environment (pollutants, meteorology, wind, derived, interactions)", "environment block"),
                                    "H3": ("C-temporal", "temporal (calendar terms + hour interactions)", "calendar terms")}.items():
            rel = abl[var]["cv_mae"] / abl["C"]["cv_mae"] - 1
            worse = int((abl[var]["mae_folds"] > abl["C"]["mae_folds"]).sum())
            vd, pr, inf = S.verdict(rel, worse, wl[wg]["p-value (HAC)"])
            hrows.append({"hypothesis": h, "comparison": f"C vs C - {label}", "delta_cv_mae_pct": 100 * rel, "folds_worse": worse, "predictive": pr, "inferential_test": f"joint Wald p {_pe(wl[wg]['p-value (HAC)'])}",
                          "inferential": inf, "verdict": vd})
        # H4: rolling-only (C-lag) vs lag-only (C-rolling)
        roll, lag = abl["C-lag"], abl["C-rolling"]
        rel4 = lag["cv_mae"] / roll["cv_mae"] - 1
        worse4 = int((lag["mae_folds"] > roll["mae_folds"]).sum())
        o_r, o_l = self.st["oof"]["C-lag"], self.st["oof"]["C-rolling"]
        ok = np.isfinite(o_r) & np.isfinite(o_l)
        dm_mean, dm_t, dm_p = S.diebold_mariano(y.to_numpy()[ok], o_r[ok], o_l[ok], "ae", maxlags=24)
        p4 = dm_p if dm_mean < 0 else 1.0
        vd4, pr4, inf4 = S.verdict(rel4, worse4, p4)
        hrows.append({"hypothesis": "H4", "comparison": "lag-only (C - rolling) relative to rolling-only (C - lag); negative = lag-only better", "delta_cv_mae_pct": 100 * rel4, "folds_worse": worse4, "predictive": pr4,
                      "inferential_test": f"DM on out-of-fold |error| (rolling-only minus lag-only): mean {dm_mean:+.3f}, p {_pe(dm_p)}", "inferential": inf4, "verdict": vd4})
        h4_stab = {"fold_std_mae_rolling_only": float(roll["mae_folds"].std(ddof=1)), "fold_std_mae_lag_only": float(lag["mae_folds"].std(ddof=1))}
        hyp = pd.DataFrame(hrows)
        self.st.update(boot=boot, coef_tab=tab, hac=hac, cls=cls, wald=wald, hyp_train=hyp, vif_final=vif, design_names=names, Z=Z, sd_y=sd_y, h4_stability=h4_stab, dm_h4=(dm_mean, dm_t, dm_p))
        for r in hrows:
            self.log(f"  {r['hypothesis']}: {r['comparison']}: dMAE {r['delta_cv_mae_pct']:+.2f} % (worse in {r['folds_worse']}/5 folds; predictive {'met' if r['predictive'] else 'not met'}); {r['inferential_test']} "
                     f"(inferential {'met' if r['inferential'] else 'not met'}) -> {r['verdict'].upper()}")
        self.log(f"  H4 stability: fold-to-fold std of CV-MAE rolling-only {h4_stab['fold_std_mae_rolling_only']:.2f} vs lag-only {h4_stab['fold_std_mae_lag_only']:.2f}")
        self.log("  H3 decomposition (Wald p): " + "; ".join(f"{k.strip()}: p {_pe(wl[k]['p-value (HAC)'])}" for k in ("  daily terms (hour_sin, hour_cos)", "  weekly terms (dow_sin, dow_cos, is_weekend)", "  monthly terms (month_sin, month_cos)")))

    # ================================================================== 11. freeze
    def stage_freeze(self):
        self.header("11. FREEZE THE DESIGN (nothing below this line may change a feature, model or hyper-parameter)")
        st, fam = self.st, self.st["final_family"]
        frozen = _py({"seed": C.SEED, "data": {"station": C.STATION, "sha256": st["provenance"]["sha256"], "n_rows": st["split"]["n_total"] + 1},
                      "split": st["split"], "missing_value_policy": {"locf_limit_hours": C.LOCF_LIMIT, "usable_training_rows": int(len(st["X_train"]))},
                      "cv": {"scheme": "TimeSeriesSplit(5) expanding", "folds": st["cv_json"]},
                      "feature_sets": {k: v for k, v in st["specs"].items()}, "derived_features": st["derived"],
                      "final_model": {"family": fam, "feature_set": "C", "hyperparameters": st["linear_cv"][fam]["best_params"], "selection": st["selection"]["why"]},
                      "linear_families": {f: {"hyperparameters": v["best_params"], "cv_mae": v["cv_mae"]} for f, v in st["linear_cv"].items()},
                      "ablation_variants": {k: {"label": v["label"], "hyperparameters": v["best_params"], "n_features": v["n_features"], "cv_mae": v["cv_mae"]} for k, v in st["ablation"].items()},
                      "baselines": {k: {"label": v["label"], "hyperparameters": v["best_params"], "cv_mae": v["cv_mae"]} for k, v in st["baseline_cv"].items()},
                      "slice_reference_from_training": E.training_reference(st["y_train"], st["X_train"]), "decision_threshold": C.DECISION_THRESHOLD,
                      "libraries": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "scikit-learn": sklearn.__version__,
                                    "statsmodels": statsmodels.__version__, "matplotlib": mpl.__version__}})
        js = json.dumps(frozen, indent=2, sort_keys=True)
        sha = hashlib.sha256(js.encode()).hexdigest()
        (C.MODELS / "frozen_design.json").write_text(js)
        (C.MODELS / "frozen_design.sha256").write_text(sha + "\n")
        fl = {"set_A_numeric": st["specs"]["A"]["numeric"], "set_A_categorical": st["specs"]["A"]["categorical"], "set_C_numeric": st["specs"]["C"]["numeric"], "set_C_pairs": [list(p) for p in st["specs"]["C"]["pairs"]],
              "set_C_design_columns": st["design_names"], "set_B_selected": list(st["ablation"]["B"]["gs"].best_estimator_.named_steps["select"].columns_),
              "set_B_prime_selected": list(st["ablation"]["Bp"]["gs"].best_estimator_.named_steps["select"].columns_)}
        (C.MODELS / "feature_lists.json").write_text(json.dumps(_py(fl), indent=2))
        (C.MODELS / "selected_hyperparameters.json").write_text(json.dumps(_py({"final_model": frozen["final_model"], "linear_families": frozen["linear_families"], "ablation_variants": frozen["ablation_variants"], "baselines": frozen["baselines"]}), indent=2))
        st.update(frozen=frozen, design_sha=sha, slice_ref=frozen["slice_reference_from_training"])
        self.log(f"design frozen. sha256 = {sha}")
        self.log(f"FINAL PREDICTOR: {fam} {st['linear_cv'][fam]['best_params']} on Set C ({len(st['design_names'])} design columns) | CV MAE {M.pm(st['linear_cv'][fam]['cv_mae'], st['linear_cv'][fam]['cv_mae_std'])}, "
                 f"CV RMSE {M.pm(st['linear_cv'][fam]['cv_rmse'], st['linear_cv'][fam]['cv_rmse_std'])}, CV R2 {M.pm(st['linear_cv'][fam]['cv_r2'], st['linear_cv'][fam]['cv_r2_std'], 4)}")

    # ================================================================== 12. single test evaluation
    def stage_final_test(self):
        self.header("12. FINAL TEST EVALUATION (the vault is opened once; models were fitted and tuned on training rows only)")
        st, fam = self.st, self.st["final_family"]
        assert hashlib.sha256((C.MODELS / "frozen_design.json").read_text().encode()).hexdigest() == st["design_sha"] == (C.MODELS / "frozen_design.sha256").read_text().strip(), "design changed after freezing"
        test_raw = st["vault"].unlock(st["design_sha"], "single evaluation of the frozen design")
        train_raw = st["train_raw"]
        k = len(train_raw)
        full = pd.concat([train_raw, test_raw], ignore_index=True)
        feats_full = F.build_candidate_features(full)
        pd.testing.assert_frame_equal(feats_full.iloc[:k].reset_index(drop=True), st["feats_train_all"].reset_index(drop=True), check_exact=True)   # the test period changed no training feature
        elig = F.eligibility(full, feats_full)
        te_mask = np.zeros(len(full), bool)
        te_mask[k:] = elig[k:]
        Xte = feats_full.loc[te_mask].reset_index(drop=True)
        yte = full.loc[te_mask, C.TARGET].reset_index(drop=True)
        meta = pd.DataFrame({"datetime": full.loc[te_mask, "datetime"].to_numpy()})
        meta["target_time"] = meta["datetime"] + pd.Timedelta(hours=1)
        meta["actual"] = yte.to_numpy(float)
        for c in (C.CURRENT, "PM10", "NO2", "CO", "TEMP", "WSPM", "wd", "PM25_roll_std_6"):
            meta[c] = Xte[c].to_numpy()
        meta["WSPM_prev"] = full["WSPM"].shift(1)[te_mask].to_numpy()
        meta["wd_prev"] = full["wd"].shift(1)[te_mask].to_numpy()
        meta["next_actual"] = full[C.TARGET].shift(-1)[te_mask].to_numpy()
        n_test_rows = len(full) - k
        self.log(f"test matrix : {len(Xte):,} usable hours of {n_test_rows:,} ({100 * len(Xte) / n_test_rows:.1f} %) | {meta['datetime'].iloc[0]} -> {meta['datetime'].iloc[-1]} | "
                 f"missing target {int(full.loc[k:, C.TARGET].isna().sum())}, other unusable {n_test_rows - len(Xte) - int(full.loc[k:, C.TARGET].isna().sum())}")
        self.log(f"test target : mean {yte.mean():.1f} median {yte.median():.0f} std {yte.std():.1f} min {yte.min():.0f} max {yte.max():.0f} (train: mean {st['y_train'].mean():.1f} median {st['y_train'].median():.0f} std {st['y_train'].std():.1f})")
        preds, metrics = {}, {}

        def score(key, est):
            p = np.asarray(est.predict(Xte), float)
            preds[key], metrics[key] = p, E.regression_metrics(yte, p)

        for kx, gs in st["baseline_fits"].items():
            score(("baseline", kx), gs.best_estimator_)
        for f, gs in st["linear_fits"].items():
            score(("linear", f), gs.best_estimator_)
        for kx, v in st["ablation"].items():
            score(("ablation", kx), v["gs"].best_estimator_)
        for kx, (label, feats, est, grid) in M.reference_estimators(st["specs"]).items():           # supplementary nonlinear references on Set C (fitted now on TRAINING rows only; never a candidate)
            gs = M.tune(est, grid, st["X_train"], st["y_train"], st["cv"])
            st.setdefault("ref_cv", {})[kx] = {"label": label, "features": feats, **M.cv_summary(gs)}
            score(("reference", kx), gs.best_estimator_)
        st.update(Xte=Xte, yte=yte, meta_test=meta, preds=preds, metrics=metrics, final_key=("linear", fam))
        fm = metrics[("linear", fam)]
        self.log(f"FINAL MODEL ON THE HELD-OUT TEST SET - {fam} {st['linear_cv'][fam]['best_params']} (Set C)")
        self.log(f"  MAE {fm['MAE']:.3f} | RMSE {fm['RMSE']:.3f} | R2 {fm['R2']:.4f} | median AE {fm['MedAE']:.2f} | bias (mean error) {fm['Bias']:+.2f} | residual std {fm['Resid_std']:.2f} | "
                 f"within +-{C.DECISION_THRESHOLD:g} ug/m3: {100 * fm['Within_tau']:.1f} % | negative predictions: {100 * fm['Neg_pred_share']:.2f} %")
        cvf = st["linear_cv"][fam]
        self.log(f"  CV (training) for comparison: MAE {M.pm(cvf['cv_mae'], cvf['cv_mae_std'])} | RMSE {M.pm(cvf['cv_rmse'], cvf['cv_rmse_std'])} | R2 {M.pm(cvf['cv_r2'], cvf['cv_r2_std'], 4)}")
        # ---- guard: the final model never saw test rows
        final_pipe = st["linear_fits"][fam].best_estimator_
        assert final_pipe.named_steps["scale"].n_samples_seen_ == len(st["X_train"]), "scaler was fitted on more than the training rows"
        self._write_test_tables()
        pd.DataFrame({"datetime": meta["datetime"], "target_time": meta["target_time"], "actual": yte, "pred_final": preds[st["final_key"]], "pred_persistence": preds[("baseline", "persistence")],
                      "pred_set_A_linear": preds[("ablation", "A")], "pred_mean": preds[("baseline", "mean")], "pred_rf_raw": preds[("baseline", "rf_raw")]}).to_csv(C.RESULTS / "test_predictions.csv", index=False)
        joblib.dump(final_pipe, C.MODELS / "final_model.joblib")
        names = st["design_names"]
        card = {"model": f"{fam} on Set C", "hyperparameters": st["linear_cv"][fam]["best_params"], "intercept": float(final_pipe.named_steps["model"].intercept_), "features": names,
                "scaler_mean": final_pipe.named_steps["scale"].mean_.tolist(), "scaler_scale": final_pipe.named_steps["scale"].scale_.tolist(), "coefficients_per_standardised_feature": final_pipe.named_steps["model"].coef_.tolist(),
                "imputer_statistics": _py(final_pipe.named_steps["impute"].statistics_), "centring_means_for_products": _py(final_pipe.named_steps["products"].means_),
                "wind_direction_reference_level": C.WD_LEVELS[0],
                "formula": "y_hat = intercept + sum_j coef_j * (x_j - mean_j) / scale_j; missing inputs are replaced by the imputer statistics; products use the centring means; wd is one-hot with reference N"}
        (C.MODELS / "final_model_card.json").write_text(json.dumps(_py(card), indent=2))
        (C.LOGS / "test_metrics.json").write_text(json.dumps(_py({f"{a}:{b}": v for (a, b), v in metrics.items()}), indent=2))

    def _write_test_tables(self):
        st, fam = self.st, self.st["final_family"]
        m, yte, preds = st["metrics"], st["yte"], st["preds"]
        rows_num, rows_fmt = [], []

        def add(label, feats, cv, key):
            t = m[key]
            rows_fmt.append({"Method": label, "CV MAE": M.pm(cv["cv_mae"], cv["cv_mae_std"]), "CV RMSE": M.pm(cv["cv_rmse"], cv["cv_rmse_std"]), "Test MAE": f"{t['MAE']:.2f}", "Test RMSE": f"{t['RMSE']:.2f}", "Test R²": f"{t['R2']:.4f}"})
            rows_num.append({"Method": label, "Features": feats, "CV_MAE": cv["cv_mae"], "CV_MAE_std": cv["cv_mae_std"], "CV_RMSE": cv["cv_rmse"], "CV_RMSE_std": cv["cv_rmse_std"], "CV_R2": cv["cv_r2"], "CV_R2_std": cv["cv_r2_std"],
                             "Test_MAE": t["MAE"], "Test_RMSE": t["RMSE"], "Test_R2": t["R2"], "Test_MedAE": t["MedAE"], "Test_bias": t["Bias"], "Test_within_tau": t["Within_tau"]})

        for kx, v in st["baseline_cv"].items():
            add(v["label"], v["features"], v, ("baseline", kx))
        for f in M.FAMILIES:
            add(f"{f}{' ★' if f == fam else ''} (linear, Set C)" if f != "ElasticNet" else f"Elastic Net{' ★' if f == fam else ''} (linear, Set C)", "Set C", st["linear_cv"][f], ("linear", f))
        num = pd.DataFrame(rows_num)
        fmt = pd.DataFrame(rows_fmt)
        bold = R.best_cells(num, {"CV MAE": ("CV_MAE", "min"), "CV RMSE": ("CV_RMSE", "min"), "Test MAE": ("Test_MAE", "min"), "Test RMSE": ("Test_RMSE", "min"), "Test R²": ("Test_R2", "max")},
                            decimals={"CV MAE": 2, "CV RMSE": 2, "Test MAE": 2, "Test RMSE": 2, "Test R²": 4})
        self.table(fmt, "table_3_model_comparison", "Baselines and linear models: 5-fold time-series CV on the training rows (mean ± std over folds) and a single evaluation on the held-out test period (µg/m³). "
                   "★ = selected final model. Bold = best value in the column over all rows.", col_format="lrrrrr", csv_df=num, bold=bold)
        st["table3_num"] = num
        self.log("\nTABLE 3 (best per column in bold in the LaTeX version):")
        self.log(fmt.to_string(index=False))
        # reference nonlinear models on Set C (supplementary)
        ref = pd.DataFrame([{"Method": v["label"], "CV MAE": M.pm(v["cv_mae"], v["cv_mae_std"]), "CV RMSE": M.pm(v["cv_rmse"], v["cv_rmse_std"]), "Test MAE": f"{m[('reference', k)]['MAE']:.2f}",
                             "Test RMSE": f"{m[('reference', k)]['RMSE']:.2f}", "Test R²": f"{m[('reference', k)]['R2']:.4f}"} for k, v in st["ref_cv"].items()])
        self.table(ref, "table_S10_nonlinear_reference_on_set_c", "Supplementary: Random Forest and Gradient Boosting on the engineered Set C features (nonlinear references; never candidates for the final model).", col_format="lrrrrr")
        # ablation table
        abl = st["ablation"]
        tC = m[("ablation", "C")]
        rows_num, rows_fmt = [], []
        for key, label, role in ABLATION_PLAN:
            v, t = abl[key], m[("ablation", key)]
            rows_fmt.append({"Feature set": v["label"], "p": v["n_features"], "CV MAE": M.pm(v["cv_mae"], v["cv_mae_std"]), "CV RMSE": M.pm(v["cv_rmse"], v["cv_rmse_std"]), "Test MAE": f"{t['MAE']:.2f}",
                             "Test RMSE": f"{t['RMSE']:.2f}", "Test R²": f"{t['R2']:.4f}"})
            rows_num.append({"Variant": v["label"], "role": role, "p": v["n_features"], "nonzero_coefficients": v["nonzero"], "hyperparameters": json.dumps(_py(v["best_params"])), "CV_MAE": v["cv_mae"], "CV_MAE_std": v["cv_mae_std"],
                             "CV_RMSE": v["cv_rmse"], "CV_RMSE_std": v["cv_rmse_std"], "Test_MAE": t["MAE"], "Test_RMSE": t["RMSE"], "Test_R2": t["R2"],
                             "delta_CV_MAE_vs_C_pct": 100 * (v["cv_mae"] / abl["C"]["cv_mae"] - 1), "delta_test_MAE_vs_C_pct": 100 * (t["MAE"] / tC["MAE"] - 1)})
        anum, afmt = pd.DataFrame(rows_num), pd.DataFrame(rows_fmt)
        mask = np.array([r != "sensitivity" for r in anum["role"]])
        bold = R.best_cells(anum, {"CV MAE": ("CV_MAE", "min"), "CV RMSE": ("CV_RMSE", "min"), "Test MAE": ("Test_MAE", "min"), "Test RMSE": ("Test_RMSE", "min"), "Test R²": ("Test_R2", "max")}, row_mask=mask,
                            decimals={"CV MAE": 2, "CV RMSE": 2, "Test MAE": 2, "Test RMSE": 2, "Test R²": 4})
        self.table(afmt, "table_4_feature_ablation", f"Feature ablation with one linear family ({fam}), identical folds and search procedure; only the feature set changes. p = number of design columns entering the model. "
                   "The last row is a sensitivity run, not a candidate. Bold = best value in the column (excluding the sensitivity row).", col_format="p{6.2cm}rrrrrr", csv_df=anum, bold=bold)
        st["table4_num"] = anum
        hv = st["hyp_train"].copy()
        hv["test_delta_mae_pct (descriptive, not used for the verdict)"] = [100 * (m[("ablation", k)]["MAE"] / tC["MAE"] - 1) for k in ("C-pm25", "C-env", "C-temporal")] + [100 * (m[("ablation", "C-rolling")]["MAE"] / m[("ablation", "C-lag")]["MAE"] - 1)]
        self.table(hv, "table_S17_hypothesis_verdicts", "Hypotheses H1-H4: pre-registered verdicts from training CV and training inference; the last column repeats the same comparison on the test period for information only (never used for a verdict).", float_fmt="{:.2f}")
        # DM tests
        fk = st["final_key"]
        comps = [(("baseline", "mean"), "Training mean"), (("baseline", "persistence"), "Persistence"), (("baseline", "ols_single"), "Single-feature OLS"), (("baseline", "rf_raw"), "Random Forest (raw)"),
                 (("baseline", "hgb_raw"), "Gradient Boosting (raw)"), (("reference", "rf_C"), "Random Forest (Set C, reference)"), (("reference", "hgb_C"), "Gradient Boosting (Set C, reference)"),
                 (("ablation", "A"), "Linear, Set A (raw)"), (("ablation", "B"), "Linear, Set B"), (("ablation", "Bp"), "Linear, Set B′"), (("ablation", "C-pm25"), "Linear, C - PM2.5 history")]
        rows = []
        for key, label in comps:
            a1, t1, p1 = S.diebold_mariano(yte, preds[fk], preds[key], "se")
            a2, t2, p2 = S.diebold_mariano(yte, preds[fk], preds[key], "ae")
            rows.append({"final model vs": label, "Δ squared error": a1, "DM p (squared error)": p1, "Δ absolute error": a2, "DM p (absolute error)": p2,
                         "better under RMSE": "final" if a1 < 0 else "comparator", "better under MAE": "final" if a2 < 0 else "comparator"})
        dm = pd.DataFrame(rows)
        self.table(dm, "table_S11_diebold_mariano", f"Diebold–Mariano tests (HAC, 24 lags) on the test period; Δ = loss(final {fam}) − loss(comparator); negative means the final model is better.", float_fmt="{:.3g}", col_format="lrrrrll")
        st["dm"] = dm
        tt = E.threshold_table(yte, {"final model (Set C)": preds[fk], "linear, Set A": preds[("ablation", "A")], "persistence": preds[("baseline", "persistence")], "training mean": preds[("baseline", "mean")]})
        self.table(tt, "table_S12_decision_threshold_sensitivity", f"Share of test predictions with |error| <= tolerance (µg/m³). The proposed tolerance is {C.DECISION_THRESHOLD:g} µg/m³ (operational, not authoritative).", float_fmt="{:.3f}")
        G.fig_threshold(tt)
        share = lambda key: 100 * float((np.abs(yte.to_numpy(float) - preds[key]) <= C.DECISION_THRESHOLD).mean())
        G.set_caption("figure_S07_decision_threshold_sensitivity", f"Share of test predictions whose absolute error is at most τ, for several tolerances τ (µg/m³). At the proposed τ = {C.DECISION_THRESHOLD:g} µg/m³ the final model is within tolerance in {share(fk):.1f} % of the hours "
                      f"(linear Set A {share(('ablation', 'A')):.1f} %, persistence {share(('baseline', 'persistence')):.1f} %, training mean {share(('baseline', 'mean')):.1f} %). "
                      "The tolerance is a configurable, non-authoritative operating choice: the proposed default is marked and the curves show how the comparison depends on it.")
        st["threshold_table"] = tt
        # figures 7 and 8
        t3 = st["table3_num"]
        G.fig_model_comparison(t3, f"{fam}" if fam != "ElasticNet" else "Elastic Net", n_baselines=len(st["baseline_cv"]))
        G.fig_ablation(anum)
        fam_label = "Elastic Net" if fam == "ElasticNet" else fam
        bl = {k: m[("baseline", k)] for k in ("mean", "persistence", "ols_single", "rf_raw", "hgb_raw")}
        lin = [m[("linear", f)] for f in M.FAMILIES]
        fmx, cvf = m[st["final_key"]], st["linear_cv"][fam]
        dm_p = dm.set_index("final model vs").loc["Persistence"]
        G.set_caption("figure_07_model_comparison",
                      f"Baselines and linear models (linear models on Set C): mean 5-fold time-series CV error on the training rows (blue) and one evaluation on the held-out test period (orange); ★ and the grey band mark the selected final model ({fam_label}). "
                      f"Top row: full scale (the training mean, test MAE {bl['mean']['MAE']:.1f}, dominates the axis); bottom row: zoom without the training mean. Test MAE: persistence {bl['persistence']['MAE']:.2f}, single-feature OLS {bl['ols_single']['MAE']:.2f}, "
                      f"Random Forest {bl['rf_raw']['MAE']:.2f}, Gradient Boosting {bl['hgb_raw']['MAE']:.2f}; the four linear families {min(v['MAE'] for v in lin):.2f}-{max(v['MAE'] for v in lin):.2f} (final model {fmx['MAE']:.2f}, "
                      f"{100 * (fmx['MAE'] / bl['persistence']['MAE'] - 1):+.1f} % vs persistence; Diebold-Mariano p = {dm_p['DM p (absolute error)']:.2g}). Test R² is {min(v['R2'] for v in lin):.3f}-{max(v['R2'] for v in lin):.3f} for the linear families and "
                      f"{bl['persistence']['R2']:.3f} for persistence. CV and test errors cover different periods and are not directly comparable; the fold-to-fold standard deviation of the final model's CV MAE is ±{cvf['cv_mae_std']:.2f}.")
        dl = {key: (r.delta_CV_MAE_vs_C_pct, r.delta_test_MAE_vs_C_pct) for (key, _, _), r in zip(ABLATION_PLAN, anum.itertuples())}
        single = ["C-temporal", "C-lag", "C-rolling", "C-interaction", "C-pollutants", "C-meteo", "C-envdiff"]
        G.set_caption("figure_08_feature_ablation",
                      f"Controlled feature ablation with one linear family ({fam_label}); folds, grids and the selection procedure are identical, only the feature set changes. (a) CV MAE (blue, ±1 standard error) and test MAE (orange) of every variant. "
                      f"(b) MAE change relative to Set C in % (> 0 = worse than C); the axis is clipped at +5 %, longer bars are labelled. Removing the PM2.5 history (current value, lags, rolling) raises MAE by {dl['C-pm25'][0]:+.0f} % (CV) / {dl['C-pm25'][1]:+.0f} % (test); "
                      f"removing only lags and rolling terms (keeping PM2.5(t)) {dl['C-history'][0]:+.1f} % / {dl['C-history'][1]:+.1f} %; removing the environment {dl['C-env'][0]:+.1f} % / {dl['C-env'][1]:+.1f} %. Removing temporal, lag, rolling, interaction, pollutant, "
                      f"meteorological or derived-difference features singly changes CV MAE by {min(dl[k][0] for k in single):+.1f} to {max(dl[k][0] for k in single):+.1f} % and test MAE by {min(dl[k][1] for k in single):+.1f} to {max(dl[k][1] for k in single):+.1f} %. "
                      f"Relative to Set C: Set A {dl['A'][0]:+.1f} % / {dl['A'][1]:+.1f} %, Set B {dl['B'][0]:+.1f} % / {dl['B'][1]:+.1f} %, Set B′ {dl['Bp'][0]:+.1f} % / {dl['Bp'][1]:+.1f} %. "
                      f"Sensitivity run (not a candidate): Set A without PM2.5(t) {dl['A-pm25'][0]:+.0f} % / {dl['A-pm25'][1]:+.0f} %.")
        self.log("\nTABLE 4 (feature ablation):")
        self.log(afmt.to_string(index=False))

    # ================================================================== 13. analysis: slices, errors, residuals, coefficients
    def stage_analysis(self):
        self.header("13. ANALYSIS ON THE FINAL TEST PREDICTIONS: SLICES, LARGEST ERRORS, RESIDUALS, COEFFICIENTS")
        st, fam = self.st, self.st["final_family"]
        meta, yte, preds, ref = st["meta_test"], st["yte"].to_numpy(float), st["preds"], st["slice_ref"]
        fk = st["final_key"]
        p_fin = preds[fk]
        # ---- slices
        masks = E.slice_masks(meta, ref)
        models = {"final model (Set C)": p_fin, "linear, Set A (raw)": preds[("ablation", "A")], "persistence": preds[("baseline", "persistence")]}
        stl = E.slice_table(yte, models, masks)
        wide_rows = []
        for (g, s), sub in stl.groupby(["group", "slice"], sort=False):
            d = {r["model"]: r for _, r in sub.iterrows()}
            f_, a_, p_ = d["final model (Set C)"], d["linear, Set A (raw)"], d["persistence"]
            wide_rows.append({"Slice group": g, "Slice": s, "n": int(f_["n"]), "Final MAE": f_["MAE"], "Final RMSE": f_["RMSE"], "Set A MAE": a_["MAE"], "Set A RMSE": a_["RMSE"], "Persistence MAE": p_["MAE"], "Persistence RMSE": p_["RMSE"],
                              "Δ MAE vs Set A": f_["MAE"] - a_["MAE"], "Δ MAE vs persistence": f_["MAE"] - p_["MAE"], "bias (final)": f_["bias"]})
        wide = pd.DataFrame(wide_rows)
        self.table(wide, "table_7_slice_level_performance", "Test-set MAE and RMSE (µg/m³) of the final model, the same linear family on the raw features (Set A) and persistence, by slice. Slice cut-offs come from the training data or the calendar. "
                   "Δ < 0 means the final model is better.", float_fmt="{:.2f}", col_format="llrrrrrrrrrr")
        G.fig_slices(stl, list(models))
        hi3, lo1 = wide.sort_values("Final MAE", ascending=False).head(3), wide.sort_values("Final MAE").iloc[0]
        beat_p, beat_a = wide["Δ MAE vs persistence"] < 0, wide["Δ MAE vs Set A"] < 0
        G.set_caption("figure_12_slice_performance",
                      f"Test-set MAE (left) and RMSE (right) of the final model (Set C), the same linear family on the raw features (Set A) and persistence, by slice ({len(wide)} slices in {wide['Slice group'].nunique()} groups; slices of different groups overlap). "
                      "Cut-offs come from the training data or the calendar. The PM2.5-level and one-hour-change slices are defined by the realised outcome: this favours persistence in stable hours and penalises every model in rapid-change hours, so they show where errors occur, not a condition known in advance. "
                      f"Largest final-model MAE: {'; '.join(f'{r.Slice} ({r._4:.1f})' for r in hi3.itertuples())}; smallest: {lo1['Slice']} ({lo1['Final MAE']:.1f}). The final model has a lower MAE than persistence in {int(beat_p.sum())} of {len(wide)} slices and than Set A in {int(beat_a.sum())}; "
                      f"it is worse than persistence in: {', '.join(wide.loc[~beat_p, 'Slice']) or 'none'}.")
        self.log("SLICE-LEVEL PERFORMANCE (final model | Set A | persistence, MAE):")
        for r in wide.itertuples():
            self.log(f"  {r[1]:28s} {r[2]:28s} n={r[3]:5d}  final {r[4]:6.2f} (RMSE {r[5]:6.2f}) | A {r[6]:6.2f} | persistence {r[8]:6.2f} | d_vs_A {r[10]:+6.2f} | d_vs_pers {r[11]:+6.2f}")
        helps = wide.sort_values("Δ MAE vs persistence").head(3)
        hurts = wide.sort_values("Δ MAE vs persistence").tail(3)
        self.log("  engineered features / model help most vs persistence in: " + "; ".join(f"{r.Slice} ({r._11:+.2f})" for r in helps.itertuples()))
        self.log("  ... and help least / hurt in: " + "; ".join(f"{r.Slice} ({r._11:+.2f})" for r in hurts.itertuples()))
        # ---- largest errors
        e = yte - p_fin
        top = E.top_errors(meta, yte, p_fin, {**ref, "pm10_equal_share": st.get("pm10_equal_share")})
        pat = E.shared_pattern(top, e, meta, ref)
        tab6 = top.assign(Time=top["time"].dt.strftime("%Y-%m-%d %H:%M"), Wind=[f"{w:.1f} m/s {d}" for w, d in zip(top["WSPM"], top["wd"])])[
            ["case", "Time", "actual", "predicted", "error", "abs_error", "hour", "month", "PM10", "NO2", "CO", "TEMP", "Wind", "possible_explanation"]]
        self.table(tab6.rename(columns={"actual": "Actual PM2.5", "predicted": "Predicted PM2.5", "error": "Error", "abs_error": "Abs. error", "hour": "Hour", "month": "Month", "TEMP": "Temp (°C)", "possible_explanation": "Possible explanation"}),
                   "table_6_three_largest_test_errors", "The three largest absolute errors of the final model on the test period (error = actual − predicted, µg/m³; time = target hour). Explanations are rule-based hypotheses, not established causes.",
                   float_fmt="{:.1f}", col_format="rlrrrrrrrrrrlp{5.2cm}", csv_df=top.drop(columns=["wd_prev"]))
        G.fig_actual_vs_predicted(meta, yte, p_fin, top.iloc[0])
        fm_ = st["metrics"][fk]
        cur_te = meta[C.CURRENT].to_numpy(float)
        chg_te = yte - cur_te
        sst_chg = float(((chg_te - chg_te.mean()) ** 2).sum())
        r2_chg = {nm: 1 - float(((chg_te - (q - cur_te)) ** 2).sum()) / sst_chg for nm, q in (("final", p_fin), ("Set A", preds[("ablation", "A")]), ("persistence", cur_te))}
        daily_te = pd.DataFrame({"t": pd.DatetimeIndex(meta["target_time"]).floor("D"), "a": yte, "p": p_fin}).groupby("t").mean()
        outside = float(((yte > 500) | (p_fin > 500) | (p_fin < -20)).mean())
        t1 = top.iloc[0]
        G.set_caption("figure_09_actual_vs_predicted",
                      f"Final model ({fam}, Set C) on the held-out test period ({len(yte):,} hours; MAE {fm_['MAE']:.2f}, R² {fm_['R2']:.3f}). (a) Actual vs predicted hourly PM2.5 (axes clipped to 0-500 µg/m³; {100 * outside:.1f} % of the hours lie outside). "
                      f"(b) Daily means over the {len(daily_te)} test days (correlation {np.corrcoef(daily_te['a'], daily_te['p'])[0, 1]:.3f}). (c) Ten-day window around the largest absolute error (orange dot; target hour {t1['time']:%Y-%m-%d %H:%M}, actual {t1['actual']:.0f}, predicted {t1['predicted']:.0f} µg/m³). "
                      f"The forecast follows the level closely but stays near the last observed value: it explains {100 * r2_chg['final']:.0f} % of the variance of the one-hour change PM2.5(t+1) - PM2.5(t) (linear Set A {100 * r2_chg['Set A']:.0f} %, persistence 0 % by construction), "
                      "so abrupt rises and falls are reproduced about one hour late.")
        self.log(f"\none-hour change PM2.5(t+1) - PM2.5(t) in the test period: share of its variance explained - final model {100 * r2_chg['final']:.1f} %, linear Set A {100 * r2_chg['Set A']:.1f} %, persistence {100 * r2_chg['persistence']:.1f} % (<= 0 by construction)")
        self.st["r2_change"] = r2_chg
        self.log("\nLARGEST ERRORS (error = actual - predicted):")
        for r in top.itertuples():
            self.log(f"  #{r.case}: target hour {r.time} (issued {r.forecast_issued}) actual {r.actual:.0f} predicted {r.predicted:.0f} error {r.error:+.0f} | PM2.5(t) {r.PM25_t:.0f}, change {r.change_actual_minus_current:+.0f} | "
                     f"PM10 {r.PM10:.0f} NO2 {r.NO2:.0f} CO {r.CO:.0f} TEMP {r.TEMP:.1f} wind {r.WSPM:.1f} {r.wd} | {r.possible_explanation}")
        self.log(f"  SHARED PATTERN: {pat['under_predictions']}/{pat['n']} are under-predictions; {pat['rapid_rises']}/{pat['n']} follow a rapid one-hour rise (>= training P90 of |change|); {pat['rapid_falls']} a rapid fall; "
                 f"|change| of the {pat['n']} cases = {[round(v) for v in pat['top_abs_changes']]} = {[round(v, 1) for v in pat['top_change_multiple_of_p95']]} x the training P95 of that direction; forecast minus PM2.5(t) = {[round(v) for v in pat['top_forecast_minus_current']]}; "
                 f"among the largest 1 % of errors ({pat['top1pct_n']} hours): {100 * pat['top1pct_under_share']:.0f} % under-predictions, {100 * pat['top1pct_rapid_change_share']:.0f} % in rapid-change hours; "
                 f"they carry {100 * pat['top1pct_share_sq_error']:.0f} % of the squared error")
        # ---- residual diagnostics
        diag = E.residual_diagnostics(yte, p_fin, ref)
        a_res = acf(e, nlags=168, fft=True)
        G.fig_residuals(meta, yte, p_fin, a_res)
        mo_ = pd.DataFrame({"m": pd.DatetimeIndex(meta["target_time"]).to_period("M"), "e": e, "ae": np.abs(e)}).groupby("m").agg(bias=("e", "mean"), mae=("ae", "mean"))
        G.set_caption("figure_10_residual_analysis",
                      f"Residual diagnostics of the final model on the test period (residual = actual - predicted). (a) The spread grows with the predicted level: the residual standard deviation in the highest prediction decile is {diag['std_ratio_top_vs_bottom_decile']:.1f} times that in the lowest "
                      f"(Breusch-Pagan p {_pe(diag['breusch_pagan_p'])}); orange: mean residual in 20 prediction bins. (b) Monthly mean residuals range from {mo_['bias'].min():+.1f} to {mo_['bias'].max():+.1f} µg/m³ (overall {diag['mean_error']:+.2f}; {100 * diag['share_under_predicted']:.0f} % of the hours under-predicted), "
                      f"but the swings grow in the pollution season: monthly MAE {mo_['mae'].min():.1f} ({mo_['mae'].idxmin().strftime('%b %Y')}) to {mo_['mae'].max():.1f} ({mo_['mae'].idxmax().strftime('%b %Y')}). (c) Residual distribution on a log density axis: skewness {diag['skew']:.2f}, excess kurtosis {diag['excess_kurtosis']:.0f}, "
                      f"far from normal (Jarque-Bera p {_pe(diag['jarque_bera_p'])}); this describes the errors and is neither evidence for nor against the model. (d) Actual vs predicted. (e) Absolute error over time. (f) Residual autocorrelation "
                      f"(lag 1: {diag['acf'][1]:+.2f}, lag 24: {diag['acf'][24]:+.2f}; Durbin-Watson {diag['durbin_watson']:.2f}; Ljung-Box p {_pe(diag['ljung_box_p'][24])} at lag 24; band: ±1.96/√n), so classical OLS standard errors would be too small - the inference tables use HAC errors and a block bootstrap.")
        dd = pd.DataFrame([{"diagnostic": "Breusch-Pagan LM (residual^2 on prediction)", "value": diag["breusch_pagan_lm"]}, {"diagnostic": "Breusch-Pagan p", "value": diag["breusch_pagan_p"]},
                           {"diagnostic": "Spearman(|resid|, prediction)", "value": diag["spearman_abs_resid_vs_pred"]}, {"diagnostic": "resid std top / bottom prediction decile", "value": diag["std_ratio_top_vs_bottom_decile"]},
                           {"diagnostic": "Durbin-Watson", "value": diag["durbin_watson"]}, *[{"diagnostic": f"residual ACF lag {k}", "value": v} for k, v in diag["acf"].items()],
                           *[{"diagnostic": f"Ljung-Box p, lag {k}", "value": v} for k, v in diag["ljung_box_p"].items()], {"diagnostic": "mean error (bias)", "value": diag["mean_error"]},
                           {"diagnostic": "share under-predicted", "value": diag["share_under_predicted"]}, {"diagnostic": "residual skewness", "value": diag["skew"]}, {"diagnostic": "residual excess kurtosis", "value": diag["excess_kurtosis"]},
                           {"diagnostic": "mean error when actual >= training P95", "value": diag["extreme_mean_error"]}, {"diagnostic": "MAE when actual >= training P95", "value": diag["extreme_mae"]},
                           {"diagnostic": "mean error when actual <= training P25", "value": diag["low_mean_error"]}, {"diagnostic": "MAE when actual <= training P25", "value": diag["low_mae"]}])
        self.table(dd, "table_S13_residual_diagnostics", "Residual diagnostics of the final model on the test period (error = actual − predicted).", float_fmt="{:.4g}")
        diag["spread_by_pred_decile"].to_csv(C.TABLES / "table_S14_residual_spread_by_prediction_decile.csv", index=False)
        self.log("\nRESIDUAL ANALYSIS (test):")
        self.log(f"  heteroscedasticity: Breusch-Pagan p {_pe(diag['breusch_pagan_p'])}; residual std in the highest vs lowest prediction decile = x{diag['std_ratio_top_vs_bottom_decile']:.1f}; Spearman(|resid|, prediction) = {diag['spearman_abs_resid_vs_pred']:+.2f}")
        self.log(f"  autocorrelation: Durbin-Watson {diag['durbin_watson']:.2f}; ACF lag1 {diag['acf'][1]:+.3f}, lag24 {diag['acf'][24]:+.3f}, lag168 {diag['acf'][168]:+.3f}; Ljung-Box p (1/24/168) = {[_fp(v) for v in diag['ljung_box_p'].values()]}")
        self.log(f"  bias: mean error {diag['mean_error']:+.2f}; {100 * diag['share_under_predicted']:.0f} % of hours under-predicted; extreme pollution (actual >= P95={ref['y_q95']:.0f}, n={diag['extreme_n']}): "
                 f"mean error {diag['extreme_mean_error']:+.1f}, MAE {diag['extreme_mae']:.1f}; clean air (actual <= P25={ref['y_q25']:.0f}): mean error {diag['low_mean_error']:+.1f}, MAE {diag['low_mae']:.1f}")
        self.log(f"  distribution: skew {diag['skew']:.2f}, excess kurtosis {diag['excess_kurtosis']:.1f}, Jarque-Bera p {_pe(diag['jarque_bera_p'])} -> residuals are heavy-tailed and NOT normal (not evidence of a good model either way)")
        # ---- coefficients table + figure (training analysis; fixed before the freeze)
        tab = st["coef_tab"]
        t5 = pd.DataFrame({"Feature": tab["feature"], "Group": tab["group"], "Coefficient (final model)": tab["coef"].map(lambda v: f"{v:.3f}"),
                           "95% CI (block bootstrap)": [f"[{a:.3f}, {b:.3f}]" for a, b in zip(tab["boot_lo"], tab["boot_hi"])], "OLS coefficient": tab["ols"].map(lambda v: f"{v:.3f}"),
                           "95% CI (OLS, HAC)": [f"[{a:.3f}, {b:.3f}]" for a, b in zip(tab["hac_lo"], tab["hac_hi"])], "p-value (HAC)": tab["p_hac"].map(_fp),
                           "p (Holm)": tab["p_holm"].map(_fp), "VIF": tab["vif"].map(lambda v: f"{v:.1f}")})
        self.table(t5, "table_5_final_model_coefficients", f"Final model ({fam}, Set C): coefficients in µg/m³ per one standard deviation of the (centred) feature with 95% intervals (moving-block bootstrap for the penalised predictive model; "
                   "HAC for the OLS statistical companion fitted on the same standardised design), HAC and Holm-adjusted p-values and VIF. Training rows only. Predictive and inferential models are distinct.",
                   col_format="llrcrcrrr", csv_df=tab.drop(columns=["abs_coef"]))
        G.fig_coefficients(tab, "Elastic Net" if fam == "ElasticNet" else fam)
        n_nz, n_ci, n_holm = int((tab["coef"].abs() > 1e-8).sum()), int(((tab["boot_lo"] > 0) | (tab["boot_hi"] < 0)).sum()), int((tab["p_holm"] < 0.05).sum())
        big = tab.reindex(tab["coef"].abs().sort_values(ascending=False).index).head(3)
        cs = tab.set_index("feature")
        collinear = ""
        if C.CURRENT in cs.index and "PM25_lag_1" in cs.index and np.sign(cs.loc[C.CURRENT, "coef"]) != np.sign(cs.loc["PM25_lag_1", "coef"]):
            collinear = (f" The current PM2.5 and its first lag are strongly collinear (VIF {cs.loc[C.CURRENT, 'vif']:.0f} and {cs.loc['PM25_lag_1', 'vif']:.0f}) and have opposite signs, so they must be read together: the pair acts as the current level plus a fraction of the last one-hour change (momentum), not as two independent effects.")
        G.set_caption("figure_11_coefficients_confidence_intervals",
                      f"Coefficients of the final model ({'Elastic Net' if fam == 'ElasticNet' else fam}, Set C) in µg/m³ of next-hour PM2.5 per one standard deviation of the (centred) feature, with 95 % intervals. Blue: penalised predictive model with moving-block-bootstrap intervals (one-week blocks); "
                      f"orange: OLS companion on the same standardised design with HAC (Newey-West) intervals. {n_nz} of the {len(tab)} coefficients are non-zero; {n_ci} bootstrap intervals exclude zero and {n_holm} OLS-HAC p-values stay below 0.05 after Holm correction. "
                      f"Largest effects: {', '.join(f'{r.feature} ({r.coef:+.1f})' for r in big.itertuples())}. The three panels have different x-scales." + collinear)
        # ---- coefficient signs by feature group, read against the hypotheses (training coefficients; descriptive, not a test)
        tab["nonzero"] = tab["coef"].abs() > 1e-8
        grows = []
        for g, sub in tab.groupby("group", sort=False):
            nz = sub[sub["nonzero"]]
            top3 = nz.reindex(nz["coef"].abs().sort_values(ascending=False).index).head(3)
            grows.append({"group": g, "n_features": len(sub), "non_zero": int(sub["nonzero"].sum()), "sum_abs_coef": float(sub["coef"].abs().sum()),
                          "ci_excludes_zero": int(((sub["boot_lo"] > 0) | (sub["boot_hi"] < 0)).sum()), "largest_effects": "; ".join(f"{r.feature} {r.coef:+.2f}" for r in top3.itertuples())})
        gtab = pd.DataFrame(grows)
        self.table(gtab, "table_S16_coefficients_by_group", "Final model: coefficients per feature group (µg/m³ per SD): number of non-zero coefficients, sum of absolute coefficients, bootstrap intervals excluding zero and the largest effects.", float_fmt="{:.2f}")
        co = tab.set_index("feature")["coef"]
        fmt_ = lambda names: ", ".join(f"{n} {co[n]:+.2f}" for n in names if n in co.index and abs(co[n]) > 1e-8) or "none non-zero"
        nzc = lambda names: f"{int(sum(abs(co[n]) > 1e-8 for n in names if n in co.index))}/{sum(n in co.index for n in names)}"
        wl_ = {r["coefficient group"].strip(): r["p-value (HAC)"] for _, r in st["wald"].iterrows()}
        self.log("\nCOEFFICIENT SIGNS READ AGAINST THE HYPOTHESES (final model, training coefficients, ug/m3 per SD; descriptive - the verdicts come from the pre-registered CV rule):")
        self.log(f"  H1 recent PM2.5: current {fmt_([C.CURRENT])} | lags {fmt_(F.LAG_COLS)} [{nzc(F.LAG_COLS)} non-zero] | rolling {fmt_(F.ROLL_COLS)} [{nzc(F.ROLL_COLS)} non-zero]")
        self.log(f"  H2 environment: pollutants {fmt_(F.POLLUTANTS)} | meteorology {fmt_(F.METEO_NUMERIC)} | derived {fmt_(F.DERIVED_CANDIDATES)} | interactions {fmt_(F.interaction_names())}")
        self.log(f"  H3 calendar: {fmt_(F.TEMPORAL_COLS)} | joint Wald p (HAC): daily {_fp(wl_['daily terms (hour_sin, hour_cos)'])}, weekly {_fp(wl_['weekly terms (dow_sin, dow_cos, is_weekend)'])}, monthly {_fp(wl_['monthly terms (month_sin, month_cos)'])}")
        self.log(f"  H4 rolling vs lag: {nzc(F.ROLL_COLS)} rolling and {nzc(F.LAG_COLS)} lag coefficients are non-zero; removing all rolling features changes CV MAE by {100 * (st['ablation']['C-rolling']['cv_mae'] / st['ablation']['C']['cv_mae'] - 1):+.2f} %, removing all lags by {100 * (st['ablation']['C-lag']['cv_mae'] / st['ablation']['C']['cv_mae'] - 1):+.2f} %")
        # redundancy figure of the final design
        G.fig_redundancy(st["design_C"], st["vif_c"])
        hv = st["vif_c"].replace(np.inf, np.nan).sort_values(ascending=False)
        n10 = int((hv > C.PROP_VIF).sum())
        G.set_caption("figure_S05_feature_redundancy",
                      f"Redundancy of the Set C training design ({len(hv)} columns). (a) Pearson correlation matrix ordered by hierarchical clustering: the PM2.5 history (current value, lags, rolling means) forms one tightly correlated block that is also related to PM10, NO2 and CO. "
                      f"(b) The 22 columns with the highest variance-inflation factor (log axis): {n10} columns exceed VIF = {C.PROP_VIF:g} (maximum {hv.iloc[0]:.0f} for {hv.index[0]}); "
                      f"the {min(n10, 10)} largest are {', '.join(hv.index[:min(n10, 10)])}. Such collinearity makes individual history coefficients unstable and motivates penalised regression.")
        self.st.update(slice_wide=wide, top_errors=top, pattern=pat, diag=diag)

    # ================================================================== 14. tables + summary
    def stage_report(self):
        self.header("14. DATASET TABLE, FEATURE-GROUP TABLE AND FINAL SUMMARY")
        st, fam = self.st, self.st["final_family"]
        tr_y, te_y = st["y_train"], st["yte"]
        rep = st["raw_report"]
        info = st["split"]
        rows = [("Source", f"UCI 'Beijing Multi-Site Air-Quality Data' (id 501), DOI {C.SOURCE_INFO['doi']}; station {C.STATION}"),
                ("Rows x columns (file)", f"{rep['rows']:,} x {rep['columns']} (hourly, {st['timeline']['start'][:10]} to {st['timeline']['end'][:10]}; {st['timeline']['span_days']} days)"),
                ("Sampling", "hourly, regular (0 missing / 0 duplicated timestamps)"),
                ("Dropped identifier columns", "No (row counter), station (constant)"),
                ("Missing values (whole file)", ", ".join(f"{r.column} {r.missing}" for r in rep["dtypes"].itertuples() if r.missing)),
                ("Task", "predict PM2.5 at t+1 (µg/m³) from information available at hour t"),
                ("Train / test split (rows)", f"{info['n_train']:,} / {info['n_test']:,} hours ({info['train_start'][:10]} to {info['train_end'][:10]} | {info['test_start'][:10]} to {info['test_end'][:10]})"),
                ("Usable rows (train / test)", f"{len(st['X_train']):,} / {len(te_y):,} (missing target, missing current PM2.5 or unusable history excluded)"),
                ("Candidate features / Set A / Set C (design columns)", f"{st['X_train'].shape[1]} / {st['design_A'].shape[1]} / {len(st['design_names'])}")]
        t1 = pd.DataFrame(rows, columns=["item", "value"])
        self.table(t1, "table_1_dataset_statistics", "Dataset statistics.", col_format="lp{11cm}")
        ts = lambda s, lab: {"split": lab, "n": len(s), "mean": s.mean(), "std": s.std(), "min": s.min(), "q25": s.quantile(.25), "median": s.median(), "q75": s.quantile(.75), "max": s.max(), "skew": stats.skew(s)}
        t1b = pd.DataFrame([ts(tr_y, "train (usable rows)"), ts(te_y, "test (usable rows)")])
        self.table(t1b, "table_1b_target_statistics", "Statistics of the target PM2.5 at t+1 (µg/m³) in the usable training and test rows.", float_fmt="{:.1f}", col_format="lrrrrrrrrr")
        X = st["X_train"]
        vs = pd.DataFrame({"variable": [F.CURRENT] + F.POLLUTANTS + F.METEO_NUMERIC})
        vs["mean"], vs["std"], vs["min"], vs["max"] = [X[c].mean() for c in vs["variable"]], [X[c].std() for c in vs["variable"]], [X[c].min() for c in vs["variable"]], [X[c].max() for c in vs["variable"]]
        self.table(vs, "table_S15_variable_summary_train", "Summary of the raw numeric predictors on the usable training rows.", float_fmt="{:.2f}")
        txt = self.summary_text()
        (C.RESULTS / "final_summary.txt").write_text(txt)
        self.log("\n" + txt)
        meta = {"seed": C.SEED, "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "runtime_minutes": round((time.time() - self.t0) / 60, 1), "design_sha256": st["design_sha"],
                "libraries": st["frozen"]["libraries"], "n_jobs": C.N_JOBS}
        (C.LOGS / "run_metadata.json").write_text(json.dumps(_py(meta), indent=2))
        (C.FIGURES / "figure_captions.md").write_text("\n\n".join(f"**{k}** - {v}" for k, v in sorted(G.FIG_CAPTIONS.items())) + "\n")
        (C.TABLES / "table_captions.md").write_text("\n\n".join(f"**{k}** - {v}" for k, v in sorted(R.TABLE_CAPTIONS.items())) + "\n")

    def summary_text(self):
        st, fam = self.st, self.st["final_family"]
        fm, cvf = st["metrics"][st["final_key"]], st["linear_cv"][fam]
        info, rep, tl = st["split"], st["raw_report"], st["timeline"]
        hyp = st["hyp_train"].set_index("hypothesis")
        abl, m = st["ablation"], st["metrics"]
        top = st["coef_tab"].head(8)
        pat, te = st["pattern"], st["top_errors"]
        diag = st["diag"]
        line = "=" * 48
        L = [line, "DATASET", line, f"Rows: {rep['rows']:,} (file) -> {info['n_total']:,} after target construction", f"Features: {len(st['design_names'])} design columns in the final Set C (from {st['X_train'].shape[1]} candidate columns); Set A: {st['design_A'].shape[1]}",
             f"Time range: {tl['start'][:16]} to {tl['end'][:16]}", f"Sampling interval: 1 hour (regular, 0 missing timestamps)", f"Station: {C.STATION}", "Target: PM2.5 concentration at t+1 hour", f"Target unit: {C.UNIT}", "",
             line, "TASK", line, "Prediction: X(t) -> PM2.5(t+1)", "Allowed at hour t: current pollutants and meteorology, calendar of t, PM2.5 up to t (current value, lags, rolling statistics of the shifted series).",
             "Forbidden: PM2.5(t+1) or any later value, future meteorology, centred rolling windows, any statistic fitted on test data.", "", line, "SPLIT", line,
             f"Train: {info['n_train']:,} hours {info['train_start'][:16]} -> {info['train_end'][:16]} ({len(st['X_train']):,} usable)", f"Test:  {info['n_test']:,} hours {info['test_start'][:16]} -> {info['test_end'][:16]} ({len(st['yte']):,} usable)", "",
             line, "HYPOTHESES (pre-registered; verdict from training CV + training inference)", line]
        names = {"H1": "Recent PM2.5 predicts next-hour PM2.5", "H2": "Environment adds information beyond PM2.5 history", "H3": "Hourly/weekly temporal patterns captured by temporal features", "H4": "Rolling statistics more stable than individual lags"}
        tdel = lambda k: 100 * (m[("ablation", k)]["MAE"] / m[("ablation", "C")]["MAE"] - 1)
        tnote = {"H1": f"descriptive test check (not used for the verdict): removing the history changes test MAE by {tdel('C-pm25'):+.1f} %",
                 "H2": f"descriptive test check: removing the environment changes test MAE by {tdel('C-env'):+.2f} % - the small training-CV gain did not replicate on the test period",
                 "H3": f"descriptive test check: removing the calendar terms changes test MAE by {tdel('C-temporal'):+.2f} %",
                 "H4": f"descriptive test check: test MAE lag-only {m[('ablation', 'C-rolling')]['MAE']:.3f} vs rolling-only {m[('ablation', 'C-lag')]['MAE']:.3f}; fold-to-fold std of CV-MAE rolling-only {st['h4_stability']['fold_std_mae_rolling_only']:.2f} vs lag-only {st['h4_stability']['fold_std_mae_lag_only']:.2f}"}
        for h in ("H1", "H2", "H3", "H4"):
            r = hyp.loc[h]
            L.append(f"{h}: {r['verdict']} - {names[h]} ({r['comparison']}: dMAE {r['delta_cv_mae_pct']:+.2f} %, worse in {r['folds_worse']}/5 folds; {r['inferential_test']}); {tnote[h]}")
        L += ["", line, "BEST MODEL", line, f"Model: {fam} regression (linear)", f"Hyperparameters: {json.dumps(_py(cvf['best_params']))}", "Feature set: C (engineered)",
              f"Why: {st['selection']['why']}", "", line, "RESULTS", line, f"CV MAE : {M.pm(cvf['cv_mae'], cvf['cv_mae_std'])} ug/m3", f"CV RMSE: {M.pm(cvf['cv_rmse'], cvf['cv_rmse_std'])} ug/m3", f"CV R2  : {M.pm(cvf['cv_r2'], cvf['cv_r2_std'], 4)}", "",
              f"Test MAE : {fm['MAE']:.3f} ug/m3", f"Test RMSE: {fm['RMSE']:.3f} ug/m3", f"Test R2  : {fm['R2']:.4f}",
              f"Share of test predictions within +-{C.DECISION_THRESHOLD:g} ug/m3 (DECISION_THRESHOLD, PROPOSED / needs your confirmation): {100 * fm['Within_tau']:.1f} % (persistence: {100 * m[('baseline', 'persistence')]['Within_tau']:.1f} %)",
              f"Reference (test MAE / RMSE / R2): training mean {m[('baseline', 'mean')]['MAE']:.2f}/{m[('baseline', 'mean')]['RMSE']:.2f}/{m[('baseline', 'mean')]['R2']:.3f}; persistence {m[('baseline', 'persistence')]['MAE']:.2f}/{m[('baseline', 'persistence')]['RMSE']:.2f}/{m[('baseline', 'persistence')]['R2']:.4f}; "
              f"single-feature OLS {m[('baseline', 'ols_single')]['MAE']:.2f}/{m[('baseline', 'ols_single')]['RMSE']:.2f}/{m[('baseline', 'ols_single')]['R2']:.4f}; Random Forest {m[('baseline', 'rf_raw')]['MAE']:.2f}/{m[('baseline', 'rf_raw')]['RMSE']:.2f}/{m[('baseline', 'rf_raw')]['R2']:.4f}; "
              f"Gradient Boosting {m[('baseline', 'hgb_raw')]['MAE']:.2f}/{m[('baseline', 'hgb_raw')]['RMSE']:.2f}/{m[('baseline', 'hgb_raw')]['R2']:.4f}",
              "Supplementary nonlinear references on the engineered Set C (never candidates for the final model; test MAE / RMSE / R2): " + "; ".join(
                  f"{lab} {m[('reference', k)]['MAE']:.2f}/{m[('reference', k)]['RMSE']:.2f}/{m[('reference', k)]['R2']:.4f} ({100 * (m[('reference', k)]['MAE'] / fm['MAE'] - 1):+.1f} % test MAE vs the final model)"
                  for k, lab in (("rf_C", "Random Forest"), ("hgb_C", "Gradient Boosting"))), "",
              line, "ABLATION (same family, folds, search; CV MAE | test MAE / RMSE / R2)", line]
        for k in ("A", "B", "Bp", "C"):
            L.append(f"Set {k.replace('Bp', 'B′')}: CV MAE {abl[k]['cv_mae']:.3f} | test {m[('ablation', k)]['MAE']:.3f} / {m[('ablation', k)]['RMSE']:.3f} / {m[('ablation', k)]['R2']:.4f}  ({abl[k]['label']}; p={abl[k]['n_features']}; {json.dumps(_py(abl[k]['best_params']))})")
        L.append("Group removal (dMAE vs C, CV / test): " + "; ".join(f"{abl[k]['label']}: {100 * (abl[k]['cv_mae'] / abl['C']['cv_mae'] - 1):+.2f} % / {100 * (m[('ablation', k)]['MAE'] / m[('ablation', 'C')]['MAE'] - 1):+.2f} %"
                                                                      for k in ("C-temporal", "C-lag", "C-rolling", "C-interaction", "C-history", "C-pm25", "C-env", "C-pollutants", "C-meteo", "C-envdiff")))
        L += ["", line, "TOP FEATURES (final model, ug/m3 per 1 SD; 95% block-bootstrap CI; OLS-companion HAC p-value, unadjusted and Holm-adjusted over all coefficients; VIF)", line]
        L += [f"{r.feature}: {r.coef:+.3f} [{r.boot_lo:+.3f}, {r.boot_hi:+.3f}] p(HAC) {_pe(r.p_hac)}, p(Holm) {_pe(r.p_holm)}, VIF {r.vif:.1f}" for r in top.itertuples()]
        L += ["", line, "ERROR ANALYSIS", line]
        for r in te.itertuples():
            L.append(f"Largest error #{r.case}: {r.time:%Y-%m-%d %H:%M} actual {r.actual:.0f}, predicted {r.predicted:.0f}, error {r.error:+.0f} - {r.possible_explanation}")
        if pat["all_top_above_p95"]:
            L.append(f"Shared failure pattern: all {pat['n']} largest errors are abrupt one-hour changes (|change| {min(pat['top_abs_changes']):.0f}-{max(pat['top_abs_changes']):.0f} ug/m3, {min(pat['top_change_multiple_of_p95']):.1f}-{max(pat['top_change_multiple_of_p95']):.1f}x the training 95th percentile "
                     f"of changes in that direction); in each, the forecast stays within {min(abs(v) for v in pat['top_forecast_minus_current']):.0f}-{max(abs(v) for v in pat['top_forecast_minus_current']):.0f} ug/m3 of the last observed PM2.5(t), i.e. it behaves like persistence; "
                     f"{pat['under_predictions']}/{pat['n']} are under-predictions (rises) and {pat['over_predictions']}/{pat['n']} over-predictions (falls).")
        else:
            L.append(f"No single shared pattern among the {pat['n']} largest errors: {pat['under_predictions']} under-predictions, {pat['over_predictions']} over-predictions, {pat['rapid_rises']} rapid rises and {pat['rapid_falls']} rapid falls (>= training P90 of |change|).")
        L.append(f"Among the largest 1 % of absolute errors ({pat['top1pct_n']} hours) {100 * pat['top1pct_rapid_change_share']:.0f} % occur in rapid-change hours (|change| >= training P90) and {100 * pat['top1pct_under_share']:.0f} % are under-predictions; "
                 f"these hours carry {100 * pat['top1pct_share_sq_error']:.0f} % of the squared error. The final model explains {100 * st['r2_change']['final']:.0f} % of the variance of the one-hour change PM2.5(t+1) - PM2.5(t) (linear Set A {100 * st['r2_change']['Set A']:.0f} %, persistence 0 % by construction).")
        L += ["", line, "LIMITATIONS", line]
        L += [f"1. One chronological split and one test window ({len(st['yte']):,} hours, {st['split']['test_start'][:10]} to {st['split']['test_end'][:10]}): no variability estimate across other periods.",
              f"2. One-step-ahead with the observed current PM2.5: the forecast is dominated by persistence (lag-1 autocorrelation {st['acf'][1]:.2f}; the final model explains only {100 * st['r2_change']['final']:.0f} % of the variance of the one-hour change); longer horizons were not evaluated.",
              "3. Rows with a missing target, a missing current PM2.5 or an unusable history are excluded (policy documented); results apply to hours with this information available.",
              "4. Heavy-tailed target: squared-error linear models under-react to abrupt episodes; target transforms / quantile models are outside the allowed family.",
              f"5. Residuals are heteroscedastic, heavy-tailed and autocorrelated (Durbin-Watson {diag['durbin_watson']:.2f}, Ljung-Box p {_pe(diag['ljung_box_p'][24])} at lag 24): inference uses HAC / block bootstrap but remains descriptive and post-selection.",
              "6. CV scores of the selected configurations are not nested; thresholds of the hypothesis and selection rules are conventions fixed in advance.",
              f"7. DECISION_THRESHOLD = {C.DECISION_THRESHOLD:g} ug/m3 is a PROPOSED operational tolerance, not an authoritative standard - please confirm or override it in src/config.py.",
              "8. A single station (Aotizhongxin); regional transport and boundary-layer variables are not observed. The UCI snippet fetch_ucirepo(id=381) refers to a different dataset; the experiment uses UCI id 501."]
        better = [(lab, m[("reference", k)]["MAE"]) for k, lab in (("rf_C", "Random Forest"), ("hgb_C", "Gradient Boosting")) if m[("reference", k)]["MAE"] < fm["MAE"]]
        if better:
            L.append("9. The linear restriction costs accuracy here: " + " and ".join(f"{lab} on the same engineered features reaches a test MAE of {v:.2f} ({100 * (v / fm['MAE'] - 1):+.1f} % vs the final model)" for lab, v in better)
                     + f" (reference only - nonlinear models are not allowed as the final model; training-CV MAE: Random Forest {st['ref_cv']['rf_C']['cv_mae']:.2f}, Gradient Boosting {st['ref_cv']['hgb_C']['cv_mae']:.2f}, final model {cvf['cv_mae']:.2f}).")
        return "\n".join(L)

    # ================================================================== run everything
    def run_all(self, until=None):
        self.stage_inspect()
        self.stage_split()
        self.stage_training_data()
        self.stage_eda()
        self.stage_preregister()
        self.stage_features()
        self.stage_baselines()
        self.stage_linear_models()
        self.stage_ablation()
        self.stage_statistics()
        self.stage_freeze()
        if until == "freeze":
            self.log("\nstopped before the test evaluation (--until freeze).")
            return self
        self.stage_final_test()
        self.stage_analysis()
        self.stage_report()
        self.log(f"\nDONE in {(time.time() - self.t0) / 60:.1f} min. Artefacts: {C.RESULTS}")
        return self


def sm_dw(res):
    from statsmodels.stats.stattools import durbin_watson
    return float(durbin_watson(res))
