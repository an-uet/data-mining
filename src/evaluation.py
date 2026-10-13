"""Evaluation: metrics (incl. the decision-threshold hit rate), slice-level analysis, error analysis and residual diagnostics.

Residual / error = actual - predicted (positive = under-prediction). MAPE is deliberately NOT used: PM2.5 spans 2-900 ug/m3 and clean-air
hours (<10 ug/m3) make relative errors unstable.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import acf

from . import config as C


# ====================================================================== metrics
def regression_metrics(y_true, y_pred, tau=C.DECISION_THRESHOLD):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    e = y_true - y_pred
    return {"MAE": float(mean_absolute_error(y_true, y_pred)), "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "R2": float(r2_score(y_true, y_pred)), "MedAE": float(median_absolute_error(y_true, y_pred)), "Bias": float(e.mean()),
            "Resid_std": float(e.std(ddof=1)), "Within_tau": float((np.abs(e) <= tau).mean()), "Neg_pred_share": float((y_pred < 0).mean())}


def threshold_table(y_true, preds, grid=C.THRESHOLD_GRID):
    """Share of predictions whose absolute error is within each candidate tolerance (decision-threshold sensitivity)."""
    rows = []
    for name, p in preds.items():
        ae = np.abs(np.asarray(y_true, float) - np.asarray(p, float))
        rows.append({"model": name, **{f"within {t:g}": float((ae <= t).mean()) for t in grid}})
    return pd.DataFrame(rows)


# ====================================================================== slices (cut-offs from TRAINING data / the calendar only)
def training_reference(y_train, X_train):
    """Cut-offs used by the slices, computed on the training rows."""
    d = (y_train.to_numpy(float) - X_train[C.CURRENT].to_numpy(float))
    return {"y_q25": float(y_train.quantile(.25)), "y_q75": float(y_train.quantile(.75)), "y_q95": float(y_train.quantile(.95)), "y_max": float(y_train.max()),
            "y_q99": float(y_train.quantile(.99)), "wspm_median": float(X_train["WSPM"].median()), "abs_change_median": float(np.median(np.abs(d))),
            "abs_change_p90": float(np.quantile(np.abs(d), .90)), "rise_p95": float(np.quantile(d[d > 0], .95)), "fall_p95": float(np.quantile(-d[d < 0], .95))}


def slice_masks(meta, ref):
    """meta columns: target_time, actual, PM25_t, WSPM.  Returns {group: {slice label: boolean mask}} (ordered)."""
    th = pd.DatetimeIndex(meta["target_time"])
    y, change = meta["actual"].to_numpy(float), meta["actual"].to_numpy(float) - meta["PM25_t"].to_numpy(float)
    day = (th.hour >= C.DAY_HOURS[0]) & (th.hour < C.DAY_HOURS[1])
    season = np.array([C.SEASONS[m] for m in th.month])
    g = {}
    g["PM2.5 level (actual)"] = {f"low (<= P25 = {ref['y_q25']:.0f})": y <= ref["y_q25"],
                                 f"mid (P25-P75)": (y > ref["y_q25"]) & (y < ref["y_q75"]),
                                 f"high (>= P75 = {ref['y_q75']:.0f})": y >= ref["y_q75"],
                                 f"episode (>= P95 = {ref['y_q95']:.0f})": y >= ref["y_q95"]}
    g["Time of day (target hour)"] = {f"day ({C.DAY_HOURS[0]:02d}-{C.DAY_HOURS[1] - 1:02d} h)": day, "night": ~day}
    g["Weekday / weekend"] = {"weekday": th.dayofweek < 5, "weekend": th.dayofweek >= 5}
    g["Season"] = {s: season == s for s in ("winter", "spring", "summer", "autumn")}
    ws = meta["WSPM"].to_numpy(float)
    g["Wind speed"] = {f"low wind (<= {ref['wspm_median']:.1f} m/s)": ws <= ref["wspm_median"], "high wind": ws > ref["wspm_median"]}
    g["One-hour change"] = {f"rapid rise (>= +{ref['abs_change_p90']:.0f})": change >= ref["abs_change_p90"], f"rapid fall (<= -{ref['abs_change_p90']:.0f})": change <= -ref["abs_change_p90"],
                            f"stable (|change| <= {ref['abs_change_median']:.0f})": np.abs(change) <= ref["abs_change_median"]}
    return g


def slice_table(y_true, preds, masks):
    """Long table: group, slice, n, model, MAE, RMSE."""
    y_true = np.asarray(y_true, float)
    rows = []
    for grp, d in masks.items():
        for lab, m in d.items():
            m = np.asarray(m, bool)
            for name, p in preds.items():
                e = y_true[m] - np.asarray(p, float)[m]
                rows.append({"group": grp, "slice": lab, "n": int(m.sum()), "model": name, "MAE": float(np.abs(e).mean()) if m.any() else np.nan,
                             "RMSE": float(np.sqrt((e ** 2).mean())) if m.any() else np.nan, "bias": float(e.mean()) if m.any() else np.nan})
    return pd.DataFrame(rows)


# ====================================================================== error analysis
def _compass_gap(a, b):
    """Angular difference (degrees) between two compass labels."""
    if a not in C.WD_LEVELS or b not in C.WD_LEVELS:
        return 0.0
    d = abs(C.WD_LEVELS.index(a) - C.WD_LEVELS.index(b)) % len(C.WD_LEVELS)
    return min(d, len(C.WD_LEVELS) - d) * 22.5


def explain_error(r, ref):
    """Rule-based POSSIBLE explanation of one large error from the data around the case (hypotheses, not established causes)."""
    notes = []
    a, cur, change = r["actual"], r["PM25_t"], r["actual"] - r["PM25_t"]
    under = r["error"] > 0
    if under and change >= ref["rise_p95"]:
        notes.append(f"sudden pollution spike: PM2.5 jumped {change:+.0f} ug/m3 within one hour (training 95th percentile of rises: +{ref['rise_p95']:.0f}); a linear model driven by the current level under-reacts to abrupt episodes")
    elif (not under) and -change >= ref["fall_p95"]:
        notes.append(f"sudden clean-up: PM2.5 fell {change:+.0f} ug/m3 within one hour (training 95th percentile of falls: -{ref['fall_p95']:.0f}); the forecast stays close to the previous high level")
    elif abs(change) >= ref["abs_change_p90"]:
        notes.append(f"rapid change of {change:+.0f} ug/m3 in one hour (above the training 90th percentile of |change| = {ref['abs_change_p90']:.0f})")
    sd6 = r.get("roll_std_6", np.nan)
    if np.isfinite(sd6) and abs(change) > 0:
        notes.append(f"limited warning in the history: the jump ({abs(change):.0f}) is {abs(change) / max(sd6, 1e-9):.1f}x the 6-hour standard deviation ({sd6:.0f}) preceding it")
    if np.isfinite(r.get("WSPM", np.nan)) and np.isfinite(r.get("WSPM_prev", np.nan)) and abs(r["WSPM"] - r["WSPM_prev"]) >= 2.0:
        notes.append(f"unusual meteorological change: wind speed moved {r['WSPM_prev']:.1f} -> {r['WSPM']:.1f} m/s")
    if isinstance(r.get("wd"), str) and isinstance(r.get("wd_prev"), str) and _compass_gap(r["wd"], r["wd_prev"]) >= 90:
        notes.append(f"wind-direction shift of {_compass_gap(r['wd'], r['wd_prev']):.0f} degrees ({r['wd_prev']} -> {r['wd']})")
    if a >= ref["y_q99"]:
        notes.append(f"extreme level (actual {a:.0f} >= training 99th percentile {ref['y_q99']:.0f}): few training examples, regression to the mean / extrapolation")
    if (not under) and abs(change) >= ref["abs_change_p90"] and np.isfinite(r.get("WSPM", np.nan)) and r["WSPM"] < ref["wspm_median"]:
        notes.append(f"the wind was calm ({r['WSPM']:.1f} m/s, below the training median {ref['wspm_median']:.1f}), so a ventilation-driven clean-up is not visible in the local wind data (possible local plume change or measurement artefact)")
    if np.isfinite(r.get("PM10", np.nan)) and r["PM10"] == cur and ref.get("pm10_equal_share") is not None:
        notes.append(f"PM10 equals PM2.5(t) exactly in this hour (a data-quality quirk present in {100 * ref['pm10_equal_share']:.0f} % of the training hours): the coarse-particle reading adds no independent information here")
    t_ = pd.Timestamp(r["time"]) if r.get("time") is not None else None
    if t_ is not None:
        for d1 in C.SPRING_FESTIVAL_DAY1:
            start, end = pd.Timestamp(d1) - pd.Timedelta(hours=6), pd.Timestamp(d1) + pd.Timedelta(days=1)       # New Year's Eve 18:00 -> end of the first day
            if start <= t_ < end:
                notes.append(f"external knowledge (not in the data): the target hour lies in the Chinese New Year fireworks window (first day {d1}); fireworks are a documented short-lived PM2.5 source in Beijing, with peaks around 01:00-02:00 of the first day")
    nxt = r.get("next_actual", np.nan)
    if np.isfinite(nxt) and abs(change) > 0 and abs(nxt - cur) <= 0.35 * abs(change):
        notes.append(f"the series returns to about its previous level one hour later (PM2.5 {nxt:.0f} vs {cur:.0f} before): an isolated one-hour excursion, possibly a short plume or a measurement glitch")
    if not notes:
        notes.append("no clear signal in the available variables (missing explanatory variable, e.g. regional transport, boundary-layer height, emissions)")
    return "; ".join(notes)


def top_errors(meta, y_true, y_pred, ref, n=C.MAX_ERRORS_SHOWN):
    """The n largest absolute errors with context. meta: one row per prediction (origin time, target time, context features)."""
    e = np.asarray(y_true, float) - np.asarray(y_pred, float)
    idx = np.argsort(-np.abs(e))[:n]
    rows = []
    for rank, i in enumerate(idx, 1):
        m = meta.iloc[i]
        r = {"case": rank, "forecast_issued": m["datetime"], "time": m["target_time"], "actual": float(y_true[i]), "predicted": float(y_pred[i]), "error": float(e[i]),
             "abs_error": float(abs(e[i])), "hour": int(m["target_time"].hour), "month": int(m["target_time"].month), "PM25_t": float(m["PM25_t"]),
             "change_actual_minus_current": float(y_true[i] - m["PM25_t"]), "PM10": m["PM10"], "NO2": m["NO2"], "CO": m["CO"], "TEMP": m["TEMP"],
             "WSPM": m["WSPM"], "wd": m["wd"], "roll_std_6": m["PM25_roll_std_6"], "WSPM_prev": m.get("WSPM_prev", np.nan), "wd_prev": m.get("wd_prev", None),
             "next_actual": m.get("next_actual", np.nan)}
        r["possible_explanation"] = explain_error({**r, "wd": m["wd"]}, ref)
        rows.append(r)
    return pd.DataFrame(rows)


def shared_pattern(tab, errors_all, meta, ref):
    """Statements about the largest errors that are checked against the data (counts, not impressions)."""
    n = len(tab)
    under = int((tab["error"] > 0).sum())
    rises = int((tab["change_actual_minus_current"] >= ref["abs_change_p90"]).sum())
    falls = int((tab["change_actual_minus_current"] <= -ref["abs_change_p90"]).sum())
    out = {"n": n, "under_predictions": under, "over_predictions": n - under, "rapid_rises": rises, "rapid_falls": falls,
           "hours": tab["hour"].tolist(), "months": tab["month"].tolist()}
    ae = np.abs(errors_all)
    top1 = ae >= np.quantile(ae, .99)
    ch = (meta["actual"].to_numpy(float) - meta["PM25_t"].to_numpy(float))
    ch_top = tab["change_actual_minus_current"].to_numpy(float)
    mult = np.where(ch_top >= 0, ch_top / ref["rise_p95"], -ch_top / ref["fall_p95"])
    out["top_abs_changes"] = np.abs(ch_top).tolist()
    out["top_change_multiple_of_p95"] = mult.tolist()
    out["top_forecast_minus_current"] = (tab["predicted"] - tab["PM25_t"]).to_numpy(float).tolist()
    out["all_top_above_p95"] = bool((mult >= 1).all())
    out["top1pct_n"] = int(top1.sum())
    out["top1pct_share_sq_error"] = float((errors_all[top1] ** 2).sum() / (errors_all ** 2).sum())
    out["top1pct_rapid_change_share"] = float((np.abs(ch[top1]) >= ref["abs_change_p90"]).mean())
    out["top1pct_under_share"] = float((errors_all[top1] > 0).mean())
    return out


# ====================================================================== residual diagnostics
def residual_diagnostics(y_true, y_pred, ref):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    e = y_true - y_pred
    out = {}
    bp = het_breuschpagan(e, sm.add_constant(y_pred))
    out["breusch_pagan_lm"], out["breusch_pagan_p"] = float(bp[0]), float(bp[1])
    out["spearman_abs_resid_vs_pred"] = float(stats.spearmanr(np.abs(e), y_pred)[0])
    dec = pd.qcut(y_pred, 10, duplicates="drop")
    spread = pd.DataFrame({"pred": y_pred, "resid": e, "abs": np.abs(e), "bin": dec}).groupby("bin", observed=True).agg(
        n=("resid", "size"), pred_mean=("pred", "mean"), resid_mean=("resid", "mean"), resid_std=("resid", "std"), mae=("abs", "mean")).reset_index(drop=True)
    out["spread_by_pred_decile"] = spread
    out["std_ratio_top_vs_bottom_decile"] = float(spread["resid_std"].iloc[-1] / spread["resid_std"].iloc[0])
    out["durbin_watson"] = float(durbin_watson(e))
    a = acf(e, nlags=168, fft=True)
    out["acf"] = {int(k): float(a[k]) for k in (1, 2, 3, 6, 12, 24, 48, 168)}
    lb = acorr_ljungbox(e, lags=[1, 24, 168], return_df=True)
    out["ljung_box_p"] = {int(l): float(p) for l, p in zip([1, 24, 168], lb["lb_pvalue"].to_numpy())}
    out["mean_error"], out["share_under_predicted"] = float(e.mean()), float((e > 0).mean())
    out["skew"], out["excess_kurtosis"] = float(stats.skew(e)), float(stats.kurtosis(e))
    out["jarque_bera_p"] = float(stats.jarque_bera(e)[1])
    ext = y_true >= ref["y_q95"]
    out["extreme_n"], out["extreme_mean_error"], out["extreme_mae"] = int(ext.sum()), float(e[ext].mean()), float(np.abs(e[ext]).mean())
    low = y_true <= ref["y_q25"]
    out["low_mean_error"], out["low_mae"] = float(e[low].mean()), float(np.abs(e[low]).mean())
    return out
