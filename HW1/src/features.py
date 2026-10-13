"""Causal feature engineering for the next-hour task  X_t -> PM2.5_(t+1).

Row t of the feature matrix may depend only on
  * PM2.5 up to and including hour t (current value + lags + rolling statistics of the *shifted* series),
  * the other pollutants / meteorology at t (and at t-3 for the 3-hour tendency of the dew point),
  * the calendar of hour t.
The label PM2.5(t+1) is stored in `target` and is never an input of any feature.
"""
import numpy as np
import pandas as pd
from scipy import stats

from . import config as C

# ====================================================================== names / groups
CURRENT = C.CURRENT
LAG_COLS = [f"PM25_lag_{k}" for k in C.LAGS]
ROLL_MEAN_COLS = [f"PM25_roll_mean_{w}" for w in C.ROLL_MEAN_W]
ROLL_STD_COLS = [f"PM25_roll_std_{w}" for w in C.ROLL_STD_W]
ROLL_COLS = ROLL_MEAN_COLS + ROLL_STD_COLS
HISTORY_COLS = LAG_COLS + ROLL_COLS
TEMPORAL_COLS = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos", "is_weekend"]
POLLUTANTS = list(C.POLLUTANTS)
METEO_NUMERIC = list(C.METEO_NUMERIC)                      # TEMP, PRES, DEWP, RAIN, WSPM
WIND_DIR = C.WIND_DIR
DERIVED_CANDIDATES = ["TEMP_minus_DEWP", "PRES_chg_3h", "TEMP_chg_3h", "DEWP_chg_3h", "PM25_PM10_ratio"]

DERIVED_RATIONALE = {
    "TEMP_minus_DEWP": "dew-point depression: a humidity proxy (small = moist air, hygroscopic growth / secondary aerosol formation)",
    "PRES_chg_3h": "3-hour pressure tendency: falling pressure marks stagnation / approaching fronts",
    "TEMP_chg_3h": "3-hour temperature tendency (temperature-related difference)",
    "DEWP_chg_3h": "3-hour dew-point tendency: rising moisture accompanies accumulation episodes",
    "PM25_PM10_ratio": "fine-particle fraction PM2.5(t)/PM10(t): secondary/combustion aerosol vs coarse dust",
}
# (input a, input b, name): products of MEAN-CENTRED inputs (means learned inside the training fold)
INTERACTION_PAIRS = [("TEMP", "DEWP", "TEMP_x_DEWP"), ("TEMP", "PM10", "TEMP_x_PM10"), ("WSPM", "PM10", "WSPM_x_PM10"),
                     ("WSPM", CURRENT, "WSPM_x_PM25"), ("TEMP", "hour_sin", "TEMP_x_hour_sin"), ("DEWP", "hour_sin", "DEWP_x_hour_sin")]
INTERACTION_RATIONALE = {
    "TEMP_x_DEWP": "the effect of temperature on aerosol formation depends on moisture (humidity-temperature coupling)",
    "TEMP_x_PM10": "the PM10->PM2.5 relation changes with season/temperature (combustion-dominated cold season vs dust-dominated warm season)",
    "WSPM_x_PM10": "ventilation dilutes the particle load: the effect of PM10 weakens at high wind speed",
    "WSPM_x_PM25": "ventilation shortens the persistence of the current PM2.5 (training slope of PM2.5(t+1) on PM2.5(t): 0.97 at low vs 0.90 at high wind) [added to the brief's list]",
    "TEMP_x_hour_sin": "diurnal boundary-layer dynamics modulate the temperature effect",
    "DEWP_x_hour_sin": "diurnal boundary-layer dynamics modulate the moisture effect",
}


def interaction_names():
    return [n for *_, n in INTERACTION_PAIRS]


# ====================================================================== feature construction
def build_candidate_features(frame, locf_limit=C.LOCF_LIMIT):
    """All candidate predictors for the rows of `frame` (chronological, regular hourly grid, raw columns).

    PM2.5 history is built from the series after *causal* last-observation-carried-forward over gaps of at most `locf_limit` hours
    (feature series only: the target is built from the raw series and is never filled). `PM25_t` is the raw current value.
    Rolling statistics are computed on the series shifted by one hour (window = t-w ... t-1) and always require a full window.
    """
    pm = frame[C.RAW_TARGET].astype(float)
    pm_hist = pm.ffill(limit=locf_limit) if locf_limit else pm.copy()
    out = pd.DataFrame(index=frame.index)
    out[CURRENT] = pm.to_numpy()
    for k in C.LAGS:
        out[f"PM25_lag_{k}"] = pm_hist.shift(k).to_numpy()                       # PM2.5(t-k)
    past = pm_hist.shift(1)                                                       # shift first, then roll: window ends at t-1
    for w in C.ROLL_MEAN_W:
        out[f"PM25_roll_mean_{w}"] = past.rolling(w, min_periods=w).mean().to_numpy()
    for w in C.ROLL_STD_W:
        out[f"PM25_roll_std_{w}"] = past.rolling(w, min_periods=w).std().to_numpy()
    dtv = pd.DatetimeIndex(frame["datetime"])                                     # calendar of the CURRENT hour t
    out["hour_sin"], out["hour_cos"] = np.sin(2 * np.pi * dtv.hour / 24), np.cos(2 * np.pi * dtv.hour / 24)
    out["dow_sin"], out["dow_cos"] = np.sin(2 * np.pi * dtv.dayofweek / 7), np.cos(2 * np.pi * dtv.dayofweek / 7)
    out["month_sin"], out["month_cos"] = np.sin(2 * np.pi * dtv.month / 12), np.cos(2 * np.pi * dtv.month / 12)
    out["is_weekend"] = (dtv.dayofweek >= 5).astype(float)
    for c in POLLUTANTS + METEO_NUMERIC:                                          # current measurements at t
        out[c] = frame[c].astype(float).to_numpy()
    out[WIND_DIR] = frame[WIND_DIR].to_numpy()
    out["TEMP_minus_DEWP"] = (frame["TEMP"] - frame["DEWP"]).to_numpy()
    for c in ("PRES", "TEMP", "DEWP"):
        out[f"{c}_chg_3h"] = (frame[c] - frame[c].shift(3)).to_numpy()            # value(t) - value(t-3)
    ratio = frame[C.RAW_TARGET] / frame["PM10"].where(frame["PM10"] > 0)
    out["PM25_PM10_ratio"] = ratio.to_numpy()
    return out


def eligibility(frame, feats):
    """A row is usable iff its label is observed, the current PM2.5 is observed and the PM2.5 history features can be computed.
    The rule depends only on data availability (never on label values) and is applied identically to every feature set / split."""
    ok = frame[C.TARGET].notna() & feats[CURRENT].notna()
    for c in HISTORY_COLS:
        ok &= feats[c].notna()
    return ok.to_numpy()


# ====================================================================== EDA-supported derived quantities (training rows only)
def _spearman(a, b):
    m = pd.notna(a) & pd.notna(b)
    return float(stats.spearmanr(np.asarray(a)[m], np.asarray(b)[m])[0]) if m.sum() > 10 else np.nan


def derived_candidate_table(feats_train, y_train):
    """Spearman of every candidate derived quantity with the target and with the next-hour change, on training rows only."""
    change = (y_train - feats_train[CURRENT]).to_numpy()
    rows = []
    for c in DERIVED_CANDIDATES:
        r_t, r_c = _spearman(feats_train[c].to_numpy(), y_train.to_numpy()), _spearman(feats_train[c].to_numpy(), change)
        rows.append({"candidate": c, "rationale": DERIVED_RATIONALE[c], "rho_with_target": r_t, "rho_with_next_hour_change": r_c,
                     "max_abs_rho": max(abs(r_t), abs(r_c)), "kept": bool(max(abs(r_t), abs(r_c)) >= C.DERIVED_MIN_ABS_RHO)})
    return pd.DataFrame(rows)


# ====================================================================== feature-set specifications
def build_specs(derived):
    """Specifications of the feature sets. A spec is {numeric, categorical, helpers, pairs}.

    * Set A: the raw current measurements (PM2.5(t), pollutants, meteorology, wind direction -> one-hot).
    * Set C: A plus the engineered groups (temporal, lags, rolling, derived differences, interactions).
      DEWP is replaced by TEMP-DEWP in the main effects (DEWP = TEMP - (TEMP-DEWP) is in the span; keeping all three would be exactly
      collinear). DEWP stays available as an input of the interaction terms ('helper').
    * Pool: every legitimate candidate (used by the correlation filter B').
    """
    derived = list(derived)
    met_main = [m for m in METEO_NUMERIC if not (m == "DEWP" and "TEMP_minus_DEWP" in derived)]
    helpers = ["DEWP"] if ("TEMP_minus_DEWP" in derived) else []
    spec_a = {"numeric": [CURRENT] + POLLUTANTS + METEO_NUMERIC, "categorical": [WIND_DIR], "helpers": [], "pairs": []}
    spec_c = {"numeric": [CURRENT] + LAG_COLS + ROLL_COLS + TEMPORAL_COLS + POLLUTANTS + met_main + derived, "categorical": [WIND_DIR],
              "helpers": helpers, "pairs": list(INTERACTION_PAIRS)}
    pool = {"numeric": [CURRENT] + LAG_COLS + ROLL_COLS + TEMPORAL_COLS + POLLUTANTS + METEO_NUMERIC + DERIVED_CANDIDATES,
            "categorical": [WIND_DIR], "helpers": [], "pairs": list(INTERACTION_PAIRS)}
    return {"A": spec_a, "C": spec_c, "pool": pool}


_VARS = {
    "temporal": set(TEMPORAL_COLS),
    "lag": set(LAG_COLS),
    "rolling": set(ROLL_COLS),
    "pm25": {CURRENT} | set(LAG_COLS) | set(ROLL_COLS),
    "pollutants": set(POLLUTANTS) | {"PM25_PM10_ratio"},
    "meteo": set(METEO_NUMERIC) | {WIND_DIR, "TEMP_minus_DEWP", "PRES_chg_3h", "TEMP_chg_3h", "DEWP_chg_3h"},
}


def drop_groups(spec, *groups):
    """Spec with whole feature groups removed (ablation). Interaction terms disappear with any of their inputs.

    groups: temporal | lag | rolling | history (= lag + rolling, keeps PM2.5(t)) | pm25 (current + lags + rolling) | pollutants |
            meteo (meteorology + wind direction + derived tendencies) | env (= pollutants + meteo + all interactions) |
            env_diff (derived differences only) | interaction
    """
    drop = set()
    drop_pairs = False
    for g in groups:
        if g in _VARS:
            drop |= _VARS[g]
        elif g == "history":
            drop |= _VARS["lag"] | _VARS["rolling"]
        elif g == "env":
            drop |= _VARS["pollutants"] | _VARS["meteo"]
            drop_pairs = True
        elif g == "env_diff":
            drop |= {"TEMP_minus_DEWP", "PRES_chg_3h", "TEMP_chg_3h", "DEWP_chg_3h", "PM25_PM10_ratio"}
        elif g == "interaction":
            drop_pairs = True
        else:
            raise KeyError(g)
    numeric = [c for c in spec["numeric"] if c not in drop]
    categorical = [c for c in spec["categorical"] if c not in drop]
    pairs = [] if drop_pairs else [p for p in spec["pairs"] if p[0] not in drop and p[1] not in drop]
    needed = {x for p in pairs for x in p[:2]}
    helpers = [h for h in spec["helpers"] if h in needed and h not in drop]
    missing = [x for x in needed if x not in numeric and x not in helpers]
    assert not missing, f"interaction inputs missing from the spec after dropping {groups}: {missing}"
    return {"numeric": numeric, "categorical": categorical, "helpers": helpers, "pairs": pairs}


def spec_columns(spec):
    """Columns the pipeline needs from the candidate matrix."""
    return list(spec["numeric"]) + list(spec["helpers"]) + list(spec["categorical"])


def group_of_design_column(name):
    """Group label of a column of the final design matrix (used in coefficient tables)."""
    if name == CURRENT:
        return "current PM2.5"
    if name in LAG_COLS:
        return "lag"
    if name in ROLL_COLS:
        return "rolling"
    if name in TEMPORAL_COLS:
        return "temporal"
    if name in POLLUTANTS:
        return "pollutants"
    if name in ("TEMP", "PRES", "DEWP", "RAIN", "WSPM") or name.startswith("wd_"):
        return "meteorology / wind"
    if name in DERIVED_CANDIDATES:
        return "derived differences"
    if name in interaction_names():
        return "interaction"
    return "other"


def feature_group_table(specs, derived_table, acf1=None):
    """Table 2: feature groups, members, rationale and the hypothesis they test."""
    kept = derived_table.loc[derived_table["kept"], "candidate"].tolist()
    c = specs["C"]
    rows = [
        ("Current PM2.5", [CURRENT], "The latest observed PM2.5 is the natural anchor of a next-hour forecast" + (" (lag-1 autocorrelation %.2f in training)." % acf1 if acf1 is not None else "."), "H1"),
        ("Temporal", TEMPORAL_COLS, "PM2.5 shows a daily cycle (night maximum, afternoon minimum), a seasonal cycle (Oct-Dec high, Aug low) and a small weekend effect in the "
                                    "training EDA; cyclical sin/cos encodings avoid artificial jumps at midnight / year end.", "H3"),
        ("PM2.5 lags", LAG_COLS, "Concentrations persist over hours (1-6 h), days (24 / 48 h) and weeks (168 h); individual lags capture specific recurrences.", "H1, H4"),
        ("PM2.5 rolling statistics", ROLL_COLS, "Means and standard deviations of the shifted series (window ends at t-1) summarise recent level and volatility and average out single noisy readings.", "H4"),
        ("Pollutants (raw)", POLLUTANTS, "Co-pollutants share sources and meteorology with PM2.5 (PM10 and CO are strongly correlated with the target in the training EDA).", "H2"),
        ("Meteorology (raw)", ["TEMP", "PRES", "RAIN", "WSPM", "wd (15 one-hot dummies, reference N)"], "Wind speed/direction ventilate the air; precipitation scavenges particles; pressure and temperature mark stagnation. "
                                                                                                         "Wind direction is one-hot encoded (no artificial ordering); DEWP enters through TEMP-DEWP.", "H2"),
        ("Derived differences", kept, "Physically motivated quantities that passed the training-only EDA rule (|Spearman| >= %.2f with the target or the next-hour change): "
                                     "moisture and aerosol-composition indicators." % C.DERIVED_MIN_ABS_RHO, "H2"),
        ("Interactions (centred products)", interaction_names(), "Context-dependent effects: ventilation x load, temperature x moisture, seasonal composition, diurnal modulation. "
                                                                  "Inputs are mean-centred inside each training fold to limit artificial collinearity.", "H2 (context)"),
    ]
    return pd.DataFrame([{"group": g, "features": ", ".join(f), "n_features": len(f), "rationale": r, "tests": h} for g, f, r, h in rows])


# ====================================================================== leakage checks
def _synthetic_frame(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    f = pd.DataFrame({"datetime": pd.date_range("2013-03-01 00:00", periods=n, freq="h")})
    pm = rng.gamma(2.0, 40.0, n).round(1)
    gaps = rng.integers(0, n - 60, 14)
    for g in gaps:                                                    # gaps of 1..9 hours, including long ones
        pm[g:g + rng.integers(1, 10)] = np.nan
    f[C.RAW_TARGET] = pm
    for c in POLLUTANTS + METEO_NUMERIC:
        f[c] = rng.normal(50, 10, n)
    f[WIND_DIR] = rng.choice(C.WD_LEVELS, n)
    f[C.TARGET] = f[C.RAW_TARGET].shift(-1)
    return f.iloc[:-1].reset_index(drop=True)


def _assert_frames_equal(a, b, msg):
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True), check_exact=True, obj=msg)


def _leakage_on_frame(frame, label, checks):
    """Perturbation / identity / prefix tests of the feature builder on one frame."""
    feats = build_candidate_features(frame)
    pm = frame[C.RAW_TARGET].to_numpy()
    n = len(frame)
    rng = np.random.default_rng(7)
    # (1) target is never an input and has the right alignment
    assert C.TARGET not in feats.columns and not any(c.startswith("target") for c in feats.columns)
    assert np.array_equal(frame[C.TARGET].to_numpy()[:-1], pm[1:], equal_nan=True), "target(t) != PM2.5(t+1)"
    assert np.array_equal(feats[CURRENT].to_numpy(), pm, equal_nan=True), "PM25_t is not the current PM2.5"
    checks.append((f"[{label}] target = PM2.5(t+1); PM25_t = PM2.5(t); target not among features", True))
    # (2) lags are exact shifts of the (LOCF-filled) history; rolling windows end at t-1 (brute force)
    hist = frame[C.RAW_TARGET].ffill(limit=C.LOCF_LIMIT).to_numpy()
    for k in C.LAGS:
        col = feats[f"PM25_lag_{k}"].to_numpy()
        assert np.isnan(col[:k]).all() and np.array_equal(col[k:], hist[:-k], equal_nan=True), f"lag {k}"
    for t in rng.integers(C.WARMUP_HOURS, n, 150):
        for w in C.ROLL_MEAN_W:
            win = hist[t - w:t]
            exp = win.mean() if not np.isnan(win).any() else np.nan
            got = feats[f"PM25_roll_mean_{w}"].iloc[t]
            assert (np.isnan(exp) and np.isnan(got)) or np.isclose(exp, got), f"roll_mean_{w} at {t} must be mean(PM2.5(t-1..t-{w}))"
        for w in C.ROLL_STD_W:
            win = hist[t - w:t]
            exp = win.std(ddof=1) if not np.isnan(win).any() else np.nan
            got = feats[f"PM25_roll_std_{w}"].iloc[t]
            assert (np.isnan(exp) and np.isnan(got)) or np.isclose(exp, got), f"roll_std_{w} at {t}"
    checks.append((f"[{label}] lag_k = PM2.5(t-k); rolling windows = PM2.5(t-1..t-w) (brute-force match, shifted, not centred)", True))
    # (3) future PM2.5 never enters row t: replace PM2.5(t0+1 ...) and the label by noise -> every feature of rows <= t0 is unchanged
    for t0 in rng.integers(C.WARMUP_HOURS + 5, n - 3, 15):
        pert = frame.copy()
        pert.loc[t0 + 1:, C.RAW_TARGET] = rng.normal(100, 30, n - t0 - 1)
        pert[C.TARGET] = rng.normal(100, 30, n)
        alt = build_candidate_features(pert)
        _assert_frames_equal(feats.iloc[:t0 + 1], alt.iloc[:t0 + 1], "features of rows <= t0 changed when PM2.5(>t0) / label were perturbed")
        assert np.array_equal([alt[CURRENT].iloc[t0]], [feats[CURRENT].iloc[t0]], equal_nan=True)  # PM2.5(t0) itself is allowed at row t0
        pos = frame.copy()
        pos.loc[t0, C.RAW_TARGET] = (0.0 if not pd.isna(pm[t0]) and pm[t0] > 1 else 999.0)         # positive control: changing PM2.5(t0) must matter
        assert not np.array_equal([build_candidate_features(pos)[CURRENT].iloc[t0]], [feats[CURRENT].iloc[t0]], equal_nan=True), "positive control failed"
    checks.append((f"[{label}] perturbing PM2.5(t>t0) and the label leaves all features of rows <= t0 unchanged (positive control passed)", True))
    # (4) future meteorology / pollutants never enter row t
    for t0 in rng.integers(C.WARMUP_HOURS + 5, n - 3, 8):
        pert = frame.copy()
        for c in POLLUTANTS + METEO_NUMERIC:
            pert.loc[t0 + 1:, c] = rng.normal(0, 99, n - t0 - 1)
        _assert_frames_equal(feats.iloc[:t0 + 1], build_candidate_features(pert).iloc[:t0 + 1], "features of rows <= t0 changed when future meteorology was perturbed")
    checks.append((f"[{label}] perturbing future meteorology / pollutants leaves rows <= t0 unchanged", True))
    # (5) prefix invariance: features of the first m rows do not depend on any later row (train features cannot depend on the test period)
    m = n // 2
    _assert_frames_equal(feats.iloc[:m], build_candidate_features(frame.iloc[:m]), "prefix invariance violated")
    checks.append((f"[{label}] features of the first half are identical whether or not the second half exists (prefix invariance)", True))
    # (6) calendar / meteorology features do not depend on PM2.5 at all
    blank = frame.copy()
    blank[C.RAW_TARGET] = np.nan
    alt = build_candidate_features(blank)
    non_pm = [c for c in feats.columns if not c.startswith("PM25_") and c != CURRENT]
    _assert_frames_equal(feats[non_pm], alt[non_pm], "non-PM2.5 features depend on PM2.5")
    checks.append((f"[{label}] calendar and meteorological features do not use PM2.5", True))


def check_no_data_leakage(train_frame=None, X_train=None, y_train=None, spec=None, cv=None, vault=None, verbose=True):
    """Run all leakage checks; raise AssertionError at the first violation. Returns a DataFrame (check, passed).

    Covers: target never an input / correct alignment; lags and rolling statistics use only the past (no centred windows); future PM2.5 and
    future meteorology never reach a feature; train features are independent of any later row; imputers, scalers and the correlation
    filter are fitted on the training rows of each CV fold only; validation folds lie strictly after their training folds; the test vault
    is locked."""
    from . import preprocessing as P                     # local import: preprocessing depends on features for specs
    checks = []
    _leakage_on_frame(_synthetic_frame(), "synthetic", checks)
    if train_frame is not None:
        _leakage_on_frame(train_frame.reset_index(drop=True), "real training rows", checks)
    if X_train is not None and spec is not None and cv is not None:
        from sklearn.linear_model import Ridge
        assert not any(c in X_train.columns for c in (C.TARGET, "target")), "label column present in X_train"
        for k, (tr, va) in enumerate(cv.split(X_train), 1):
            assert tr.max() < va.min(), f"fold {k}: validation not strictly after training"
            Xtr, ytr = X_train.iloc[tr], y_train.iloc[tr]
            pipe = P.build_pipeline(spec, Ridge(alpha=1.0)).fit(Xtr, ytr)
            imp = pipe.named_steps["impute"]
            for c, med in imp.statistics_.items():                                   # imputer fitted on the training fold only
                if c in Xtr.columns and Xtr[c].dtype != object:
                    assert np.isclose(med, Xtr[c].median()), f"fold {k}: imputer statistic of {c} is not the training-fold median"
            Ztr = pipe[:-2].transform(Xtr)
            sc = pipe.named_steps["scale"]
            assert np.allclose(sc.mean_, np.asarray(Ztr, float).mean(0)), f"fold {k}: scaler mean differs from the training-fold mean"
            assert sc.n_samples_seen_ == len(tr), f"fold {k}: scaler saw {sc.n_samples_seen_} rows, expected {len(tr)}"
            sel = P.build_pipeline(spec, Ridge(alpha=1.0), topk={"k": 5, "method": "spearman"}).fit(Xtr, ytr)
            Zpre = sel[:-3].transform(Xtr)                                                  # design before the filter (cols, impute, products, wind)
            score = Zpre.apply(lambda s_: float(np.nan_to_num(abs(stats.spearmanr(s_, ytr)[0]))))
            assert set(sel.named_steps["select"].columns_) == set(score.nlargest(5).index), f"fold {k}: top-k differs from the ranking on the training fold"
        checks.append(("[CV] imputer medians, scaler mean/count and top-k ranking come from the training part of every fold; validation strictly after training", True))
    if vault is not None:
        assert vault.unlock_count == 0, "the test vault has already been opened"
        checks.append(("[vault] test rows still locked while designing / tuning", True))
    out = pd.DataFrame(checks, columns=["check", "passed"])
    if verbose:
        for c, ok in checks:
            print("  ✓" if ok else "  ✗", c)
    return out
