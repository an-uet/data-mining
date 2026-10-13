"""Publication-quality figures. One function per figure; every function returns the Matplotlib figure and saves PNG (300 dpi) + PDF.

Design: validated categorical palette in fixed order, one blue ramp for magnitude, blue<->red diverging with a neutral midpoint, thin marks,
recessive solid grid, legends for >= 2 series, no dual axes, text in neutral ink (never in series colours)."""
import os

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

from . import config as C

PAL = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100", "magenta": "#e87ba4", "green": "#008300", "violet": "#4a3aa7", "red": "#e34948"}
INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7", "mid": "#f0efec"}
CONTEXT = "#b4b2a8"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ_CMAP = LinearSegmentedColormap.from_list("seq_blue", BLUE_RAMP)
DIV_CMAP = LinearSegmentedColormap.from_list("div_bluered", [PAL["blue"], INK["mid"], PAL["red"]])
FIG_CAPTIONS = {}


def apply_style():
    mpl.rcParams.update({
        "figure.dpi": 100, "savefig.dpi": 300, "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "font.family": "DejaVu Sans", "font.size": 9.5, "text.color": INK["primary"],
        "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlecolor": INK["primary"],
        "axes.labelsize": 9.5, "axes.labelcolor": INK["secondary"], "axes.edgecolor": INK["axis"], "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
        "axes.grid": True, "grid.color": INK["grid"], "grid.linewidth": 0.8, "grid.linestyle": "-",
        "xtick.color": INK["axis"], "ytick.color": INK["axis"], "xtick.labelcolor": INK["secondary"], "ytick.labelcolor": INK["secondary"],
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "lines.linewidth": 1.5, "lines.solid_capstyle": "round",
        "legend.frameon": False, "legend.fontsize": 8.5, "legend.labelcolor": INK["secondary"],
        "axes.prop_cycle": cycler(color=[PAL[k] for k in ("blue", "orange", "aqua", "yellow", "magenta", "green", "violet", "red")]),
        "pdf.fonttype": 42, "ps.fonttype": 42})


apply_style()


def save_fig(fig, stem, caption, show=False):
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(C.FIGURES / f"{stem}.png", bbox_inches="tight")
    fig.savefig(C.FIGURES / f"{stem}.pdf", bbox_inches="tight")
    FIG_CAPTIONS[stem] = caption
    if show:
        plt.show()
    else:
        plt.close(fig)
    return stem


def set_caption(stem, text):
    """Override the default caption of a saved figure (used by the experiment to state numbers computed from the data)."""
    FIG_CAPTIONS[stem] = text


def _date_axis(ax):
    ax.xaxis.set_major_locator(mpl.dates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%b\n%Y"))


def _grid(ax, which="y"):
    ax.grid(False)
    if which in ("x", "both"):
        ax.grid(axis="x")
    if which in ("y", "both"):
        ax.grid(axis="y")
    return ax


# ====================================================================== Figure 1: pipeline
def fig_pipeline(n_rows, n_cols, test_fraction, n_splits):
    fig, ax = plt.subplots(figsize=(13.0, 6.4))
    ax.set_xlim(0, 100), ax.set_ylim(0, 53), ax.axis("off"), ax.grid(False)
    FC = {"data": INK["mid"], "train": "#d6e6fa", "test": "#fbd9cb"}

    def box(x, y, w, h, title, body, kind):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=1.2", fc=FC[kind], ec=INK["axis"], lw=1.0))
        ax.text(x + w / 2, y + h - 2.1, title, ha="center", va="top", fontsize=9.3, fontweight="bold", color=INK["primary"])
        ax.text(x + w / 2, y + h - 5.7, body, ha="center", va="top", fontsize=8.0, color=INK["secondary"], linespacing=1.35)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color=INK["secondary"], lw=1.3, shrinkA=0, shrinkB=0, mutation_scale=11))

    W, H, xs = 23.0, 11.2, [1.0, 26.0, 51.0, 76.0]
    r1, r2, r3 = 36.5, 20.0, 3.5
    spec = [
        (xs[0], r1, "1  Raw data", f"UCI #501 · Aotizhongxin station\n{n_rows:,} hourly rows × {n_cols} columns", "data"),
        (xs[1], r1, "2  Parse & validate", "datetime from year/month/day/hour\nsort · regular hourly grid (0 gaps)", "data"),
        (xs[2], r1, "3  Target + time split", f"target = PM2.5(t+1); last row dropped\n{100 * (1 - test_fraction):.0f} % train | {100 * test_fraction:.0f} % test (locked)\nno shuffling", "data"),
        (xs[3], r1, "4  Causal features", "current PM2.5 · lags · shifted rolling\ncalendar · differences · interactions", "train"),
        (xs[3], r2, "5  EDA + hypotheses", "training rows only; H1–H4 and the\ndecision rules pre-registered", "train"),
        (xs[2], r2, "6  Feature sets", "A raw · B top-k correlated ·\nC engineered (leakage-tested)", "train"),
        (xs[1], r2, "7  TimeSeriesSplit CV", f"{n_splits} expanding folds; imputer, scaler\nand top-k fitted inside each\ntraining fold", "train"),
        (xs[0], r2, "8  Model selection", "OLS / Ridge / Lasso / Elastic Net\nCV-MAE; 1-SE tie → rule from\nthe data properties", "train"),
        (xs[0], r3, "9  Ablation & inference", "A/B/C + group removal;\nHAC OLS · bootstrap CI · VIF", "train"),
        (xs[1], r3, "10  Freeze design", "feature lists, hyper-parameters,\nfolds, seed → SHA-256", "train"),
        (xs[2], r3, "11  Open test set once", "predict with frozen models;\nMAE · RMSE · R² · DM tests", "test"),
        (xs[3], r3, "12  Slices, errors, residuals", "slice analysis, 3 largest errors,\nresidual diagnostics", "data")]
    for x, y, t, b, k in spec:
        box(x, y, W, H, t, b, k)
    for i in range(3):
        arrow(xs[i] + W, r1 + H / 2, xs[i + 1], r1 + H / 2)
    arrow(xs[3] + W / 2, r1, xs[3] + W / 2, r2 + H)
    for i in (3, 2, 1):
        arrow(xs[i], r2 + H / 2, xs[i - 1] + W, r2 + H / 2)
    arrow(xs[0] + W / 2, r2, xs[0] + W / 2, r3 + H)
    for i in range(3):
        arrow(xs[i] + W, r3 + H / 2, xs[i + 1], r3 + H / 2)
    for k, lab, x in [("data", "data / reporting", 2.0), ("train", "uses training rows only", 24.0), ("test", "held-out test (opened once)", 52.0)]:
        ax.add_patch(FancyBboxPatch((x, 50.2), 2.6, 1.8, boxstyle="round,pad=0,rounding_size=0.4", fc=FC[k], ec=INK["axis"], lw=0.9))
        ax.text(x + 3.5, 51.1, lab, va="center", fontsize=8.5, color=INK["secondary"])
    return save_fig(fig, "figure_01_prediction_pipeline", "Prediction pipeline for X(t) → PM2.5(t+1). Blue boxes use training rows only; the held-out test period (orange) is opened once per run, after the design has been frozen and fingerprinted.")


def fig_split_timeline(info, folds):
    """Chronological train/test split and the expanding-window CV folds on the usable training rows (dates only - no test values are drawn)."""
    import matplotlib.dates as mdates
    n = len(folds)
    fig, ax = plt.subplots(figsize=(10.8, 0.52 * (n + 1) + 1.5))
    rows = [("final split", [(info["train_start"], info["train_end"], PAL["blue"], "training / CV period"), (info["test_start"], info["test_end"], PAL["orange"], "held-out test (locked until the freeze)")])]
    for f in folds:
        rows.append((f"CV fold {f['fold']}", [(f["train_time"][0], f["train_time"][1], PAL["blue"], None), (f["val_time"][0], f["val_time"][1], PAL["aqua"], "validation block")]))
    done = set()
    for i, (lab, segs) in enumerate(rows):
        y = len(rows) - 1 - i
        for a, b, col, name in segs:
            x0, x1 = mdates.date2num(pd.Timestamp(a)), mdates.date2num(pd.Timestamp(b))
            ax.barh(y, x1 - x0, left=x0, height=0.56, color=col, label=name if name and name not in done else None)
            done.add(name)
    ax.set_yticks(range(len(rows))), ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=8.8)
    ax.xaxis_date(), ax.xaxis.set_major_locator(mdates.YearLocator()), ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.axvline(mdates.date2num(pd.Timestamp(info["test_start"])), color=INK["secondary"], lw=1.0, ls=(0, (4, 3)))
    ax.set_title("Chronological split and expanding-window cross-validation folds"), ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=8.5), _grid(ax, "x")
    return save_fig(fig, "figure_S08_split_and_cv_folds", "Chronological 80/20 split and the five expanding-window folds used for every model.")


# ====================================================================== EDA figures (training rows only)
def fig_distribution(y, mean, median):
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.5))
    ax = axes[0]
    ax.hist(y, bins=np.arange(0, 610, 10), color=PAL["blue"], edgecolor="white", linewidth=0.5)
    top = ax.get_ylim()[1]
    for v, lab, yy in [(median, f"median {median:.0f}", 0.93), (mean, f"mean {mean:.0f}", 0.80)]:
        ax.axvline(v, color=INK["secondary"], lw=1.1)
        ax.text(v + 8, top * yy, lab, color=INK["secondary"], fontsize=8.5, va="top")
    ax.text(0.98, 0.55, f"{int((y > 600).sum())} hours > 600 not shown\n(maximum {y.max():.0f})", transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color=INK["secondary"])
    ax.set_xlabel("PM2.5 at t+1 (µg/m³)"), ax.set_ylabel("hours"), ax.set_xlim(0, 600), ax.set_title("(a) Histogram, 10 µg/m³ bins"), _grid(ax, "y")
    ax = axes[1]
    v = np.sort(np.asarray(y, float))
    ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=PAL["blue"], lw=1.8)
    for q in (0.5, 0.9, 0.99):
        xq = np.quantile(v, q)
        ax.plot([xq], [q], "o", ms=6.5, color=PAL["blue"], mec="white", mew=1.6, zorder=3)
        ax.text(xq * 1.12, q - 0.05, f"P{int(q * 100)} = {xq:.0f}", fontsize=8.5, color=INK["secondary"], va="top")
    ax.set_xscale("log"), ax.minorticks_off(), ax.set_xlim(2, 1200), ax.set_ylim(0, 1.02)
    ax.set_xticks([5, 10, 20, 50, 100, 200, 500, 1000]), ax.set_xticklabels([5, 10, 20, 50, 100, 200, 500, 1000])
    ax.set_xlabel("PM2.5 at t+1 (µg/m³, log scale)"), ax.set_ylabel("cumulative share of hours"), ax.set_title("(b) Empirical CDF"), _grid(ax, "both")
    fig.tight_layout()
    return save_fig(fig, "figure_02_pm25_distribution", "Distribution of the target (PM2.5 one hour ahead) in the training rows. The series is right-skewed (skewness ≈ 1.9): the median is 60 µg/m³ while episodes exceed 400 µg/m³. Heavy tails make squared-error metrics (RMSE) much larger than MAE.")


def fig_temporal_pattern(E):
    """E: DataFrame with target_time and target (training rows)."""
    th = pd.DatetimeIndex(E["target_time"])
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.7))
    ax = axes[0]
    g = E.groupby(th.hour)["target"]
    mean, q1, q3 = g.mean(), g.quantile(.25), g.quantile(.75)
    ax.fill_between(mean.index, q1, q3, color=PAL["blue"], alpha=0.14, lw=0)
    ax.plot(mean.index, mean.values, color=PAL["blue"], lw=1.8)
    ax.set_xlabel("hour of the target time"), ax.set_ylabel("PM2.5 (µg/m³)"), ax.set_xticks(range(0, 24, 3)), ax.set_xlim(0, 23)
    ax.set_title("(a) Daily cycle: mean and IQR"), _grid(ax, "y")
    ax = axes[1]
    gm = E.groupby(th.month)["target"]
    ax.bar(gm.mean().index, gm.mean().values, width=0.6, color=PAL["blue"], label="mean")
    ax.plot(gm.median().index, gm.median().values, "o", ms=6, color=PAL["orange"], mec="white", mew=1.2, label="median", zorder=3)
    ax.set_xticks(range(1, 13)), ax.set_xticklabels(list("JFMAMJJASOND")), ax.set_xlabel("month"), ax.set_title("(b) Seasonal cycle"), ax.legend(loc="upper right"), _grid(ax, "y")
    ax = axes[2]
    gd = E.groupby(th.dayofweek)["target"].mean()
    ax.bar(gd.index, gd.values, width=0.6, color=PAL["blue"])
    ax.set_xticks(range(7)), ax.set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]), ax.set_title("(c) Weekly pattern (mean)"), _grid(ax, "y")
    for a in axes[1:]:
        a.set_ylim(0, max(gm.mean().max(), gd.max()) * 1.18)
    fig.tight_layout()
    return save_fig(fig, "figure_03_pm25_temporal_pattern", "Training-only temporal structure of the target. (a) PM2.5 peaks around midnight and is lowest in the afternoon. (b) A strong seasonal cycle: autumn/winter (Oct–Dec, Mar) high, summer (Aug) low. (c) A small weekend increase. These patterns motivate the cyclical temporal features (H3).")


def fig_time_series(daily):
    fig, ax = plt.subplots(figsize=(11.5, 3.3))
    ax.plot(daily.index, daily.values, color=CONTEXT, lw=0.7, label="daily mean")
    ax.plot(daily.index, daily.rolling(30, min_periods=10, center=False).mean().values, color=PAL["blue"], lw=1.8, label="30-day trailing mean")
    ax.set_ylabel("PM2.5 (µg/m³)"), ax.set_title("PM2.5 over the training period"), ax.legend(loc="upper right", ncol=2), _grid(ax, "y")
    return save_fig(fig, "figure_S01_pm25_over_time", "Daily mean PM2.5 in the training period (2013-03 to 2016-05) with a 30-day trailing mean (past data only): pronounced seasonality and irregular multi-day pollution episodes.")


def fig_scatter_grid(E, names):
    fig, axes = plt.subplots(2, 3, figsize=(11.6, 6.6), sharey=True)
    for ax, (col, lab) in zip(axes.ravel(), names):
        x = E[col].to_numpy(float)
        m = np.isfinite(x)
        xlo, xhi = np.nanpercentile(x, [0.5, 99.5])
        hb = ax.hexbin(x[m], E["target"].to_numpy(float)[m], gridsize=(32, 26), extent=(xlo, xhi, 0, 500), cmap=SEQ_CMAP, mincnt=1, bins="log", linewidths=0)
        q = pd.qcut(x[m], 15, duplicates="drop")
        gdf = pd.DataFrame({"x": x[m], "y": E["target"].to_numpy(float)[m], "q": q}).groupby("q", observed=True).agg(x=("x", "mean"), med=("y", "median"))
        ax.plot(gdf["x"], gdf["med"], color=PAL["orange"], lw=1.9, marker="o", ms=3.6, mfc=PAL["orange"], mec="white", mew=0.8)
        from scipy import stats
        rho = stats.spearmanr(x[m], E["target"].to_numpy(float)[m])[0]
        ax.set_title(lab, fontsize=9.2, loc="left"), ax.text(0.97, 0.95, f"ρ = {rho:+.2f}", transform=ax.transAxes, ha="right", va="top", fontsize=8.8, color=INK["secondary"])
        ax.set_xlim(xlo, xhi), ax.set_ylim(0, 500), _grid(ax, "both")
    for ax in axes[:, 0]:
        ax.set_ylabel("PM2.5 at t+1 (µg/m³)")
    cbar = fig.colorbar(hb, ax=axes, shrink=0.55, pad=0.015)
    cbar.set_label("hours per cell (log10)", color=INK["secondary"]), cbar.outline.set_visible(False)
    return save_fig(fig, "figure_S02_target_vs_pollutants_meteorology", "Next-hour PM2.5 against PM10, NO2, CO, O3, temperature and dew point (training rows; y clipped at 500 µg/m³). Hexagons: density; orange: median within 15 equal-count bins; ρ = Spearman correlation. Co-pollutants move together with PM2.5; temperature alone is uninformative while the dew point is positively related.")


def fig_wind(E, wd_order):
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.7))
    ax = axes[0]
    g = E.groupby("wd")["target"]
    med = g.median().reindex(wd_order)
    ax.bar(range(len(med)), med.values, width=0.65, color=PAL["blue"])
    ax.set_xticks(range(len(med))), ax.set_xticklabels(med.index, rotation=45, fontsize=8), ax.set_ylabel("median PM2.5 at t+1 (µg/m³)")
    ax.set_title("(a) By wind direction (compass order)"), _grid(ax, "y")
    ax = axes[1]
    bins = [0, 0.5, 1, 1.5, 2, 3, 4, 6, 20]
    cut = pd.cut(E["WSPM"], bins, include_lowest=True)
    gm = E.groupby(cut, observed=True)["target"]
    ax.plot(range(len(gm.mean())), gm.mean().values, "o-", color=PAL["blue"], ms=6, mec="white", mew=1.2, label="mean")
    ax.plot(range(len(gm.median())), gm.median().values, "o-", color=PAL["orange"], ms=6, mec="white", mew=1.2, label="median")
    ax.set_xticks(range(len(gm.mean()))), ax.set_xticklabels([f"{iv.left:g}–{iv.right:g}" for iv in gm.mean().index], rotation=30, fontsize=8)
    ax.set_xlabel("wind speed at t (m/s)"), ax.set_title("(b) By wind speed"), ax.legend(), _grid(ax, "y")
    fig.tight_layout()
    return save_fig(fig, "figure_S03_wind_effects", "Training-only wind effects. (a) Winds from the north / north-west (N, NNW, NW) are associated with much cleaner next-hour air than southerly and easterly winds. (b) PM2.5 falls steeply with wind speed (ventilation). These are the reasons for encoding wind direction without an artificial ordering (one-hot) and for the wind × PM interactions.")


def fig_missingness(runs, limits, eligible_counts, n_rows):
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.5))
    ax = axes[0]
    ax.hist(runs, bins=np.unique(np.r_[np.arange(1, 25), 48, 96, 400]), color=PAL["blue"], edgecolor="white", linewidth=0.5)
    ax.set_yscale("log"), ax.set_xscale("log"), ax.set_xlabel("length of a PM2.5 gap (hours)"), ax.set_ylabel("number of gaps (log)")
    ax.set_title("(a) Gaps in the training PM2.5 series"), _grid(ax, "both")
    ax = axes[1]
    ax.plot(limits, 100 * np.asarray(eligible_counts) / n_rows, "o-", color=PAL["blue"], ms=6.5, mec="white", mew=1.3)
    k = list(limits).index(C.LOCF_LIMIT)
    ax.plot([limits[k]], [100 * eligible_counts[k] / n_rows], "o", ms=9, color=PAL["orange"], mec="white", mew=1.5, zorder=4)
    ax.annotate(f"chosen: {C.LOCF_LIMIT} h\n{100 * eligible_counts[k] / n_rows:.1f} %", (limits[k], 100 * eligible_counts[k] / n_rows), xytext=(12, -34), textcoords="offset points", fontsize=8.5, color=INK["secondary"])
    ax.set_xlabel("forward-fill limit for the PM2.5 history (hours)"), ax.set_ylabel("usable training rows (%)"), ax.set_title("(b) Usable rows vs forward-fill limit"), _grid(ax, "both")
    fig.tight_layout()
    return save_fig(fig, "figure_S04_missing_values_policy", "Missing-value policy (training rows). (a) Most PM2.5 gaps are 1–3 hours long, a few last days. (b) Causally forward-filling the PM2.5 *history* over gaps of up to 3 hours keeps 92.7 % of the rows; longer limits add < 1.5 points. The target is never filled; rows with unusable history are excluded identically for every model.")


def fig_corr_heatmaps(Dc):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.6))
    for ax, method in zip(axes, ("pearson", "spearman")):
        Cm = Dc.corr(method=method)
        im = ax.imshow(Cm.values, cmap=DIV_CMAP, vmin=-1, vmax=1)
        ax.set_xticks(range(len(Cm))), ax.set_yticks(range(len(Cm)))
        ax.set_xticklabels(Cm.columns, rotation=90, fontsize=8), ax.set_yticklabels(Cm.columns, fontsize=8)
        for i in range(len(Cm)):
            for j in range(len(Cm)):
                v = Cm.values[i, j]
                ax.text(j, i, f"{v:.2f}".replace("0.", ".").replace("-.", "−."), ha="center", va="center", fontsize=5.6, color="white" if abs(v) > 0.65 else INK["secondary"])
        ax.set_title(f"{method.capitalize()} correlation"), ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    cb = fig.colorbar(im, ax=axes, shrink=0.7, pad=0.012)
    cb.set_label("correlation", color=INK["secondary"]), cb.outline.set_visible(False)
    return save_fig(fig, "figure_04_correlation_heatmap_train", "Training-only Pearson (left) and Spearman (right) correlation matrices of the raw measurements and the target (next-hour PM2.5). PM2.5(t), PM10 and CO are almost interchangeable with the target; TEMP and DEWP are strongly mutually correlated.")


def fig_target_ranking(assoc):
    raw = assoc[assoc["set"] == "raw"].sort_values("abs_spearman")
    eng = assoc[assoc["set"] == "engineered"].sort_values("abs_spearman")
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 9.4), gridspec_kw={"width_ratios": [0.85, 1.15]})
    for ax, tab, title in [(axes[0], raw, "(a) Raw measurements (Set A)"), (axes[1], eng, "(b) Engineered features")]:
        pos = np.arange(len(tab))
        ax.barh(pos - 0.2, tab["abs_pearson"], height=0.36, color=PAL["blue"], label="|Pearson r|")
        ax.barh(pos + 0.2, tab["abs_spearman"], height=0.36, color=PAL["orange"], label="|Spearman ρ|")
        ax.set_yticks(pos), ax.set_yticklabels(tab["feature"], fontsize=8), ax.set_xlim(0, 1.0)
        ax.set_xlabel("absolute correlation with next-hour PM2.5 (training rows)"), ax.set_title(title), _grid(ax, "x")
    axes[0].legend(loc="lower right")
    fig.tight_layout()
    return save_fig(fig, "figure_05_feature_target_correlation_ranking", "Absolute Pearson and Spearman correlation of every candidate feature with the target, computed on the training rows only. The current PM2.5 and the PM2.5 history dominate (r > 0.9); among the raw environment variables PM10, CO and NO2 are strongest; temperature, pressure, rain and the wind-direction dummies are weak marginally.")


def fig_redundancy(D, vif):
    R = D.corr().abs().fillna(0)
    link = hierarchy.linkage(squareform(1 - R.values, checks=False), "average")
    order = hierarchy.leaves_list(link)
    Rc = D.corr().fillna(0).iloc[order, order]
    fig = plt.figure(figsize=(14.6, 7.2))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.6, 0.04, 0.55, 0.9], wspace=0.12)
    ax, cax, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[3])
    im = ax.imshow(Rc.values, cmap=DIV_CMAP, vmin=-1, vmax=1)
    ax.set_xticks(range(len(Rc))), ax.set_yticks(range(len(Rc)))
    ax.set_xticklabels(Rc.columns, rotation=90, fontsize=5.5), ax.set_yticklabels(Rc.columns, fontsize=5.5)
    ax.set_title("(a) Set C design: hierarchically clustered correlations (training)"), ax.grid(False)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    cb = fig.colorbar(im, cax=cax)
    cb.outline.set_visible(False)
    v = vif.replace(np.inf, np.nan).sort_values().tail(22)
    ax2.barh(np.arange(len(v)), v.values, height=0.6, color=PAL["blue"])
    ax2.axvline(C.PROP_VIF, color=INK["secondary"], lw=1.0)
    ax2.text(C.PROP_VIF * 1.03, 0.2, f"VIF = {C.PROP_VIF:g}", fontsize=8, color=INK["secondary"])
    ax2.set_yticks(np.arange(len(v))), ax2.set_yticklabels(v.index, fontsize=7.5), ax2.set_xscale("log"), ax2.set_xlabel("VIF (log scale)")
    ax2.set_title("(b) Highest variance-inflation factors"), _grid(ax2, "x")
    return save_fig(fig, "figure_S05_feature_redundancy", "Feature-feature redundancy in the training design of Set C.")


def fig_acf(acf_vals, pacf_vals):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    for ax, v, title in [(axes[0], acf_vals, "(a) Autocorrelation of PM2.5 (training)"), (axes[1], pacf_vals, "(b) Partial autocorrelation")]:
        lags = np.arange(1, len(v))
        ax.vlines(lags, 0, v[1:], color=PAL["blue"], lw=1.0)
        ax.axhline(0, color=INK["axis"], lw=0.8)
        for L in (24, 168):
            if L < len(v):
                ax.axvline(L, color=CONTEXT, lw=0.8)
                ax.text(L + 1, ax.get_ylim()[1] * 0.92, f"{L} h", fontsize=8, color=INK["secondary"])
        ax.set_xlabel("lag (hours)"), ax.set_title(title), _grid(ax, "y")
    fig.tight_layout()
    return save_fig(fig, "figure_S06_autocorrelation", "Autocorrelation and partial autocorrelation of hourly PM2.5 in the training rows. The first lag dominates (> 0.96); the partial autocorrelation shows the remaining short-memory structure and weak daily (24 h) / weekly (168 h) echoes that motivate the lag set {1, 3, 6, 12, 24, 48, 168}.")


# ====================================================================== modelling figures
def fig_tuning_curve(cvres, family, best_alpha, best_l1, ref_mae=None):
    """cvres: cv_results_ DataFrame of the final family (param_model__alpha [, param_model__l1_ratio], per-fold test_mae)."""
    fig, ax = plt.subplots(figsize=(7.4, 4.1))
    fold_cols = [c for c in cvres.columns if c.startswith("split") and c.endswith("_test_mae")]
    S = -cvres[fold_cols].to_numpy()
    mean, se = S.mean(1), S.std(1, ddof=1) / np.sqrt(S.shape[1])
    al = cvres["param_model__alpha"].astype(float).to_numpy()
    if family == "ElasticNet":
        l1 = cvres["param_model__l1_ratio"].astype(float).to_numpy()
        levels = sorted(set(l1))
        cols = SEQ_CMAP(np.linspace(0.3, 1.0, len(levels)))
        for c, v in zip(cols, levels):
            s = l1 == v
            ax.plot(al[s], mean[s], color=c, lw=1.6, label=f"l1_ratio {v:g}")
        ax.legend(loc="upper left", fontsize=8, ncol=2)
        by = float(mean[(al == best_alpha) & (l1 == best_l1)][0])
    else:
        ax.fill_between(al, mean - se, mean + se, color=PAL["blue"], alpha=0.14, lw=0)
        ax.plot(al, mean, color=PAL["blue"], lw=1.8)
        by = float(mean[np.isclose(al, best_alpha)][0])
    ax.plot([best_alpha], [by], "o", ms=8, color=PAL["orange"], mec="white", mew=1.6, zorder=4)
    ax.annotate(f"selected α = {best_alpha:.3g}" + (f", l1_ratio = {best_l1:g}" if family == "ElasticNet" else ""), (best_alpha, by), xytext=(10, 14), textcoords="offset points", fontsize=8.5, color=INK["secondary"])
    ax.set_xscale("log"), ax.set_xlabel("α (regularisation strength)"), ax.set_ylabel("CV MAE (µg/m³)"), ax.set_title(f"CV tuning curve — {family} on Set C"), _grid(ax, "y")
    lo, hi = float(mean.min()), float(mean.min()) + 0.6 * (float(np.median(mean)) - float(mean.min()) + 1.0)
    ax.set_ylim(lo - 0.15, min(float(mean.max()), lo + 6.0))
    return save_fig(fig, "figure_06_tuning_curve", f"Mean 5-fold time-series CV MAE of the selected family ({family}) against the regularisation strength α" + (" for each l1_ratio" if family == "ElasticNet" else " (band: ±1 standard error)") + ". Orange: selected configuration. The curve is flat over a wide range of α: with ~21,000 rows and strongly informative predictors the penalty mainly stabilises correlated coefficients rather than changing accuracy.")


def _bars(ax, y, a, b, la, lb, ca=PAL["blue"], cb=PAL["orange"], fmt="{:.2f}", off=0.19, h=0.34, label_b=True):
    ax.barh(y + off, a, height=h, color=ca, label=la)
    ax.barh(y - off, b, height=h, color=cb, label=lb)
    if label_b:
        for yy, v in zip(y - off, b):
            ax.text(v + 0.012 * ax.get_xlim()[1], yy, fmt.format(v), va="center", fontsize=7.8, color=INK["secondary"])


def fig_model_comparison(t, final_label, n_baselines):
    """t: numeric Table 3 (Method, CV_MAE, CV_RMSE, Test_MAE, Test_RMSE, Test_R2). Top row: full scale. Bottom row: zoom without the training mean."""
    n = len(t)
    yy = np.arange(n)[::-1].astype(float)
    fin = [i for i, m in enumerate(t["Method"]) if m.startswith(final_label)][0]
    fig, axes = plt.subplots(2, 3, figsize=(14.2, 10.4), sharey=True, gridspec_kw={"width_ratios": [1, 1, 0.8], "wspace": 0.08, "hspace": 0.36})
    for ax, (cv, te, title, xl) in zip(axes[0, :2], [("CV_MAE", "Test_MAE", "(a) MAE", "MAE (µg/m³)"), ("CV_RMSE", "Test_RMSE", "(b) RMSE", "RMSE (µg/m³)")]):
        ax.axhspan(yy[fin] - 0.5, yy[fin] + 0.5, color=INK["mid"], zorder=0)
        ax.set_xlim(0, max(t[cv].max(), t[te].max()) * 1.14)
        _bars(ax, yy, t[cv].to_numpy(), t[te].to_numpy(), "5-fold CV (training)", "held-out test", fmt="{:.1f}")
        ax.set_xlabel(xl), ax.set_title(title), _grid(ax, "x")
    ax = axes[0, 2]
    ax.axhspan(yy[fin] - 0.5, yy[fin] + 0.5, color=INK["mid"], zorder=0)
    ax.barh(yy, t["Test_R2"], height=0.5, color=PAL["orange"])
    for y_, v in zip(yy, t["Test_R2"]):
        ax.text(max(v, 0) + 0.015, y_, f"{v:.3f}", va="center", ha="left", fontsize=7.8, color=INK["secondary"])
    ax.axvline(0, color=INK["secondary"], lw=1.0), ax.set_xlabel("R² (test)"), ax.set_title("(c) Test R²"), _grid(ax, "x"), ax.set_xlim(-0.02, 1.15)
    sub = np.arange(n) >= 1                                           # zoom: omit the training mean
    for ax, (cv, te, title, xl) in zip(axes[1, :2], [("CV_MAE", "Test_MAE", "(d) MAE — zoom (training mean omitted)", "MAE (µg/m³)"), ("CV_RMSE", "Test_RMSE", "(e) RMSE — zoom (training mean omitted)", "RMSE (µg/m³)")]):
        ax.axhspan(yy[fin] - 0.5, yy[fin] + 0.5, color=INK["mid"], zorder=0)
        lo, hi = min(t[cv][sub].min(), t[te][sub].min()), max(t[cv][sub].max(), t[te][sub].max())
        pad = 0.10 * (hi - lo)
        ax.hlines(yy[sub], t[cv][sub], t[te][sub], color=INK["axis"], lw=1.6, zorder=1)
        ax.plot(t[cv][sub], yy[sub], "o", color=PAL["blue"], ms=7.5, mec="white", mew=1.2, label="5-fold CV (training)", zorder=3)
        ax.plot(t[te][sub], yy[sub], "o", color=PAL["orange"], ms=7.5, mec="white", mew=1.2, label="held-out test", zorder=3)
        for y_, v in zip(yy[sub], t[te][sub]):
            ax.text(v, y_ + 0.30, f"{v:.2f}", ha="center", fontsize=7.6, color=INK["secondary"])
        ax.set_xlim(lo - pad, hi + pad), ax.set_xlabel(xl), ax.set_title(title), _grid(ax, "both")
    ax = axes[1, 2]
    ax.axhspan(yy[fin] - 0.5, yy[fin] + 0.5, color=INK["mid"], zorder=0)
    r2 = t["Test_R2"][sub]
    ax.plot(r2, yy[sub], "o", color=PAL["orange"], ms=7.5, mec="white", mew=1.2, zorder=3)
    for y_, v in zip(yy[sub], r2):
        ax.text(v, y_ + 0.30, f"{v:.4f}", ha="center", fontsize=7.6, color=INK["secondary"])
    pad = 0.15 * (r2.max() - r2.min())
    ax.set_xlim(r2.min() - pad, r2.max() + pad), ax.set_xlabel("R² (test)"), ax.set_title("(f) Test R² — zoom"), _grid(ax, "both")
    for a in (axes[0, 0], axes[1, 0]):
        a.set_yticks(yy), a.set_yticklabels(t["Method"].tolist(), fontsize=8.5)
        a.get_yticklabels()[fin].set_fontweight("bold")
    for a in axes.ravel():
        a.axhline(yy[n_baselines - 1] - 0.5, color=INK["axis"], lw=0.9)
    axes[0, 0].legend(loc="lower right", fontsize=8), axes[1, 0].legend(loc="lower right", fontsize=8)
    return save_fig(fig, "figure_07_model_comparison", "Baselines (top rows of each panel) and the four linear models on Set C (bottom); the grey band marks the selected final model. Top row: full scale; bottom row: zoom without the training mean (dumbbells join the cross-validated error on the training period and the single test evaluation).")


def fig_ablation(t):
    n = len(t)
    yv = np.arange(n)[::-1].astype(float)
    c_row = [i for i, v in enumerate(t["Variant"]) if v.startswith("Set C —")][0]
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 6.6), sharey=True, gridspec_kw={"width_ratios": [1.1, 1], "wspace": 0.06})
    ax = axes[0]
    ax.axhspan(yv[c_row] - 0.5, yv[c_row] + 0.5, color=INK["mid"], zorder=0)
    ax.set_xlim(0, max(t["CV_MAE"].max(), t["Test_MAE"].max()) * 1.16)
    ax.barh(yv + 0.19, t["CV_MAE"], height=0.34, xerr=t["CV_MAE_std"] / np.sqrt(C.N_SPLITS), color=PAL["blue"], error_kw=dict(ecolor=INK["muted"], lw=1.0, capsize=0), label="5-fold CV (±1 SE)")
    ax.barh(yv - 0.19, t["Test_MAE"], height=0.34, color=PAL["orange"], label="held-out test")
    for y_, v in zip(yv - 0.19, t["Test_MAE"]):
        ax.text(v + 0.15, y_, f"{v:.2f}", va="center", fontsize=7.8, color=INK["secondary"])
    ax.set_yticks(yv), ax.set_yticklabels(t["Variant"].tolist(), fontsize=8.3), ax.get_yticklabels()[c_row].set_fontweight("bold")
    ax.set_xlabel("MAE (µg/m³)"), ax.set_title("(a) MAE by feature set"), ax.legend(loc="upper right", fontsize=8), _grid(ax, "x")
    ax = axes[1]
    ax.axhspan(yv[c_row] - 0.5, yv[c_row] + 0.5, color=INK["mid"], zorder=0)
    lo, hi = -1.5, 5.0
    for off, col, lab, key in [(0.19, PAL["blue"], "CV", "delta_CV_MAE_vs_C_pct"), (-0.19, PAL["orange"], "test", "delta_test_MAE_vs_C_pct")]:
        vals = t[key].to_numpy(float)
        ax.barh(yv + off, np.clip(vals, lo, hi), height=0.34, color=col, label=lab)
        for y_, v in zip(yv + off, vals):
            if v > hi:
                ax.text(hi - 0.08, y_, f"{v:+.0f} %  →", va="center", ha="right", fontsize=7.6, color="white", fontweight="bold")
    ax.axvline(0, color=INK["secondary"], lw=1.0)
    ax.set_xlim(lo, hi), ax.set_xlabel("MAE change relative to Set C (%)   [> 0: worse than C]   (axis clipped at +5 %)"), ax.set_title("(b) Effect of removing / replacing a group (zoom)")
    ax.legend(loc="center right", fontsize=8), _grid(ax, "x")
    return save_fig(fig, "figure_08_feature_ablation", "Controlled feature ablation with one linear family and identical folds / search procedure.")


def fig_actual_vs_predicted(meta, y, p, e_top):
    fig = plt.figure(figsize=(12.4, 8.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.85], width_ratios=[1, 1.45], hspace=0.38, wspace=0.22)
    ax = fig.add_subplot(gs[0, 0])
    hb = ax.hexbin(y, p, gridsize=36, extent=(0, 500, -20, 500), cmap=SEQ_CMAP, mincnt=1, bins="log", linewidths=0)
    ax.plot([0, 500], [0, 500], color=INK["secondary"], lw=1.1)
    ax.set_xlabel("actual PM2.5(t+1) (µg/m³)"), ax.set_ylabel("predicted (µg/m³)"), ax.set_title("(a) Actual vs predicted (axes clipped)"), _grid(ax, "both")
    cb = fig.colorbar(hb, ax=ax, pad=0.02, shrink=0.85)
    cb.set_label("hours per cell (log10)", color=INK["secondary"]), cb.outline.set_visible(False)
    ax = fig.add_subplot(gs[0, 1])
    dd = pd.DataFrame({"t": pd.DatetimeIndex(meta["target_time"]).floor("D"), "a": y, "p": p}).groupby("t").mean()
    ax.plot(dd.index, dd["a"], color=CONTEXT, lw=1.3, label="actual"), ax.plot(dd.index, dd["p"], color=PAL["blue"], lw=1.3, label="predicted")
    ax.set_ylabel("daily mean (µg/m³)"), ax.set_title("(b) Daily means over the test period"), ax.legend(loc="upper right", ncol=2), _grid(ax, "y")
    _date_axis(ax)
    ax = fig.add_subplot(gs[1, :])
    t0 = pd.Timestamp(e_top["time"]) - pd.Timedelta(days=4)
    tt = pd.DatetimeIndex(meta["target_time"])
    w = (tt >= t0) & (tt < t0 + pd.Timedelta(days=10))
    ax.plot(tt[w], np.asarray(y)[w], color=CONTEXT, lw=1.5, label="actual"), ax.plot(tt[w], np.asarray(p)[w], color=PAL["blue"], lw=1.4, label="predicted")
    ax.plot([e_top["time"]], [e_top["actual"]], "o", ms=8, color=PAL["orange"], mec="white", mew=1.5, zorder=4)
    ax.annotate("largest error", (e_top["time"], e_top["actual"]), xytext=(-11, 0), textcoords="offset points", ha="right", va="center", fontsize=8.5, color=INK["secondary"])
    ax.set_ylabel("PM2.5 (µg/m³)"), ax.set_title("(c) Ten-day window around the largest error"), ax.legend(loc="upper right", ncol=2), _grid(ax, "y")
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%d %b"))
    return save_fig(fig, "figure_09_actual_vs_predicted", "Final model on the held-out test period. (a) Actual vs predicted hourly values. (b) Daily means. (c) A ten-day window of hourly values around the largest error: the forecast tracks the level closely but reacts to abrupt episodes one hour late.")


def fig_residuals(meta, y, p, acf_vals):
    e = np.asarray(y, float) - np.asarray(p, float)
    tt = pd.DatetimeIndex(meta["target_time"])
    fig, axes = plt.subplots(2, 3, figsize=(14.2, 8.0))
    ax = axes[0, 0]
    hb = ax.hexbin(p, e, gridsize=(40, 30), extent=(-10, 450, -150, 200), cmap=SEQ_CMAP, mincnt=1, bins="log", linewidths=0)
    qb = pd.qcut(p, 20, duplicates="drop")
    g = pd.DataFrame({"x": p, "e": e, "q": qb}).groupby("q", observed=True).agg(x=("x", "mean"), m=("e", "mean"))
    ax.plot(g["x"], g["m"], color=PAL["orange"], lw=1.9, marker="o", ms=4, mec="white", mew=0.8), ax.axhline(0, color=INK["secondary"], lw=1.0)
    ax.set_xlabel("predicted (µg/m³)"), ax.set_ylabel("residual = actual − predicted"), ax.set_title("(a) Residual vs predicted"), _grid(ax, "both")
    ax = axes[0, 1]
    ax.plot(tt, e, color=CONTEXT, lw=0.35, rasterized=True)
    dm = pd.Series(e, index=tt).resample("D").mean()
    ax.plot(dm.index, dm.values, color=PAL["blue"], lw=1.4, label="daily mean"), ax.axhline(0, color=INK["secondary"], lw=1.0)
    ax.set_ylabel("residual (µg/m³)"), ax.set_title("(b) Residual over time"), ax.legend(loc="upper right"), _grid(ax, "y"), _date_axis(ax)
    ax = axes[0, 2]
    lo, hi = np.percentile(e, [0.2, 99.8])
    ax.hist(e, bins=np.linspace(lo, hi, 90), color=PAL["blue"], edgecolor="white", linewidth=0.3, density=True)
    xs = np.linspace(lo, hi, 300)
    from scipy import stats
    ax.plot(xs, stats.norm.pdf(xs, e.mean(), e.std()), color=PAL["orange"], lw=1.6, label="normal, same mean/std")
    ax.set_yscale("log"), ax.set_ylim(1e-5, None), ax.set_xlabel("residual (µg/m³)"), ax.set_ylabel("density (log)"), ax.set_title("(c) Residual distribution"), ax.legend(loc="upper left"), _grid(ax, "both")
    ax = axes[1, 0]
    hb2 = ax.hexbin(y, p, gridsize=34, extent=(0, 450, -10, 450), cmap=SEQ_CMAP, mincnt=1, bins="log", linewidths=0)
    ax.plot([0, 450], [0, 450], color=INK["secondary"], lw=1.0)
    ax.set_xlabel("actual (µg/m³)"), ax.set_ylabel("predicted (µg/m³)"), ax.set_title("(d) Actual vs predicted"), _grid(ax, "both")
    ax = axes[1, 1]
    ae = pd.Series(np.abs(e), index=tt)
    ax.plot(ae.index, ae.values, color=CONTEXT, lw=0.35, rasterized=True)
    d = ae.resample("D").mean()
    ax.plot(d.index, d.values, color=PAL["blue"], lw=1.4, label="daily MAE")
    ax.set_ylabel("absolute error (µg/m³)"), ax.set_title("(e) Absolute error over time"), ax.legend(loc="upper right"), _grid(ax, "y"), _date_axis(ax)
    ax = axes[1, 2]
    lags = np.arange(1, len(acf_vals))
    ax.vlines(lags, 0, acf_vals[1:], color=PAL["blue"], lw=1.0), ax.axhline(0, color=INK["axis"], lw=0.8)
    band = 1.96 / np.sqrt(len(e))
    ax.axhspan(-band, band, color=INK["grid"], alpha=0.9, lw=0)
    ax.set_xlabel("lag (hours)"), ax.set_ylabel("autocorrelation"), ax.set_title("(f) Residual autocorrelation"), _grid(ax, "y")
    fig.tight_layout()
    return save_fig(fig, "figure_10_residual_analysis", "Residual diagnostics on the test period. (a) The residual spread grows with the predicted level (heteroscedasticity) and large positive residuals appear at all levels. (b) No drift over time, but larger swings in the pollution season. (c) Heavy-tailed, non-normal residuals (log scale). (d) Actual vs predicted. (e) Absolute error follows the pollution episodes. (f) Residuals remain autocorrelated, so classical OLS standard errors would be too small (HAC errors are used).")


def fig_coefficients(tab, model_label):
    """tab: coefficient table (feature, group, coef, boot_lo/hi, ols, hac_lo/hi)."""
    panels = [("(a) PM2.5 history", ["current PM2.5", "lag", "rolling"]),
              ("(b) Pollutants, meteorology, derived differences, interactions", ["pollutants", "meteorology / wind", "derived differences", "interaction"]),
              ("(c) Calendar and wind direction", ["temporal", "wind"])]
    tab = tab.copy()
    tab["panel_group"] = np.where(tab["feature"].str.startswith("wd_"), "wind", tab["group"])
    tab.loc[tab["feature"].str.startswith("wd_"), "panel_group"] = "wind"
    sizes = [int(tab["panel_group"].isin(g).sum()) for _, g in panels]
    fig, axes = plt.subplots(3, 1, figsize=(8.8, 0.26 * sum(sizes) + 4.0), gridspec_kw={"height_ratios": sizes, "hspace": 0.28})
    for ax, (title, groups), n in zip(axes, panels, sizes):
        part = tab[tab["panel_group"].isin(groups)].copy()
        part["_o"] = part["panel_group"].map({g: i for i, g in enumerate(groups)})
        part = part.sort_values(["_o", "coef"], key=lambda s: s if s.name == "_o" else s.abs(), ascending=[True, False]).reset_index(drop=True)
        yy = np.arange(len(part))[::-1].astype(float)
        c_, lo, hi = part["coef"].to_numpy(), part["boot_lo"].to_numpy(), part["boot_hi"].to_numpy()
        o_, olo, ohi = part["ols"].to_numpy(), part["hac_lo"].to_numpy(), part["hac_hi"].to_numpy()
        ax.errorbar(c_, yy + 0.17, xerr=[c_ - lo, hi - c_], fmt="o", color=PAL["blue"], ms=5.2, lw=1.4, capsize=0, mec="white", mew=0.7, label=f"{model_label} (final model), 95 % block-bootstrap CI")
        ax.errorbar(o_, yy - 0.17, xerr=[o_ - olo, ohi - o_], fmt="s", color=PAL["orange"], ms=4.4, lw=1.4, capsize=0, mec="white", mew=0.7, label="OLS companion, 95 % HAC CI")
        ax.axvline(0, color=INK["secondary"], lw=1.0)
        ax.set_yticks(yy), ax.set_yticklabels(part["feature"].tolist(), fontsize=8), ax.set_ylim(-0.7, len(part) - 0.3)
        ax.set_title(title), _grid(ax, "both")
    axes[-1].set_xlabel("effect on next-hour PM2.5 (µg/m³ per 1 SD of the feature)")
    axes[0].legend(loc="lower right")
    return save_fig(fig, "figure_11_coefficients_confidence_intervals", "Standardised coefficients of the final model (effect on the forecast in µg/m³ per one standard deviation of the feature) with 95 % intervals. Blue: penalised predictive model with moving-block-bootstrap intervals (one-week blocks). Orange: OLS companion with HAC (Newey–West) intervals. The three panels have different x-scales. Intervals that exclude zero indicate effects distinguishable from zero; many weather and calendar effects are small relative to the PM2.5 history.")


def fig_slices(st, models):
    """st: long slice table (group, slice, n, model, MAE, RMSE). models: ordered model names."""
    rows = st[["group", "slice", "n"]].drop_duplicates().reset_index(drop=True)
    yy = np.arange(len(rows))[::-1].astype(float)
    cols = {models[0]: PAL["blue"], models[1]: PAL["orange"], models[2]: CONTEXT}
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 0.36 * len(rows) + 2.2), sharey=True, gridspec_kw={"wspace": 0.05})
    h = 0.26
    for ax, metric in zip(axes, ("MAE", "RMSE")):
        for k, m in enumerate(models):
            vals = [float(st[(st["slice"] == s) & (st["group"] == g) & (st["model"] == m)][metric].iloc[0]) for g, s in zip(rows["group"], rows["slice"])]
            ax.barh(yy + (1 - k) * h, vals, height=h, color=cols[m], label=m)
        ax.set_xlabel(f"{metric} (µg/m³)"), ax.set_title(metric), _grid(ax, "x")
        prev = None
        for g, y_ in zip(rows["group"], yy):
            if g != prev and prev is not None:
                ax.axhline(y_ + 0.5, color=INK["axis"], lw=0.8)
            prev = g
    axes[0].set_yticks(yy), axes[0].set_yticklabels([f"{s}  (n={n})" for s, n in zip(rows["slice"], rows["n"])], fontsize=8)
    axes[0].legend(loc="lower right", fontsize=8)
    return save_fig(fig, "figure_12_slice_performance", "Test-set MAE (left) and RMSE (right) of the final model (Set C), the same linear family on the raw features (Set A) and persistence, by slice. Slice cut-offs come from the training data or the calendar. Errors rise strongly in high-pollution hours, rapid changes and winter; the engineered features help most where the load is changing.")


def fig_threshold(tt):
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    cols = [c for c in tt.columns if c.startswith("within")]
    xs = [float(c.split()[1]) for c in cols]
    for i, (_, r) in enumerate(tt.iterrows()):
        ax.plot(xs, [100 * r[c] for c in cols], "o-", ms=5.5, mec="white", mew=1.0, label=r["model"], color=list(PAL.values())[i % 8])
    ax.axvline(C.DECISION_THRESHOLD, color=INK["secondary"], lw=1.0)
    ax.text(C.DECISION_THRESHOLD + 0.6, 6, f"proposed τ = {C.DECISION_THRESHOLD:g}", fontsize=8.5, color=INK["secondary"])
    ax.set_xlabel("error tolerance τ (µg/m³)"), ax.set_ylabel("share of test predictions with |error| ≤ τ (%)"), ax.set_title("Decision-threshold sensitivity"), ax.legend(loc="lower right", fontsize=8), _grid(ax, "both")
    return save_fig(fig, "figure_S07_decision_threshold_sensitivity", "Share of test-set predictions within an error tolerance τ for several candidate tolerances. The decision threshold is a configurable, non-authoritative operating choice (proposed default marked); the figure shows how the conclusion depends on it.")
