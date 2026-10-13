"""Data loading, schema inspection, timeline validation, target construction, chronological split and the test-set vault."""
import datetime as dt
import hashlib
import io
import json
import urllib.request
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C


# ====================================================================== acquisition
def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ensure_raw_data(path=None, download=True):
    """Return the path of the raw CSV. If it is missing it is downloaded from the documented UCI archive (id 501) and cached."""
    path = Path(path) if path else C.RAW_DIR / C.RAW_FILENAME
    if path.exists():
        return path
    if not download:
        raise FileNotFoundError(f"{path} not found and download disabled")
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(C.UCI_ZIP_URL, timeout=180) as r:                      # outer archive of the UCI page
        outer = zipfile.ZipFile(io.BytesIO(r.read()))
    inner = zipfile.ZipFile(io.BytesIO(outer.read(C.UCI_NESTED_ZIP)))                  # contains the 12 station files
    member = next(n for n in inner.namelist() if n.endswith(C.RAW_FILENAME))
    path.write_bytes(inner.read(member))
    return path


def load_raw(path=None):
    """Read the Aotizhongxin CSV exactly as delivered and verify it against the expected size / fingerprint."""
    p = ensure_raw_data(path)
    df = pd.read_csv(p)
    assert len(df) == C.EXPECTED_ROWS, f"unexpected number of rows: {len(df)}"
    if sha256_file(p) != C.EXPECTED_SHA256:
        warnings.warn("raw file fingerprint differs from the one used to develop the experiment (row count matches)")
    return df


def provenance(path=None):
    p = ensure_raw_data(path)
    return {**C.SOURCE_INFO, "file": p.name, "sha256": sha256_file(p), "rows": C.EXPECTED_ROWS, "station": C.STATION}


# ====================================================================== inspection
def build_datetime(df):
    """datetime from year/month/day/hour; chronological sort. Returns (frame, was_sorted_in_file)."""
    out = df.copy()
    out["datetime"] = pd.to_datetime(out[["year", "month", "day", "hour"]])
    was_sorted = bool(out["datetime"].is_monotonic_increasing)
    out = out.sort_values("datetime", kind="stable").reset_index(drop=True)
    return out, was_sorted


def validate_timeline(df, freq="h"):
    """Chronological order, duplicated timestamps, regular hourly grid, missing timestamps — asserted, not assumed."""
    ts = pd.DatetimeIndex(df["datetime"])
    steps = pd.Series(ts[1:] - ts[:-1])
    grid = pd.date_range(ts[0], ts[-1], freq=freq)
    info = {"monotonic_increasing": bool(ts.is_monotonic_increasing), "duplicated_timestamps": int(ts.duplicated().sum()),
            "sampling_intervals": {str(k): int(v) for k, v in steps.value_counts().items()},
            "start": str(ts[0]), "end": str(ts[-1]), "expected_hourly_stamps": int(len(grid)), "observed_stamps": int(len(ts)),
            "missing_timestamps": int(len(grid.difference(ts))), "span_days": round((ts[-1] - ts[0]).total_seconds() / 86400, 2)}
    assert info["monotonic_increasing"] and info["duplicated_timestamps"] == 0
    assert info["missing_timestamps"] == 0 and list(info["sampling_intervals"]) == ["0 days 01:00:00"], "timeline is not a regular hourly grid"
    return info


def inspect_raw(df):
    """Schema / quality report (rows, columns, dtypes, missing values, duplicates, categorical levels, target statistics)."""
    dtypes = pd.DataFrame({"column": df.columns, "dtype": [str(t) for t in df.dtypes], "non_null": df.notna().sum().to_numpy(),
                           "missing": df.isna().sum().to_numpy()})
    dtypes["missing_pct"] = 100 * dtypes["missing"] / len(df)
    tgt = df[C.RAW_TARGET]
    return {"rows": len(df), "columns": df.shape[1], "column_names": list(df.columns), "dtypes": dtypes,
            "duplicate_rows": int(df.duplicated().sum()), "duplicate_rows_ignoring_No": int(df.drop(columns=[c for c in ["No"] if c in df]).duplicated().sum()),
            "wd_levels": df[C.WIND_DIR].value_counts(dropna=False).to_dict(),
            "stations": sorted(df["station"].unique().tolist()) if "station" in df else [],
            "target_stats": {"count": int(tgt.notna().sum()), "missing": int(tgt.isna().sum()), "mean": float(tgt.mean()), "std": float(tgt.std()),
                             "min": float(tgt.min()), "q25": float(tgt.quantile(.25)), "median": float(tgt.median()), "q75": float(tgt.quantile(.75)),
                             "max": float(tgt.max()), "skew": float(tgt.skew())}}


def drop_identifier_columns(df):
    """`No` is a row counter and `station` is constant for this file: neither is a predictor."""
    ids = [c for c in ("No", "station") if c in df.columns]
    const = {c: df[c].unique().tolist() for c in ids if df[c].nunique() == 1}
    return df.drop(columns=ids), {"dropped": ids, "constant_values": const}


# ====================================================================== task construction
def make_target(df):
    """target(t) = PM2.5(t+1). The last hour has no t+1 and is removed. The target is built from the RAW (never filled) series."""
    out = df.copy()
    out[C.TARGET] = out[C.RAW_TARGET].shift(-1)
    out = out.iloc[:-1].reset_index(drop=True)
    pm = out[C.RAW_TARGET].to_numpy()
    assert np.array_equal(out[C.TARGET].to_numpy()[:-1], pm[1:], equal_nan=True), "target is not PM2.5(t+1)"
    assert C.TARGET not in (C.RAW_TARGET, C.CURRENT)
    return out


# ====================================================================== split + vault
class HeldOutTest:
    """Keeps the test rows out of reach until the design is frozen. Only meta-data is public; unlock() logs every access."""

    def __init__(self, frame):
        self._frame = frame.reset_index(drop=True).copy()
        self.n_rows, self.start, self.end = len(frame), frame["datetime"].iloc[0], frame["datetime"].iloc[-1]
        self.unlock_count = 0

    def __repr__(self):
        return f"HeldOutTest(rows={self.n_rows}, {self.start} -> {self.end}, unlocked_in_this_session={self.unlock_count})"

    def unlock(self, design_sha256, reason, log_path=None):
        self.unlock_count += 1
        log = Path(log_path) if log_path else C.LOGS / "test_access_log.json"
        log.parent.mkdir(parents=True, exist_ok=True)
        entries = json.loads(log.read_text()) if log.exists() else []
        same = all(e["design_sha256"] == design_sha256 for e in entries)
        entries.append({"time_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "reason": reason,
                        "design_sha256": design_sha256, "n_test_rows": self.n_rows})
        log.write_text(json.dumps(entries, indent=2))
        print(f"TEST SET UNLOCKED (#{self.unlock_count} in this session; {len(entries) - 1} earlier logged unlock(s), identical design fingerprint "
              f"in all: {same}) - design sha256 {design_sha256[:16]}...")
        return self._frame.copy()


def chronological_split(df, test_fraction=C.TEST_FRACTION):
    """Earliest 80 % of the hourly timeline = train, latest 20 % = test. No shuffling; the test rows go into the vault."""
    n = len(df)
    k = int(n * (1 - test_fraction))
    train = df.iloc[:k].reset_index(drop=True)
    vault = HeldOutTest(df.iloc[k:])
    assert train["datetime"].is_monotonic_increasing and train["datetime"].iloc[-1] < vault.start
    assert vault.start - train["datetime"].iloc[-1] == pd.Timedelta(hours=1), "split is not contiguous"
    info = {"n_total": n, "split_index": k, "n_train": len(train), "n_test": vault.n_rows,
            "train_start": str(train["datetime"].iloc[0]), "train_end": str(train["datetime"].iloc[-1]),
            "test_start": str(vault.start), "test_end": str(vault.end), "test_fraction": test_fraction, "shuffled": False}
    return train, vault, info
