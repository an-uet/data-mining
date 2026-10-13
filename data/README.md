# Data

| | |
|---|---|
| File | `raw/PRSA_Data_Aotizhongxin_20130301-20170228.csv` (unmodified copy of the file as delivered) |
| Content | hourly air-quality and meteorological observations at the **Aotizhongxin** monitoring station, Beijing |
| Period / size | 2013-03-01 00:00 → 2017-02-28 23:00 · 35,064 rows × 18 columns · regular hourly grid (no missing or duplicated timestamps) |
| Source | UCI Machine Learning Repository, *Beijing Multi-Site Air-Quality Data* (dataset **id 501**) — https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data · DOI 10.24432/C5RK5G |
| Underlying study | Zhang, S., Guo, B., Dong, A., He, J., Xu, Z., Chen, S. X. (2017). *Cautionary tales on air-quality improvement in Beijing.* Proceedings of the Royal Society A 473(2205): 20170457. DOI 10.1098/rspa.2017.0457 |
| SHA-256 | `a94fbcfc71708b6ffdc033360163d91efb7f675e496f6b5860929aa96b96351b` (checked by `src/data.py`) |

> **Not the same as UCI id 381.** The `fetch_ucirepo(id=381)` snippet returns the older *Beijing PM2.5* dataset (43,824 rows, US-embassy station, columns `cbwd/Iws/Is/Ir`).
> This project uses id 501, which `ucimlrepo` cannot import; `src/data.py` reads the CSV above and downloads it from the UCI archive if it is missing.

## Columns (units as listed on the UCI page — please verify there)

| Column | Meaning |
|---|---|
| `No` | row counter (dropped: identifier) |
| `year`, `month`, `day`, `hour` | timestamp parts (combined into `datetime`) |
| `PM2.5`, `PM10` | particulate matter concentration (µg/m³) — `PM2.5` is the forecast target (one hour ahead) |
| `SO2`, `NO2`, `CO`, `O3` | gas concentrations (µg/m³) |
| `TEMP` | temperature (°C) |
| `PRES` | pressure (hPa) |
| `DEWP` | dew-point temperature (°C) |
| `RAIN` | precipitation (mm) |
| `wd` | wind direction (16 compass levels; encoded one-hot, never as integers) |
| `WSPM` | wind speed (m/s) |
| `station` | station name (constant `Aotizhongxin`; dropped) |

Missing values in the file: PM2.5 925 · PM10 718 · SO2 935 · NO2 1,023 · CO 1,776 · O3 1,719 · TEMP, PRES, DEWP, RAIN 20 each · wd 81 · WSPM 14
(`results/tables/table_S01_schema_and_missing_values.csv`). How missing values are handled is described in the report.

## Licence

The data are **not** covered by the MIT licence of this repository. Check the licence shown on the UCI dataset page and cite the DOI above when reusing the data.
(The licence terms were not verified when this repository was prepared.)
