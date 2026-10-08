# -*- coding: utf-8 -*-
"""
MOROCCO CMIP6 MONTHLY DOWNSCALING / BIAS-ADJUSTMENT WORKFLOW
==============================================================

Scientific design
-----------------
1) Reference data: ERA5-Land, 1981-2025.
2) CMIP6 calibration input folder: 1981-2005.
3) CMIP6 validation input folder: 2006-2025.
4) CMIP6 future input folder: 2026-2100.
5) Method/threshold selection: blocked cross-validation ONLY inside 1981-2005.
6) Final fitting of the selected classical method: all calibration data, 1981-2005.
7) Independent validation: 2006-2025; these years are never used in method selection.
8) Future analysis: 2026-2100.
9) Every SSP is processed separately. SSPs are never averaged.

Methods compared
----------------
Precipitation:
    - occurrence correction + monthly linear scaling
    - occurrence correction + empirical quantile mapping (EQM)
    - occurrence correction + quantile delta mapping (QDM)
    - reference dry thresholds tested: 0, 0.5, 1, 2 mm/month

Temperature:
    - additive monthly delta
    - EQM
    - additive QDM

PET / shortwave radiation and other positive variables:
    - multiplicative monthly factor
    - EQM
    - multiplicative QDM

Soil moisture:
    - additive monthly delta, clipped to [0, 1]
    - EQM, clipped to [0, 1]
    - additive QDM, clipped to [0, 1]

Important interpretation
------------------------
This is bias adjustment/statistical downscaling of distributions. A free-running
climate model is not expected to reproduce the exact observed month in the exact
same year. Selection metrics are therefore distributional and climatological,
not based mainly on time-correlations.

Required packages
-----------------
pandas, numpy, scipy, openpyxl

The input readers use the user's previous split wide-Excel structure:
    root/calibration/<scenario>/*.xlsx
    root/validation/<scenario>/*.xlsx
    root/future/<scenario>/*.xlsx
Point matching is performed using exact cleaned ID names, not a fixed count:
    first column = point ID
    optional longitude/latitude columns before the monthly columns
    monthly columns such as Jan_1981, January-1981, 1981-01, etc.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math
import re
import warnings
from typing import Iterable, Literal

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

warnings.filterwarnings("ignore", category=RuntimeWarning)


# =============================================================================
# 1. USER SETTINGS
# =============================================================================

# -------------------------------------------------------------------------
# INPUTS: SAME PATH STRUCTURE AS THE PREVIOUS CLASSICAL CODE
# -------------------------------------------------------------------------
# Observed/reference ERA5-Land Excel files.
REFERENCE_FOLDER = Path(
    r"C:\morocco\Drought\climate results\original_data"
)

# Previous split CMIP6 MIROC6 structure:
#   root/calibration/<scenario>/*.xlsx   -> 1981-2005
#   root/validation/<scenario>/*.xlsx    -> 2006-2025
#   root/future/<scenario>/*.xlsx        -> 2026-2100
CMIP6_SPLIT_EXCEL_ROOT = Path(
    r"C:\morocco\Drought\climate results\CMIP6_MIROC6_calibration_validation_future_reference_style_by_catchment"
)

CALIBRATION_EXCEL_DIR = CMIP6_SPLIT_EXCEL_ROOT / "calibration"
VALIDATION_EXCEL_DIR = CMIP6_SPLIT_EXCEL_ROOT / "validation"
FUTURE_EXCEL_DIR = CMIP6_SPLIT_EXCEL_ROOT / "future"

# The period label is explicit. The code therefore never decides the period
# from the filename or from a 2014/2015 boundary.
MODEL_PERIOD_ROOTS = {
    "MIROC6": {
        "calibration": CALIBRATION_EXCEL_DIR,
        "validation": VALIDATION_EXCEL_DIR,
        "future": FUTURE_EXCEL_DIR,
    },
}

# Compatibility alias used in the settings report.
MODEL_ROOTS = MODEL_PERIOD_ROOTS

# ERA5-Land reference files may be CSV or Excel.
# Selection is based on a meaningful PART of the file name rather than one
# exact full file name. This allows files such as:
#
#   era5_land_pet_hargreaves_mm_month.xlsx
#   my_pet_hargreaves_results.xlsm
#   Temperature max (°C).xlsx
#   Temperature min (°C).xlsx
#
# Variable mapping rules:
#   - any filename/sheet containing "pet_hargreaves" -> potential_evaporation
#   - temperature maximum / Tmax / tasmax           -> temperature_max
#   - temperature minimum / Tmin / tasmin           -> temperature_min
#   - Tmean / temperature_mean                      -> ignored completely
#
# Other existing ERA5-Land variables remain enabled.
REFERENCE_ALLOWED_NAME_PARTS = {
    # Precipitation
    "precipitation",
    "rainfall",

    # Temperature: Tmax and Tmin only
    "temperature_max",
    "temperature_maximum",
    "tmax",
    "tasmax",
    "temperature_min",
    "temperature_minimum",
    "tmin",
    "tasmin",

    # Direct or previously calculated potential evaporation/PET
    "pet_hargreaves",
    "potential_evaporation",
    "potential_evapotranspiration",
    "reference_evaporation",
    "reference_evapotranspiration",
    "eto",
    "et0",

    # Other reference variables
    "soil_moisture_0_10cm",
    "soil_moisture",
    "shortwave",
    "shortwave_radiation",
}

# These variables are explicitly excluded even when such files are present.
REFERENCE_EXCLUDED_NAME_PARTS = {
    "mean_temperature",
    "tmean",
    "tasmean",
}


OUTPUT_ROOT = Path(
    r"C:\morocco\downscaling_method_selection_MIROC6"
)

# ID handling:
# The workflow does NOT require a fixed number such as 393.
# For each model and parameter, it finds the exact point IDs that have complete
# reference, historical, and required-SSP records. Only those common ID names
# are processed. Different parameters may therefore use different point counts.
ID_ALIGNMENT_MODE = "intersection"  # allowed: "intersection" or "strict"
MIN_COMMON_IDS = 1

SAVE_WIDE_EXCEL = True
STRICT_DATA_VALIDATION = True

# Every SSP is processed independently. Set this list to the scenarios that
# must exist for every common model/reference parameter.
REQUIRED_SCENARIOS = ["ssp1_2_6", "ssp2_4_5", "ssp3_7_0", "ssp5_8_5"]

REFERENCE_START = 1981
REFERENCE_END = 2025

# Periods are selected from the corresponding input folders and are also
# checked again from the year columns. Validation is never included in CV.
CALIBRATION_START = 1981
CALIBRATION_END = 2005
VALIDATION_START = 2006
VALIDATION_END = 2025
FUTURE_START = 2026
FUTURE_END = 2100

# Compatibility aliases for helper functions and old report names.
HISTORICAL_START = CALIBRATION_START
HISTORICAL_END = CALIBRATION_END
NEAR_PRESENT_START = VALIDATION_START
NEAR_PRESENT_END = VALIDATION_END

PERIOD_YEAR_RANGES = {
    "calibration": (CALIBRATION_START, CALIBRATION_END),
    "validation": (VALIDATION_START, VALIDATION_END),
    "future": (FUTURE_START, FUTURE_END),
}

# QDM is applied independently to each future horizon. The independent
# validation period is corrected separately and is not part of this dictionary.
APPLICATION_PERIODS = {
    "near_future_2026_2050": (2026, 2050),
    "mid_future_2051_2075": (2051, 2075),
    "far_future_2076_2100": (2076, 2100),
}

# Five non-overlapping blocked-CV folds entirely inside calibration 1981-2005.
# For each fold, that block is held out and all other calibration years train.
CV_BLOCKS = {
    "fold_1_1981_1985": (1981, 1985),
    "fold_2_1986_1990": (1986, 1990),
    "fold_3_1991_1995": (1991, 1995),
    "fold_4_1996_2000": (1996, 2000),
    "fold_5_2001_2005": (2001, 2005),
}

PRECIP_REFERENCE_THRESHOLDS = [0.0, 0.5, 1.0, 2.0]
MIN_TRAIN_VALUES_PER_POINT_MONTH = 18
MIN_APPLY_VALUES_FOR_QDM = 4
EPS = 1e-8

# QDM uses empirical quantiles. These are plotting-position limits, not climate
# thresholds. They prevent exactly 0 and 1 probabilities.
PROB_EPS = 1e-4

# -----------------------------------------------------------------------------
# Unit control
# -----------------------------------------------------------------------------
# The workflow converts every input to the target units below BEFORE method
# selection. The defaults assume that preprocessed Excel files are
# already in the desired units. Specify source units in INPUT_UNITS when a file
# is still in native ERA5/CMIP6 units.
TARGET_UNITS = {
    "precipitation": "mm/month",
    "temperature_max": "degC",
    "temperature_min": "degC",
    "potential_evaporation": "mm/month",
    "soil_moisture": "m3/m3",
    "shortwave_radiation": "W/m2",
}

REFERENCE_INPUT_UNITS = {
    "precipitation": "mm/month",
    "temperature_max": "degC",
    "temperature_min": "degC",
    "potential_evaporation": "mm/month",
    "soil_moisture": "m3/m3",
    "shortwave_radiation": "W/m2",
}

MODEL_INPUT_UNITS = {
    "MIROC6": {
        "precipitation": "mm/month",
            "temperature_max": "degC",
        "temperature_min": "degC",
        "potential_evaporation": "mm/month",
        "soil_moisture": "m3/m3",
        "shortwave_radiation": "W/m2",
    },
    # Add or edit units when the other models are activated:
    # "CanESM5": {...},
    # "MPI-ESM1-2-LR": {...},
}

# Use -1 only when a source stores PET/PE as a negative upward flux. Keep 1
# when PET has already been converted to positive mm/month.
REFERENCE_SIGN_MULTIPLIERS = {"potential_evaporation": 1.0}
MODEL_SIGN_MULTIPLIERS = {
    "MIROC6": {"potential_evaporation": 1.0},
}

# Parameter names internally used by the workflow.
SUPPORTED_PARAMETERS = {
    "precipitation",
    "temperature_max",
    "temperature_min",
    "potential_evaporation",
    "soil_moisture",
    "shortwave_radiation",
}


# =============================================================================
# 2. OUTPUT FOLDERS
# =============================================================================

DIR_INVENTORY = OUTPUT_ROOT / "00_inventory"
DIR_CV = OUTPUT_ROOT / "01_blocked_cv_inside_calibration_1981_2005"
DIR_SELECTION = OUTPUT_ROOT / "02_selected_methods"
DIR_CALIBRATION = OUTPUT_ROOT / "03_corrected_calibration_1981_2005"
DIR_VALIDATION = OUTPUT_ROOT / "04_independent_validation_2006_2025"
DIR_FUTURE = OUTPUT_ROOT / "05_corrected_future_2026_2100"
DIR_SIGNAL = OUTPUT_ROOT / "06_change_signal_diagnostics"
DIR_WIDE = OUTPUT_ROOT / "07_wide_excel_outputs"
DIR_LOG = OUTPUT_ROOT / "08_logs"
DIR_PERFORMANCE = OUTPUT_ROOT / "09_performance_metrics"

# Compatibility aliases used by helper code.
DIR_HIST = DIR_CALIBRATION
DIR_NEAR = DIR_VALIDATION

for _d in [
    OUTPUT_ROOT, DIR_INVENTORY, DIR_CV, DIR_SELECTION, DIR_CALIBRATION,
    DIR_VALIDATION, DIR_FUTURE, DIR_SIGNAL, DIR_WIDE, DIR_LOG,
    DIR_PERFORMANCE,
]:
    _d.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 3. NAME / DATE / NUMBER HELPERS
# =============================================================================

MONTH_MAP = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}
MONTH_ABBR = {i: pd.Timestamp(2000, i, 1).strftime("%b") for i in range(1, 13)}


def normalize_text(value: object) -> str:
    text = str(value).strip().lower()
    text = text.replace("−", "-").replace("–", "-").replace("—", "-")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def clean_id(value: object) -> str:
    if pd.isna(value):
        return ""
    try:
        x = float(value)
        if x.is_integer():
            return str(int(x))
    except Exception:
        pass
    return str(value).strip()


def clean_number(value: object) -> float:
    """Parse numeric strings, including decimal commas, safely."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    s = str(value).strip().replace("−", "-").replace("–", "-").replace("—", "-")
    if s.lower() in {"", "nan", "none", "null", "na", "n/a", "missing", "--"}:
        return np.nan

    negative = s.startswith("(") and s.endswith(")")
    if negative:
        s = s[1:-1]

    s = s.replace(" ", "")
    if "," in s and "." not in s:
        # Treat one comma followed by 1-6 digits as a decimal comma.
        if re.fullmatch(r"[-+]?\d+,\d{1,6}", s):
            s = s.replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s and "." in s:
        # Assume commas are thousands separators when a decimal point exists.
        s = s.replace(",", "")

    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    if not match:
        return np.nan
    try:
        out = float(match.group(0))
        return -out if negative else out
    except ValueError:
        return np.nan


def parse_month_column(column: object) -> tuple[int, int] | None:
    if isinstance(column, (pd.Timestamp,)):
        return int(column.year), int(column.month)

    text = str(column).strip()

    m = re.fullmatch(r"([A-Za-z]+)[_\-\s/]+(\d{4})", text)
    if m and m.group(1).lower() in MONTH_MAP:
        return int(m.group(2)), MONTH_MAP[m.group(1).lower()]

    m = re.fullmatch(r"(\d{4})[_\-\s/]+(\d{1,2})", text)
    if m:
        year, month = int(m.group(1)), int(m.group(2))
        if 1 <= month <= 12:
            return year, month

    m = re.fullmatch(r"(\d{1,2})[_\-\s/]+(\d{4})", text)
    if m:
        month, year = int(m.group(1)), int(m.group(2))
        if 1 <= month <= 12:
            return year, month

    return None


def normalize_scenario(value: object) -> str:
    n = normalize_text(value).replace("_", "")
    mapping = {
        "ssp126": "ssp1_2_6",
        "ssp245": "ssp2_4_5",
        "ssp370": "ssp3_7_0",
        "ssp585": "ssp5_8_5",
        "historical": "historical",
    }
    for key, output in mapping.items():
        if key in n:
            return output
    return "unknown"


def infer_parameter(text: object) -> str | None:
    """
    Infer the internal parameter name from a filename or Excel sheet name.

    Required behavior:
      * pet_hargreaves -> potential_evaporation
      * Tmax/tasmax/temperature_max -> temperature_max
      * Tmin/tasmin/temperature_min -> temperature_min
      * Tmean/tas/temperature_mean -> ignored
    """
    n = normalize_text(text)
    tokens = set(n.split("_"))

    # Explicitly ignore mean temperature.
    if (
        "temperature_mean" in n
        or "mean_temperature" in n
        or "tmean" in tokens
        or "tasmean" in tokens
        or n in {"tas", "temperature", "temperature_mean"}
    ):
        return None

    # Hargreaves potential evapotranspiration.
    # A filename or sheet name containing this exact normalized phrase is
    # always interpreted as monthly potential evaporation.
    if "pet_hargreaves" in n:
        return "potential_evaporation"

    # Also preserve recognition of other clearly named PET/ET0 files.
    if (
        ("potential" in n and ("evap" in n or "eto" in n or "pet" in n))
        or "pet" in tokens
        or "eto" in tokens
        or "et0" in tokens
    ):
        return "potential_evaporation"

    if (
        "shortwave" in n
        or "rsds" in n
        or ("solar" in n and "radiation" in n)
    ):
        return "shortwave_radiation"

    if "soil" in n and "moist" in n:
        return "soil_moisture"

    if (
        "precip" in n
        or "rainfall" in n
        or n in {"pr", "rain"}
    ):
        return "precipitation"

    if (
        (
            "temperature" in n
            or "temp" in n
            or "tas" in n
        )
        and (
            "max" in n
            or "tasmax" in n
            or "tmax" in n
        )
    ):
        return "temperature_max"

    if (
        (
            "temperature" in n
            or "temp" in n
            or "tas" in n
        )
        and (
            "min" in n
            or "tasmin" in n
            or "tmin" in n
        )
    ):
        return "temperature_min"

    return None


def infer_scenario_from_path(path: Path, root: Path) -> str:
    """
    Infer SSP scenario from the complete file path and configured root.

    Important:
    Each MIROC6 future root is itself scenario-specific, for example:
        .../cmip6_future/ssp1_2_6/miroc6

    The previous implementation inspected only the path relative to that root.
    Therefore, when a PET filename did not include the scenario text, its
    scenario became "unknown". Scenario identification examines the full
    file path and the root path, so generic PET filenames are assigned to the
    correct SSP according to their parent folder.
    """
    candidates = [
        str(path),
        str(root),
        path.parent.name,
        root.parent.name,
        path.stem,
    ]

    for candidate in candidates:
        scenario = normalize_scenario(candidate)
        if scenario != "unknown":
            return scenario

    return "unknown"


def find_lon_lat_columns(columns: Iterable[object]) -> tuple[object | None, object | None]:
    lon_col = lat_col = None
    for col in columns:
        n = normalize_text(col)
        if lon_col is None and n in {"lon", "longitude", "x", "long"}:
            lon_col = col
        if lat_col is None and n in {"lat", "latitude", "y"}:
            lat_col = col
    return lon_col, lat_col


def normalize_unit(unit: str) -> str:
    """Normalize supported unit labels to compact canonical strings."""
    u = str(unit).strip().lower()
    u = u.replace("°", "deg").replace("−", "-").replace("–", "-")
    u = re.sub(r"\s+", "", u)
    aliases = {
        "mm/month": "mm/month",
        "mmmonth-1": "mm/month",
        "mmmon-1": "mm/month",
        "mm/mo": "mm/month",
        "mm/day": "mm/day",
        "mmd-1": "mm/day",
        "m/month": "m/month",
        "mmonth-1": "m/month",
        "kgm-2s-1": "kg/m2/s",
        "kg/m2/s": "kg/m2/s",
        "kgm^-2s^-1": "kg/m2/s",
        "k": "k",
        "kelvin": "k",
        "degc": "degc",
        "c": "degc",
        "celsius": "degc",
        "m3/m3": "m3/m3",
        "m^3/m^3": "m3/m3",
        "%": "percent",
        "percent": "percent",
        "w/m2": "w/m2",
        "wm-2": "w/m2",
        "j/m2/month": "j/m2/month",
        "jm-2month-1": "j/m2/month",
        "mj/m2/month": "mj/m2/month",
        "mjm-2month-1": "mj/m2/month",
    }
    return aliases.get(u, u)


def seconds_in_month(dates: pd.Series) -> np.ndarray:
    dates = pd.to_datetime(dates)
    return dates.dt.days_in_month.to_numpy(float) * 86400.0


def days_in_month(dates: pd.Series) -> np.ndarray:
    dates = pd.to_datetime(dates)
    return dates.dt.days_in_month.to_numpy(float)


def convert_parameter_units(
    values: pd.Series,
    dates: pd.Series,
    parameter: str,
    input_unit: str,
) -> pd.Series:
    """Convert one parameter to TARGET_UNITS using explicit user settings."""
    unit = normalize_unit(input_unit)
    x = pd.to_numeric(values, errors="coerce").astype(float).to_numpy()
    sec = seconds_in_month(dates)
    days = days_in_month(dates)

    if parameter in {"precipitation", "potential_evaporation"}:
        if unit == "mm/month":
            out = x
        elif unit == "mm/day":
            out = x * days
        elif unit == "m/month":
            out = x * 1000.0
        elif unit == "kg/m2/s":
            # 1 kg m-2 of liquid water equals 1 mm water depth.
            out = x * sec
        else:
            raise ValueError(
                f"Unsupported unit '{input_unit}' for {parameter}. "
                "Supported: mm/month, mm/day, m/month, kg/m2/s."
            )

    elif parameter.startswith("temperature_"):
        if unit == "degc":
            out = x
        elif unit == "k":
            out = x - 273.15
        else:
            raise ValueError(
                f"Unsupported unit '{input_unit}' for {parameter}. Supported: degC, K."
            )

    elif parameter == "soil_moisture":
        if unit == "m3/m3":
            out = x
        elif unit == "percent":
            out = x / 100.0
        else:
            raise ValueError(
                f"Unsupported unit '{input_unit}' for soil_moisture. Supported: m3/m3, percent."
            )

    elif parameter == "shortwave_radiation":
        if unit == "w/m2":
            out = x
        elif unit == "j/m2/month":
            out = x / sec
        elif unit == "mj/m2/month":
            out = x * 1.0e6 / sec
        else:
            raise ValueError(
                f"Unsupported unit '{input_unit}' for shortwave_radiation. "
                "Supported: W/m2, J/m2/month, MJ/m2/month."
            )
    else:
        out = x

    return pd.Series(out, index=values.index, dtype=float)


def apply_unit_conversion(
    df: pd.DataFrame,
    source_name: str,
    unit_map: dict[str, str],
    sign_multipliers: dict[str, float],
) -> pd.DataFrame:
    """Apply explicit unit conversion and sign convention per parameter."""
    out = df.copy()
    conversion_log: list[dict] = []
    for parameter, idx in out.groupby("parameter").groups.items():
        if parameter not in TARGET_UNITS:
            continue
        input_unit = unit_map.get(parameter)
        if input_unit is None:
            raise ValueError(
                f"No input unit configured for {source_name} | {parameter}. "
                "Add it to REFERENCE_INPUT_UNITS or MODEL_INPUT_UNITS."
            )
        before = pd.to_numeric(out.loc[idx, "value"], errors="coerce")
        converted = convert_parameter_units(
            before,
            out.loc[idx, "date"],
            parameter,
            input_unit,
        )
        multiplier = float(sign_multipliers.get(parameter, 1.0))
        converted = converted * multiplier
        out.loc[idx, "value"] = converted.to_numpy()
        conversion_log.append({
            "source": source_name,
            "parameter": parameter,
            "input_unit": input_unit,
            "target_unit": TARGET_UNITS[parameter],
            "sign_multiplier": multiplier,
            "n_values": int(len(idx)),
            "before_min": float(before.min()) if before.notna().any() else np.nan,
            "before_max": float(before.max()) if before.notna().any() else np.nan,
            "after_min": float(converted.min()) if converted.notna().any() else np.nan,
            "after_max": float(converted.max()) if converted.notna().any() else np.nan,
        })

    pd.DataFrame(conversion_log).to_csv(
        DIR_INVENTORY / f"{normalize_text(source_name)}_unit_conversion.csv",
        index=False,
    )
    return out


def physical_range_violations(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """Return clearly impossible values after conversion to target units."""
    rows: list[dict] = []
    bounds = {
        "precipitation": (0.0, 5000.0),
        "potential_evaporation": (0.0, 3000.0),
        "temperature_min": (-90.0, 65.0),
        "temperature_max": (-70.0, 80.0),
        "soil_moisture": (0.0, 1.0),
        "shortwave_radiation": (0.0, 600.0),
    }
    for parameter, group in df.groupby("parameter"):
        if parameter not in bounds:
            continue
        lo, hi = bounds[parameter]
        bad = group[(group["value"] < lo - 1e-9) | (group["value"] > hi + 1e-9)]
        rows.append({
            "source": source_name,
            "parameter": parameter,
            "target_unit": TARGET_UNITS.get(parameter, ""),
            "lower_bound": lo,
            "upper_bound": hi,
            "n_violations": int(len(bad)),
            "sample_values": "; ".join(map(str, bad["value"].head(5).round(6).tolist())),
        })
    return pd.DataFrame(rows)


# =============================================================================
# 4. INPUT READERS
# =============================================================================


def dataframe_wide_to_long(
    df: pd.DataFrame,
    parameter: str,
    source_file: str,
    source_sheet: str,
    scenario_hint: str,
    model: str,
) -> pd.DataFrame:
    if df is None or df.empty or df.shape[1] < 2:
        return pd.DataFrame()

    month_info: dict[object, tuple[int, int]] = {}
    for col in df.columns:
        parsed = parse_month_column(col)
        if parsed is not None:
            month_info[col] = parsed
    if not month_info:
        return pd.DataFrame()

    first_month_position = min(df.columns.get_loc(c) for c in month_info)
    metadata_cols = list(df.columns[:first_month_position])
    id_col = metadata_cols[0] if metadata_cols else df.columns[0]
    lon_col, lat_col = find_lon_lat_columns(metadata_cols)

    keep_cols = [id_col]
    for col in [lon_col, lat_col]:
        if col is not None and col not in keep_cols:
            keep_cols.append(col)
    keep_cols += list(month_info.keys())

    temp = df[keep_cols].copy()
    temp = temp.rename(columns={id_col: "point_id"})
    temp["point_id"] = temp["point_id"].map(clean_id)

    id_vars = ["point_id"]
    if lon_col is not None:
        temp = temp.rename(columns={lon_col: "lon"})
        id_vars.append("lon")
    else:
        temp["lon"] = np.nan
        id_vars.append("lon")
    if lat_col is not None:
        temp = temp.rename(columns={lat_col: "lat"})
        id_vars.append("lat")
    else:
        temp["lat"] = np.nan
        id_vars.append("lat")

    long = temp.melt(
        id_vars=id_vars,
        value_vars=list(month_info.keys()),
        var_name="month_column",
        value_name="value",
    )
    long["year"] = long["month_column"].map(lambda c: month_info[c][0])
    long["month"] = long["month_column"].map(lambda c: month_info[c][1])
    long["date"] = pd.to_datetime(
        long["year"].astype(str) + "-" + long["month"].astype(str).str.zfill(2) + "-01"
    )
    long["value"] = long["value"].map(clean_number)
    long["lon"] = long["lon"].map(clean_number)
    long["lat"] = long["lat"].map(clean_number)
    long["parameter"] = parameter
    long["scenario_hint"] = scenario_hint
    long["model"] = model
    long["source_file"] = source_file
    long["source_sheet"] = source_sheet
    long = long[(long["point_id"] != "") & long["value"].notna()].copy()

    return long[[
        "model", "scenario_hint", "parameter", "point_id", "lon", "lat",
        "date", "year", "month", "value", "source_file", "source_sheet",
    ]]


def read_reference_folder(folder: Path) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    inventory: list[dict] = []

    paths: list[Path] = []
    for ext in ("*.xlsx", "*.xls", "*.xlsm", "*.csv"):
        paths.extend(folder.rglob(ext))

    for path in sorted(set(paths)):
        if path.name.startswith("~$"):
            continue

        # Read selected ERA5-Land files by filename PART, for both CSV
        # and Excel inputs. Tmean is explicitly excluded.
        normalized_stem = normalize_text(path.stem)
        normalized_tokens = set(normalized_stem.split("_"))

        is_excluded_mean_temperature = (
            any(
                part in normalized_stem
                for part in REFERENCE_EXCLUDED_NAME_PARTS
            )
            or "tmean" in normalized_tokens
            or "tasmean" in normalized_tokens
        )
        if is_excluded_mean_temperature:
            continue

        is_allowed_reference_file = any(
            normalize_text(part) in normalized_stem
            for part in REFERENCE_ALLOWED_NAME_PARTS
        )
        if not is_allowed_reference_file:
            continue

        file_parameter = infer_parameter(path.stem)

        if path.suffix.lower() == ".csv":
            sheets = {"csv": pd.read_csv(path)}
        else:
            sheets = pd.read_excel(path, sheet_name=None)

        for sheet_name, df in sheets.items():
            normalized_sheet = normalize_text(sheet_name)
            sheet_tokens = set(normalized_sheet.split("_"))

            # Never read a Tmean sheet.
            if (
                "temperature_mean" in normalized_sheet
                or "mean_temperature" in normalized_sheet
                or "tmean" in sheet_tokens
                or "tasmean" in sheet_tokens
            ):
                continue

            parameter = infer_parameter(sheet_name) or file_parameter
            if parameter not in SUPPORTED_PARAMETERS:
                continue
            long = dataframe_wide_to_long(
                df=df,
                parameter=parameter,
                source_file=str(path),
                source_sheet=str(sheet_name),
                scenario_hint="reference",
                model="ERA5_Land",
            )
            if not long.empty:
                long = long[(long["year"] >= REFERENCE_START) & (long["year"] <= REFERENCE_END)]
                parts.append(long)
                inventory.append({
                    "source": "reference", "file": str(path), "sheet": str(sheet_name),
                    "parameter": parameter, "rows": len(long),
                    "points": long["point_id"].nunique(),
                    "start_year": int(long["year"].min()),
                    "end_year": int(long["year"].max()),
                })

    if not parts:
        raise FileNotFoundError(f"No usable reference monthly data found in: {folder}")

    out = pd.concat(parts, ignore_index=True)
    out = (
        out.groupby(["parameter", "point_id", "date", "year", "month"], as_index=False)
        .agg(value=("value", "mean"), lon=("lon", "first"), lat=("lat", "first"))
    )
    out = apply_unit_conversion(
        out,
        source_name="ERA5_Land",
        unit_map=REFERENCE_INPUT_UNITS,
        sign_multipliers=REFERENCE_SIGN_MULTIPLIERS,
    )

    # Same convention as the previous code: observed/reference PET may be stored
    # as a negative upward flux. Convert it to positive mm/month safely.
    pet_mask = out["parameter"] == "potential_evaporation"
    out.loc[pet_mask, "value"] = out.loc[pet_mask, "value"].abs()

    pd.DataFrame(inventory).to_csv(DIR_INVENTORY / "reference_inventory.csv", index=False)
    return out


def read_model_root(
    model: str,
    roots: dict[str, Path] | Path | list[Path] | tuple[Path, ...],
) -> pd.DataFrame:
    """Read split calibration, validation and future MIROC6 Excel folders.

    Preferred input is a mapping with the keys ``calibration``, ``validation``
    and ``future``. The folder controls the period; the year columns are then
    checked against the corresponding range. The scenario is read from the
    scenario subfolder/path and is retained for all three periods.
    """
    parts: list[pd.DataFrame] = []
    inventory: list[dict] = []

    if isinstance(roots, dict):
        root_entries = [(str(period), Path(root)) for period, root in roots.items()]
    elif isinstance(roots, Path):
        root_entries = [(normalize_text(roots.name), roots)]
    else:
        root_entries = [(normalize_text(Path(root).name), Path(root)) for root in roots]

    required_periods = {"calibration", "validation", "future"}
    configured_periods = {period for period, _ in root_entries}
    missing_periods = sorted(required_periods - configured_periods)
    if missing_periods:
        raise ValueError(
            f"Missing configured period folder(s) for {model}: {missing_periods}. "
            "MODEL_PERIOD_ROOTS must contain calibration, validation and future."
        )

    missing_roots = [str(root) for _, root in root_entries if not root.exists()]
    if missing_roots:
        raise FileNotFoundError(
            "The following configured MIROC6 split input folder(s) do not exist:\n"
            + "\n".join(missing_roots)
        )

    discovered: list[tuple[str, Path, Path]] = []
    for period_type, root in root_entries:
        if period_type not in PERIOD_YEAR_RANGES:
            continue
        for ext in ("*.xlsx", "*.xls", "*.xlsm", "*.csv"):
            for path in root.rglob(ext):
                discovered.append((period_type, path, root))

    discovered = sorted(
        set(discovered),
        key=lambda item: (item[0], str(item[2]).lower(), str(item[1]).lower()),
    )

    for period_type, path, root in discovered:
        if path.name.startswith("~$"):
            continue

        normalized_name = normalize_text(path.name)
        if any(token in normalized_name for token in [
            "quality", "report", "summary", "diagnostic", "downscaled",
        ]):
            continue

        # The previous root name contains MIROC6, so this safety test accepts
        # files even when the individual workbook name does not contain MIROC6.
        path_text = normalize_text(str(path))
        root_text = normalize_text(str(root))
        if "miroc6" not in path_text and "miroc6" not in root_text:
            continue

        scenario_hint = infer_scenario_from_path(path, root)
        scenario = normalize_scenario(scenario_hint)
        if scenario == "unknown":
            raise ValueError(
                "Could not infer the SSP scenario from the split input path.\n"
                f"Period: {period_type}\nFile: {path}\nConfigured root: {root}\n"
                "Place each workbook under an SSP folder such as ssp1_2_6, "
                "ssp2_4_5, ssp3_7_0 or ssp5_8_5."
            )

        file_parameter = infer_parameter(path.stem)
        if path.suffix.lower() == ".csv":
            sheets = {"csv": pd.read_csv(path)}
        else:
            sheets = pd.read_excel(path, sheet_name=None)

        period_start, period_end = PERIOD_YEAR_RANGES[period_type]

        for sheet_name, df in sheets.items():
            if normalize_text(sheet_name) in {"summary", "readme", "metadata"}:
                continue

            parameter = infer_parameter(sheet_name) or file_parameter
            if parameter not in SUPPORTED_PARAMETERS:
                continue

            long = dataframe_wide_to_long(
                df=df,
                parameter=parameter,
                source_file=str(path),
                source_sheet=str(sheet_name),
                scenario_hint=scenario,
                model=model,
            )
            if long.empty:
                continue

            long["scenario"] = scenario
            long["period_type"] = period_type
            long = long[long["year"].between(period_start, period_end)].copy()
            if long.empty:
                continue

            parts.append(long)
            inventory.append({
                "source": "cmip6",
                "model": model,
                "period_type": period_type,
                "input_root": str(root),
                "file": str(path),
                "sheet": str(sheet_name),
                "parameter": parameter,
                "scenario": scenario,
                "rows": len(long),
                "points": long["point_id"].nunique(),
                "expected_start_year": period_start,
                "expected_end_year": period_end,
                "actual_start_year": int(long["year"].min()),
                "actual_end_year": int(long["year"].max()),
            })

    if not parts:
        roots_text = "\n".join(f"{period}: {root}" for period, root in root_entries)
        raise FileNotFoundError(
            f"No usable MIROC6 monthly data were found in the split folders:\n{roots_text}"
        )

    out = pd.concat(parts, ignore_index=True)

    duplicate_report = (
        out.groupby(
            ["period_type", "scenario", "parameter", "point_id", "date"],
            as_index=False,
        )["value"]
        .agg(["min", "max", "count"])
        .reset_index()
    )
    duplicate_report["spread"] = duplicate_report["max"] - duplicate_report["min"]
    duplicate_report.to_csv(
        DIR_INVENTORY / f"{normalize_text(model)}_split_input_duplicate_spread.csv",
        index=False,
        encoding="utf-8-sig",
    )

    final = (
        out.groupby(
            [
                "model", "period_type", "scenario", "parameter", "point_id",
                "date", "year", "month",
            ],
            as_index=False,
        )
        .agg(value=("value", "mean"), lon=("lon", "first"), lat=("lat", "first"))
    )

    if model not in MODEL_INPUT_UNITS:
        raise ValueError(
            f"No MODEL_INPUT_UNITS entry configured for '{model}'. "
            "Add explicit input units before running the workflow."
        )

    final = apply_unit_conversion(
        final,
        source_name=model,
        unit_map=MODEL_INPUT_UNITS[model],
        sign_multipliers=MODEL_SIGN_MULTIPLIERS.get(model, {}),
    )

    pd.DataFrame(inventory).to_csv(
        DIR_INVENTORY / f"{normalize_text(model)}_cmip6_split_inventory.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return final


# =============================================================================
# 4B. STRICT DATA VALIDATION
# =============================================================================


def _check_record(
    rows: list[dict],
    dataset: str,
    parameter: str,
    scenario: str,
    check: str,
    passed: bool,
    expected: object,
    actual: object,
    details: str = "",
) -> None:
    rows.append({
        "dataset": dataset,
        "parameter": parameter,
        "scenario": scenario,
        "check": check,
        "status": "PASS" if passed else "ERROR",
        "expected": expected,
        "actual": actual,
        "details": details,
    })


def complete_ids_for_period(
    group: pd.DataFrame,
    start_year: int,
    end_year: int,
) -> set[str]:
    """
    Return exact point-ID names having one valid value for every expected month.

    The returned set is based on ID identity, not on a required point count.
    """
    expected_dates = pd.date_range(
        f"{start_year}-01-01", f"{end_year}-12-01", freq="MS"
    )
    expected_per_point = len(expected_dates)

    valid = group[group["year"].between(start_year, end_year)].copy()
    if valid.empty:
        return set()

    valid["point_id"] = valid["point_id"].astype(str)
    counts = (
        valid.drop_duplicates(["point_id", "date"])
        .groupby("point_id")["date"]
        .nunique()
    )
    return set(counts[counts == expected_per_point].index.astype(str))


def validate_group_coverage(
    group: pd.DataFrame,
    dataset: str,
    parameter: str,
    scenario: str,
    start_year: int,
    end_year: int,
    expected_ids: set[str] | None = None,
) -> list[dict]:
    """
    Validate IDs by their exact names and validate monthly coverage.

    In intersection mode, incomplete or unmatched IDs are reported but are
    excluded later instead of forcing every parameter to contain 393 points.
    In strict mode, the ID set must exactly match the corresponding reference.
    """
    rows: list[dict] = []
    expected_dates = pd.date_range(
        f"{start_year}-01-01", f"{end_year}-12-01", freq="MS"
    )
    ids = set(group["point_id"].astype(str).unique())

    _check_record(
        rows,
        dataset,
        parameter,
        scenario,
        "at_least_one_point_id",
        len(ids) >= MIN_COMMON_IDS,
        f">= {MIN_COMMON_IDS} ID name(s)",
        len(ids),
        f"ID sample={sorted(ids)[:10]}",
    )

    duplicate_count = int(
        group.duplicated(["point_id", "date"], keep=False).sum()
    )
    _check_record(
        rows,
        dataset,
        parameter,
        scenario,
        "duplicate_point_month_rows",
        duplicate_count == 0,
        0,
        duplicate_count,
        "Duplicates must be resolved before downscaling.",
    )

    valid_range = group[group["year"].between(start_year, end_year)].copy()
    expected_per_point = len(expected_dates)
    counts = (
        valid_range.drop_duplicates(["point_id", "date"])
        .groupby("point_id")["date"]
        .nunique()
    )
    complete_ids = set(
        counts[counts == expected_per_point].index.astype(str)
    )
    incomplete_ids = sorted(ids - complete_ids)

    # With ID intersection, incomplete IDs are removed later and do not stop
    # the entire workflow. We still report their exact names.
    coverage_passed = (
        len(complete_ids) >= MIN_COMMON_IDS
        if ID_ALIGNMENT_MODE == "intersection"
        else not incomplete_ids
    )
    _check_record(
        rows,
        dataset,
        parameter,
        scenario,
        "complete_monthly_IDs_available",
        coverage_passed,
        (
            f">= {MIN_COMMON_IDS} complete ID(s)"
            if ID_ALIGNMENT_MODE == "intersection"
            else f"{expected_per_point} months for every ID"
        ),
        (
            f"complete={len(complete_ids)}, "
            f"incomplete={len(incomplete_ids)}"
        ),
        f"incomplete ID sample={incomplete_ids[:20]}",
    )

    actual_start = (
        int(valid_range["year"].min()) if not valid_range.empty else None
    )
    actual_end = (
        int(valid_range["year"].max()) if not valid_range.empty else None
    )
    _check_record(
        rows,
        dataset,
        parameter,
        scenario,
        "year_range",
        actual_start == start_year and actual_end == end_year,
        f"{start_year}-{end_year}",
        f"{actual_start}-{actual_end}",
        "",
    )

    if expected_ids is not None:
        expected_ids = set(map(str, expected_ids))
        missing_ids = sorted(expected_ids - complete_ids)
        extra_ids = sorted(complete_ids - expected_ids)
        common_ids = sorted(expected_ids.intersection(complete_ids))

        if ID_ALIGNMENT_MODE == "strict":
            passed = not missing_ids and not extra_ids
            expected_text = f"exact same {len(expected_ids)} ID names as reference"
        else:
            passed = len(common_ids) >= MIN_COMMON_IDS
            expected_text = (
                f">= {MIN_COMMON_IDS} common complete ID name(s) with reference"
            )

        _check_record(
            rows,
            dataset,
            parameter,
            scenario,
            "point_ID_name_alignment",
            passed,
            expected_text,
            (
                f"reference={len(expected_ids)}, complete_model={len(complete_ids)}, "
                f"common={len(common_ids)}, missing={len(missing_ids)}, "
                f"extra={len(extra_ids)}"
            ),
            (
                f"common sample={common_ids[:10]}; "
                f"missing sample={missing_ids[:10]}; "
                f"extra sample={extra_ids[:10]}"
            ),
        )

    return rows


def build_id_alignment_for_model(
    reference: pd.DataFrame,
    cmip: pd.DataFrame,
    model: str,
) -> tuple[dict[str, set[str]], pd.DataFrame, pd.DataFrame]:
    """Find IDs complete in reference and every scenario/period combination.

    The intersection is calculated separately for each parameter across:
      * ERA5-Land reference, 1981-2025;
      * calibration, 1981-2005, for every required SSP;
      * validation, 2006-2025, for every required SSP;
      * future, 2026-2100, for every required SSP.
    """
    parameter_ids: dict[str, set[str]] = {}
    summary_rows: list[dict] = []
    excluded_rows: list[dict] = []

    common_parameters = sorted(
        set(reference["parameter"])
        .intersection(cmip["parameter"])
        .intersection(SUPPORTED_PARAMETERS)
    )

    for parameter in common_parameters:
        ref_group = reference[reference["parameter"] == parameter]
        ref_ids = complete_ids_for_period(ref_group, REFERENCE_START, REFERENCE_END)
        source_sets: dict[str, set[str]] = {
            "ERA5_reference_1981_2025": ref_ids,
        }

        period_counts: dict[tuple[str, str], int] = {}
        for scenario in REQUIRED_SCENARIOS:
            for period_type, (start_year, end_year) in PERIOD_YEAR_RANGES.items():
                group = cmip[
                    (cmip["parameter"] == parameter)
                    & (cmip["scenario"] == scenario)
                    & (cmip["period_type"] == period_type)
                ]
                ids = complete_ids_for_period(group, start_year, end_year)
                key = f"CMIP6_{scenario}_{period_type}_{start_year}_{end_year}"
                source_sets[key] = ids
                period_counts[(scenario, period_type)] = len(ids)

        if ID_ALIGNMENT_MODE == "strict":
            all_sets = list(source_sets.values())
            first = all_sets[0] if all_sets else set()
            if not all(values == first for values in all_sets[1:]):
                counts = {name: len(values) for name, values in source_sets.items()}
                raise ValueError(
                    f"Strict ID matching failed for {model} | {parameter}. "
                    f"Counts by source: {counts}"
                )
            common_ids = set(first)
        else:
            common_ids = (
                set.intersection(*(set(values) for values in source_sets.values()))
                if source_sets else set()
            )

        if len(common_ids) < MIN_COMMON_IDS:
            counts = {name: len(values) for name, values in source_sets.items()}
            raise ValueError(
                f"No sufficient common complete ID names for {model} | {parameter}. "
                f"Counts by source: {counts}"
            )

        parameter_ids[parameter] = common_ids
        row = {
            "model": model,
            "parameter": parameter,
            "alignment_mode": ID_ALIGNMENT_MODE,
            "common_ID_count": len(common_ids),
            "common_IDs": "; ".join(sorted(common_ids)),
            "reference_complete_ID_count": len(ref_ids),
        }
        for scenario in REQUIRED_SCENARIOS:
            for period_type in PERIOD_YEAR_RANGES:
                row[f"{scenario}_{period_type}_complete_ID_count"] = period_counts.get(
                    (scenario, period_type), 0
                )
        summary_rows.append(row)

        for source_name, source_ids in source_sets.items():
            for point_id in sorted(source_ids - common_ids):
                excluded_rows.append({
                    "model": model,
                    "parameter": parameter,
                    "source_or_period": source_name,
                    "point_id": point_id,
                    "reason": (
                        "Valid in this source but excluded because it is not complete "
                        "in every required scenario and period."
                    ),
                })

    return parameter_ids, pd.DataFrame(summary_rows), pd.DataFrame(excluded_rows)



def validate_reference_dataset(reference: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for parameter, group in reference.groupby("parameter"):
        rows.extend(validate_group_coverage(
            group=group,
            dataset="ERA5_Land",
            parameter=parameter,
            scenario="reference",
            start_year=REFERENCE_START,
            end_year=REFERENCE_END,
        ))

    range_report = physical_range_violations(reference, "ERA5_Land")
    for _, row in range_report.iterrows():
        _check_record(
            rows,
            "ERA5_Land",
            str(row["parameter"]),
            "reference",
            "physical_range_after_unit_conversion",
            int(row["n_violations"]) == 0,
            0,
            int(row["n_violations"]),
            str(row["sample_values"]),
        )
    return pd.DataFrame(rows)


def validate_model_dataset(
    cmip: pd.DataFrame,
    reference: pd.DataFrame,
    model: str,
) -> pd.DataFrame:
    rows: list[dict] = []
    common_parameters = sorted(set(reference["parameter"]).intersection(cmip["parameter"]))

    for parameter in common_parameters:
        ref_ids = set(
            reference[reference["parameter"] == parameter]["point_id"]
            .astype(str).unique()
        )

        for period_type in PERIOD_YEAR_RANGES:
            present = set(
                cmip[
                    (cmip["parameter"] == parameter)
                    & (cmip["period_type"] == period_type)
                ]["scenario"].dropna().astype(str).unique()
            )
            missing = [s for s in REQUIRED_SCENARIOS if s not in present]
            _check_record(
                rows, model, parameter, period_type,
                "required_scenarios_present",
                not missing,
                ", ".join(REQUIRED_SCENARIOS),
                ", ".join(sorted(present)),
                f"missing={missing}",
            )

        for scenario in REQUIRED_SCENARIOS:
            for period_type, (start_year, end_year) in PERIOD_YEAR_RANGES.items():
                group = cmip[
                    (cmip["parameter"] == parameter)
                    & (cmip["scenario"] == scenario)
                    & (cmip["period_type"] == period_type)
                    & cmip["year"].between(start_year, end_year)
                ]
                rows.extend(validate_group_coverage(
                    group=group,
                    dataset=model,
                    parameter=parameter,
                    scenario=f"{scenario}_{period_type}",
                    start_year=start_year,
                    end_year=end_year,
                    expected_ids=ref_ids,
                ))

    range_report = physical_range_violations(cmip, model)
    for _, row in range_report.iterrows():
        _check_record(
            rows,
            model,
            str(row["parameter"]),
            "all_split_periods",
            "physical_range_after_unit_conversion",
            int(row["n_violations"]) == 0,
            0,
            int(row["n_violations"]),
            str(row["sample_values"]),
        )
    return pd.DataFrame(rows)


def save_and_enforce_validation(report: pd.DataFrame, filename: str) -> None:
    path = DIR_INVENTORY / filename
    report.to_csv(path, index=False, encoding="utf-8-sig")
    errors = report[report["status"] == "ERROR"]
    if STRICT_DATA_VALIDATION and not errors.empty:
        sample = errors[["dataset", "parameter", "scenario", "check", "actual"]].head(12)
        raise ValueError(
            f"Strict data validation failed. See: {path}\n"
            + sample.to_string(index=False)
        )


# =============================================================================
# 5. EMPIRICAL DISTRIBUTION FUNCTIONS
# =============================================================================


def finite(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    return arr[np.isfinite(arr)]


def empirical_probabilities(sample: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Empirical CDF probabilities using linear interpolation of plotting positions."""
    sample = np.sort(finite(sample))
    values = np.asarray(values, dtype=float)
    if sample.size == 0:
        return np.full(values.shape, np.nan)
    if sample.size == 1:
        return np.full(values.shape, 0.5)

    probs = (np.arange(sample.size) + 0.5) / sample.size
    out = np.interp(values, sample, probs, left=probs[0], right=probs[-1])
    return np.clip(out, PROB_EPS, 1.0 - PROB_EPS)


def empirical_quantiles(sample: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    sample = np.sort(finite(sample))
    p = np.asarray(probabilities, dtype=float)
    if sample.size == 0:
        return np.full(p.shape, np.nan)
    if sample.size == 1:
        return np.full(p.shape, sample[0])
    probs = (np.arange(sample.size) + 0.5) / sample.size
    return np.interp(np.clip(p, probs[0], probs[-1]), probs, sample)


# =============================================================================
# 6. ADJUSTMENT METHODS FOR ONE POINT AND ONE CALENDAR MONTH
# =============================================================================


def model_wet_threshold(x_train: np.ndarray, y_train: np.ndarray, ref_threshold: float) -> float:
    x = finite(x_train)
    y = finite(y_train)
    if x.size == 0 or y.size == 0:
        return np.nan
    p_wet = float(np.mean(y > ref_threshold))
    if p_wet <= 0:
        return np.inf
    if p_wet >= 1:
        return -np.inf
    return float(np.quantile(x, 1.0 - p_wet))


def adjust_precipitation_group(
    x_apply: np.ndarray,
    x_train: np.ndarray,
    y_train: np.ndarray,
    method: Literal["linear_scaling", "eqm", "qdm"],
    ref_threshold: float,
) -> np.ndarray:
    x_apply = np.asarray(x_apply, dtype=float)
    output = np.full(x_apply.shape, np.nan)

    mask_train = np.isfinite(x_train) & np.isfinite(y_train)
    xt = np.asarray(x_train, dtype=float)[mask_train]
    yt = np.asarray(y_train, dtype=float)[mask_train]
    if xt.size < 4:
        return output

    threshold_model = model_wet_threshold(xt, yt, ref_threshold)
    wet_train_model = xt > threshold_model
    wet_train_ref = yt > ref_threshold
    xw = xt[wet_train_model]
    yw = yt[wet_train_ref]

    wet_apply = np.isfinite(x_apply) & (x_apply > threshold_model)
    output[np.isfinite(x_apply) & ~wet_apply] = 0.0
    if wet_apply.sum() == 0:
        return output
    if xw.size < 4 or yw.size < 4:
        # Conservative fallback.
        factor = np.nanmean(yw) / max(np.nanmean(xw), EPS) if xw.size and yw.size else 1.0
        output[wet_apply] = np.maximum(0.0, x_apply[wet_apply] * factor)
        return output

    values = x_apply[wet_apply]

    if method == "linear_scaling":
        factor = np.nanmean(yw) / max(np.nanmean(xw), EPS)
        corrected = values * factor

    elif method == "eqm":
        probabilities = empirical_probabilities(xw, values)
        corrected = empirical_quantiles(yw, probabilities)

    elif method == "qdm":
        # Percentile is calculated from the model distribution in the apply period.
        apply_wet_values = values[np.isfinite(values)]
        if apply_wet_values.size < MIN_APPLY_VALUES_FOR_QDM:
            probabilities = empirical_probabilities(xw, values)
            corrected = empirical_quantiles(yw, probabilities)
        else:
            probabilities = empirical_probabilities(apply_wet_values, values)
            q_hist_model = empirical_quantiles(xw, probabilities)
            q_hist_ref = empirical_quantiles(yw, probabilities)
            relative_change = values / np.maximum(q_hist_model, EPS)
            corrected = q_hist_ref * relative_change
    else:
        raise ValueError(f"Unknown precipitation method: {method}")

    # Preserve the occurrence classification after intensity correction:
    # values classified as wet must remain strictly above the reference dry
    # threshold, while dry values remain exactly zero.
    minimum_wet = np.nextafter(max(float(ref_threshold), 0.0), np.inf)
    output[wet_apply] = np.maximum(corrected, minimum_wet)
    return output


def adjust_continuous_group(
    x_apply: np.ndarray,
    x_train: np.ndarray,
    y_train: np.ndarray,
    method: str,
    kind: Literal["additive", "multiplicative"],
    bounds: tuple[float | None, float | None] = (None, None),
) -> np.ndarray:
    x_apply = np.asarray(x_apply, dtype=float)
    output = np.full(x_apply.shape, np.nan)
    train_mask = np.isfinite(x_train) & np.isfinite(y_train)
    xt = np.asarray(x_train, dtype=float)[train_mask]
    yt = np.asarray(y_train, dtype=float)[train_mask]
    valid_apply = np.isfinite(x_apply)
    values = x_apply[valid_apply]

    if xt.size < 4:
        return output

    if method == "delta":
        corrected = values + (np.nanmean(yt) - np.nanmean(xt))

    elif method == "factor":
        corrected = values * (np.nanmean(yt) / max(abs(np.nanmean(xt)), EPS))

    elif method == "eqm":
        probabilities = empirical_probabilities(xt, values)
        corrected = empirical_quantiles(yt, probabilities)

    elif method == "qdm":
        if values.size < MIN_APPLY_VALUES_FOR_QDM:
            probabilities = empirical_probabilities(xt, values)
            corrected = empirical_quantiles(yt, probabilities)
        else:
            probabilities = empirical_probabilities(values, values)
            q_hist_model = empirical_quantiles(xt, probabilities)
            q_hist_ref = empirical_quantiles(yt, probabilities)
            if kind == "additive":
                corrected = q_hist_ref + (values - q_hist_model)
            else:
                corrected = q_hist_ref * (values / np.maximum(np.abs(q_hist_model), EPS))
    else:
        raise ValueError(f"Unknown continuous method: {method}")

    lower, upper = bounds
    if lower is not None:
        corrected = np.maximum(corrected, lower)
    if upper is not None:
        corrected = np.minimum(corrected, upper)
    output[valid_apply] = corrected
    return output


# =============================================================================
# 7. DATAFRAME-LEVEL ADJUSTMENT
# =============================================================================


@dataclass(frozen=True)
class Candidate:
    method: str
    threshold: float | None = None

    @property
    def label(self) -> str:
        if self.threshold is None:
            return self.method
        return f"{self.method}__ref_threshold_{self.threshold:g}"


def candidates_for_parameter(parameter: str) -> list[Candidate]:
    if parameter == "precipitation":
        return [
            Candidate(method=m, threshold=t)
            for t in PRECIP_REFERENCE_THRESHOLDS
            for m in ["linear_scaling", "eqm", "qdm"]
        ]
    if parameter.startswith("temperature_"):
        return [Candidate("delta"), Candidate("eqm"), Candidate("qdm")]
    if parameter in {"potential_evaporation", "shortwave_radiation"}:
        return [Candidate("factor"), Candidate("eqm"), Candidate("qdm")]
    if parameter == "soil_moisture":
        return [Candidate("delta"), Candidate("eqm"), Candidate("qdm")]
    return []


def parameter_properties(parameter: str) -> tuple[str, tuple[float | None, float | None]]:
    if parameter.startswith("temperature_"):
        return "additive", (None, None)
    if parameter == "soil_moisture":
        return "additive", (0.0, 1.0)
    if parameter in {"potential_evaporation", "shortwave_radiation"}:
        return "multiplicative", (0.0, None)
    return "additive", (None, None)


def align_training(reference: pd.DataFrame, model_hist: pd.DataFrame, parameter: str) -> pd.DataFrame:
    ref = reference[reference["parameter"] == parameter].copy()
    mod = model_hist[model_hist["parameter"] == parameter].copy()
    merged = mod.merge(
        ref[["point_id", "date", "year", "month", "value"]].rename(columns={"value": "reference_value"}),
        on=["point_id", "date", "year", "month"],
        how="inner",
    )
    return merged.rename(columns={"value": "model_value"})


def apply_candidate(
    apply_df: pd.DataFrame,
    training_df: pd.DataFrame,
    parameter: str,
    candidate: Candidate,
) -> pd.DataFrame:
    output = apply_df.copy()
    output["corrected_value"] = np.nan

    # Point-month groups are used whenever enough values exist; otherwise the
    # fallback is all points for that calendar month.
    exact_groups = {
        key: group
        for key, group in training_df.groupby(["point_id", "month"])
        if len(group) >= MIN_TRAIN_VALUES_PER_POINT_MONTH
    }
    month_groups = {month: group for month, group in training_df.groupby("month")}

    for (point_id, month), group_apply in output.groupby(["point_id", "month"], sort=False):
        train = exact_groups.get((point_id, month), month_groups.get(month, training_df))
        if train is None or train.empty:
            continue

        x_train = train["model_value"].to_numpy(float)
        y_train = train["reference_value"].to_numpy(float)
        x_apply = group_apply["value"].to_numpy(float)

        if parameter == "precipitation":
            corrected = adjust_precipitation_group(
                x_apply=x_apply,
                x_train=x_train,
                y_train=y_train,
                method=candidate.method,  # type: ignore[arg-type]
                ref_threshold=float(candidate.threshold),
            )
        else:
            kind, bounds = parameter_properties(parameter)
            corrected = adjust_continuous_group(
                x_apply=x_apply,
                x_train=x_train,
                y_train=y_train,
                method=candidate.method,
                kind=kind,  # type: ignore[arg-type]
                bounds=bounds,
            )

        output.loc[group_apply.index, "corrected_value"] = corrected

    return output


# =============================================================================
# 8. DISTRIBUTIONAL METRICS
# =============================================================================


def relative_error(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), EPS)


def normalized_absolute_error(a: float, b: float, scale: float) -> float:
    """Absolute error normalized by a robust, non-zero physical scale."""
    return abs(a - b) / max(abs(scale), EPS)


def threshold_reference_precipitation(values: np.ndarray, threshold: float) -> np.ndarray:
    """Apply the candidate ERA5 dry threshold consistently during evaluation."""
    x = np.asarray(values, dtype=float).copy()
    valid = np.isfinite(x)
    x[valid & (x <= threshold)] = 0.0
    return x


def group_distribution_metrics(
    observed: np.ndarray,
    corrected: np.ndarray,
    parameter: str,
    reference_threshold: float | None,
) -> dict[str, float]:
    obs = finite(observed)
    cor = finite(corrected)
    if obs.size < 3 or cor.size < 3:
        return {}

    if parameter == "precipitation":
        threshold = float(reference_threshold or 0.0)

        # The same candidate dry threshold used to define ERA5 wet months is
        # now also applied to ERA5 values during scoring. This removes the old
        # inconsistency where ERA5 retained small positive values while the
        # corresponding corrected model values were set to zero.
        obs_eval = threshold_reference_precipitation(obs, threshold)
        cor_eval = threshold_reference_precipitation(cor, threshold)

        obs_wet = obs_eval[obs_eval > threshold]
        cor_wet = cor_eval[cor_eval > threshold]

        metrics: dict[str, float] = {
            "dry_frequency_error": abs(
                float(np.mean(cor_eval == 0.0)) - float(np.mean(obs_eval == 0.0))
            ),
            "n_observed_wet": float(obs_wet.size),
            "n_corrected_wet": float(cor_wet.size),
        }

        # A robust intensity scale avoids exploding relative errors in arid
        # point-month groups where the all-month median is zero.
        if obs_wet.size:
            wet_iqr = (
                float(np.quantile(obs_wet, 0.75) - np.quantile(obs_wet, 0.25))
                if obs_wet.size >= 4 else 0.0
            )
            wet_scale = max(
                float(np.mean(obs_wet)),
                float(np.median(obs_wet)),
                wet_iqr,
                threshold,
                EPS,
            )
        else:
            wet_scale = max(threshold, 1.0)

        full_scale = max(wet_scale, float(np.mean(obs_eval)), EPS)
        metrics["mean_error"] = normalized_absolute_error(
            float(np.mean(cor_eval)), float(np.mean(obs_eval)), full_scale
        )
        metrics["std_error"] = normalized_absolute_error(
            float(np.std(cor_eval, ddof=1)),
            float(np.std(obs_eval, ddof=1)),
            full_scale,
        )
        metrics["wasserstein_error"] = float(
            wasserstein_distance(obs_eval, cor_eval) / full_scale
        )

        # Intensity quantiles are evaluated only among wet months. This keeps
        # occurrence and intensity as two distinct parts of precipitation bias.
        if obs_wet.size >= 3 and cor_wet.size >= 3:
            metrics["wet_mean_error"] = normalized_absolute_error(
                float(np.mean(cor_wet)), float(np.mean(obs_wet)), wet_scale
            )
            metrics["wet_std_error"] = normalized_absolute_error(
                float(np.std(cor_wet, ddof=1)),
                float(np.std(obs_wet, ddof=1)),
                wet_scale,
            )
            for label, q in [("wet_q50_error", 0.50), ("wet_q90_error", 0.90), ("wet_q95_error", 0.95)]:
                metrics[label] = normalized_absolute_error(
                    float(np.quantile(cor_wet, q)),
                    float(np.quantile(obs_wet, q)),
                    wet_scale,
                )
        else:
            for label in [
                "wet_mean_error", "wet_std_error", "wet_q50_error",
                "wet_q90_error", "wet_q95_error",
            ]:
                metrics[label] = np.nan
        return metrics

    metrics = {
        "mean_error": relative_error(float(np.mean(cor)), float(np.mean(obs))),
        "std_error": relative_error(float(np.std(cor, ddof=1)), float(np.std(obs, ddof=1))),
        "q50_error": relative_error(float(np.quantile(cor, 0.50)), float(np.quantile(obs, 0.50))),
        "q90_error": relative_error(float(np.quantile(cor, 0.90)), float(np.quantile(obs, 0.90))),
        "q95_error": relative_error(float(np.quantile(cor, 0.95)), float(np.quantile(obs, 0.95))),
    }
    scale = max(float(np.quantile(obs, 0.75) - np.quantile(obs, 0.25)), abs(float(np.mean(obs))), EPS)
    metrics["wasserstein_error"] = float(wasserstein_distance(obs, cor) / scale)
    return metrics


def evaluate_distributionally(
    observed_df: pd.DataFrame,
    corrected_df: pd.DataFrame,
    parameter: str,
    reference_threshold: float | None,
) -> tuple[dict[str, float], pd.DataFrame]:
    merged = corrected_df.merge(
        observed_df[["point_id", "date", "year", "month", "value"]].rename(columns={"value": "observed_value"}),
        on=["point_id", "date", "year", "month"],
        how="inner",
    )

    rows: list[dict] = []
    for (point_id, month), group in merged.groupby(["point_id", "month"]):
        result = group_distribution_metrics(
            observed=group["observed_value"].to_numpy(float),
            corrected=group["corrected_value"].to_numpy(float),
            parameter=parameter,
            reference_threshold=reference_threshold,
        )
        if result:
            result.update({"point_id": point_id, "month": int(month)})
            rows.append(result)

    detail = pd.DataFrame(rows)
    if detail.empty:
        return {}, detail

    metric_cols = [c for c in detail.columns if c.endswith("_error")]
    summary = {c: float(detail[c].replace([np.inf, -np.inf], np.nan).median()) for c in metric_cols}
    summary["n_point_month_groups"] = int(len(detail))
    return summary, detail


# =============================================================================
# 8B. PAIRED VALIDATION METRICS
# =============================================================================


def paired_skill_metrics(
    observed: np.ndarray,
    simulated: np.ndarray,
    parameter: str,
) -> dict[str, float]:
    """
    Calculate paired time-series validation metrics.

    Notes
    -----
    1. RMSE and MAE retain the physical unit of the parameter.
    2. KGE is calculated only for positive-scale variables:
       precipitation, PET, soil moisture, and shortwave radiation.
    3. KGE is intentionally not calculated for Tmax/Tmin in degrees Celsius,
       because its mean-ratio component is not invariant to the temperature
       scale and can be misleading around 0 degC.
    4. For a free-running GCM, paired metrics are supplementary diagnostics.
       Distributional metrics remain the primary method-selection criteria.
    """
    obs = np.asarray(observed, dtype=float)
    sim = np.asarray(simulated, dtype=float)
    valid = np.isfinite(obs) & np.isfinite(sim)
    obs = obs[valid]
    sim = sim[valid]

    if obs.size < 3:
        return {}

    errors = sim - obs
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    bias = float(np.mean(errors))

    obs_mean = float(np.mean(obs))
    sim_mean = float(np.mean(sim))
    obs_std = float(np.std(obs, ddof=1))
    sim_std = float(np.std(sim, ddof=1))

    if obs_std > EPS and sim_std > EPS:
        r = float(np.corrcoef(obs, sim)[0, 1])
        if not np.isfinite(r):
            r = np.nan
    else:
        r = np.nan

    r2 = float(r ** 2) if np.isfinite(r) else np.nan

    obs_sum = float(np.sum(obs))
    pbias = (
        float(100.0 * np.sum(errors) / obs_sum)
        if abs(obs_sum) > EPS
        else np.nan
    )

    # Robust normalized errors are included because RMSE and MAE cannot be
    # compared directly across variables with different physical units.
    obs_iqr = float(np.quantile(obs, 0.75) - np.quantile(obs, 0.25))
    robust_scale = max(obs_iqr, abs(obs_mean), EPS)
    nrmse = float(rmse / robust_scale)
    nmae = float(mae / robust_scale)

    kge_applicable = parameter in {
        "precipitation",
        "potential_evaporation",
        "soil_moisture",
        "shortwave_radiation",
    }

    kge = np.nan
    alpha = np.nan
    beta = np.nan
    if (
        kge_applicable
        and np.isfinite(r)
        and obs_std > EPS
        and abs(obs_mean) > EPS
    ):
        alpha = float(sim_std / obs_std)
        beta = float(sim_mean / obs_mean)
        kge = float(
            1.0
            - np.sqrt(
                (r - 1.0) ** 2
                + (alpha - 1.0) ** 2
                + (beta - 1.0) ** 2
            )
        )

    return {
        "n_pairs": float(obs.size),
        "mae": mae,
        "rmse": rmse,
        "nmae": nmae,
        "nrmse": nrmse,
        "mean_bias": bias,
        "pbias_percent": pbias,
        "pearson_r": r,
        "r2": r2,
        "kge": kge,
        "kge_alpha": alpha,
        "kge_beta": beta,
        "kge_applicable": float(kge_applicable),
    }


def evaluate_paired_skill(
    observed_df: pd.DataFrame,
    simulated_df: pd.DataFrame,
    simulated_value_col: str,
    parameter: str,
) -> tuple[dict[str, float], pd.DataFrame]:
    """
    Compute paired metrics for each point ID and calendar month, then summarize
    them using the median across all valid point-month groups.
    """
    required_sim_cols = [
        "point_id", "date", "year", "month", simulated_value_col
    ]
    merged = simulated_df[required_sim_cols].merge(
        observed_df[
            ["point_id", "date", "year", "month", "value"]
        ].rename(columns={"value": "observed_value"}),
        on=["point_id", "date", "year", "month"],
        how="inner",
    )

    rows: list[dict] = []
    for (point_id, month), group in merged.groupby(
        ["point_id", "month"], sort=False
    ):
        metrics = paired_skill_metrics(
            observed=group["observed_value"].to_numpy(float),
            simulated=group[simulated_value_col].to_numpy(float),
            parameter=parameter,
        )
        if metrics:
            metrics.update({
                "point_id": point_id,
                "month": int(month),
            })
            rows.append(metrics)

    detail = pd.DataFrame(rows)
    if detail.empty:
        return {}, detail

    metric_cols = [
        "mae", "rmse", "nmae", "nrmse", "mean_bias",
        "pbias_percent", "pearson_r", "r2", "kge",
        "kge_alpha", "kge_beta",
    ]
    summary: dict[str, float] = {}
    for col in metric_cols:
        values = pd.to_numeric(
            detail[col], errors="coerce"
        ).replace([np.inf, -np.inf], np.nan)
        summary[col] = (
            float(values.median())
            if values.notna().any()
            else np.nan
        )

    summary["n_point_month_groups"] = int(len(detail))
    summary["n_valid_kge_groups"] = int(
        pd.to_numeric(detail["kge"], errors="coerce").notna().sum()
    )
    summary["kge_applicable"] = bool(
        parameter
        in {
            "precipitation",
            "potential_evaporation",
            "soil_moisture",
            "shortwave_radiation",
        }
    )
    return summary, detail


def prefix_metric_dict(
    metrics: dict[str, float],
    prefix: str,
) -> dict[str, float]:
    return {f"{prefix}{key}": value for key, value in metrics.items()}


def paired_improvement_metrics(
    raw_metrics: dict[str, float],
    corrected_metrics: dict[str, float],
) -> dict[str, float]:
    """Calculate improvement from raw MIROC6 to bias-corrected MIROC6."""
    result: dict[str, float] = {}

    for metric in ["rmse", "mae", "nrmse", "nmae"]:
        raw_value = raw_metrics.get(metric, np.nan)
        corrected_value = corrected_metrics.get(metric, np.nan)
        if (
            np.isfinite(raw_value)
            and np.isfinite(corrected_value)
            and abs(raw_value) > EPS
        ):
            result[f"{metric}_improvement_percent"] = float(
                100.0 * (raw_value - corrected_value) / abs(raw_value)
            )
        else:
            result[f"{metric}_improvement_percent"] = np.nan

    for metric in ["kge", "r2", "pearson_r"]:
        raw_value = raw_metrics.get(metric, np.nan)
        corrected_value = corrected_metrics.get(metric, np.nan)
        if np.isfinite(raw_value) and np.isfinite(corrected_value):
            result[f"{metric}_improvement"] = float(
                corrected_value - raw_value
            )
        else:
            result[f"{metric}_improvement"] = np.nan

    return result


# =============================================================================
# 9. BLOCKED CROSS-VALIDATION AND METHOD SELECTION
# =============================================================================


def run_blocked_cv(
    model: str,
    parameter: str,
    aligned_hist: pd.DataFrame,
    reference: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict] = []
    detail_parts: list[pd.DataFrame] = []
    scenario_name = (
        str(aligned_hist["scenario"].dropna().iloc[0])
        if "scenario" in aligned_hist.columns and aligned_hist["scenario"].notna().any()
        else "unknown"
    )

    for candidate in candidates_for_parameter(parameter):
        for fold_name, (start, end) in CV_BLOCKS.items():
            train = aligned_hist[(aligned_hist["year"] < start) | (aligned_hist["year"] > end)].copy()
            apply = aligned_hist[(aligned_hist["year"] >= start) & (aligned_hist["year"] <= end)].copy()
            if train.empty or apply.empty:
                continue

            apply_model = apply[[
                "model", "scenario", "parameter", "point_id", "lon", "lat",
                "date", "year", "month", "model_value",
            ]].rename(columns={"model_value": "value"})

            corrected = apply_candidate(
                apply_df=apply_model,
                training_df=train,
                parameter=parameter,
                candidate=candidate,
            )

            observed = reference[
                (reference["parameter"] == parameter)
                & (reference["year"] >= start)
                & (reference["year"] <= end)
            ].copy()

            metrics, detail = evaluate_distributionally(
                observed_df=observed,
                corrected_df=corrected,
                parameter=parameter,
                reference_threshold=candidate.threshold,
            )
            if not metrics:
                continue

            # Paired validation metrics for the raw and corrected series.
            raw_skill, _ = evaluate_paired_skill(
                observed_df=observed,
                simulated_df=apply_model,
                simulated_value_col="value",
                parameter=parameter,
            )
            corrected_skill, _ = evaluate_paired_skill(
                observed_df=observed,
                simulated_df=corrected,
                simulated_value_col="corrected_value",
                parameter=parameter,
            )
            skill_improvement = paired_improvement_metrics(
                raw_metrics=raw_skill,
                corrected_metrics=corrected_skill,
            )

            summary_rows.append({
                "model": model,
                "scenario": scenario_name,
                "parameter": parameter,
                "candidate": candidate.label,
                "method": candidate.method,
                "reference_threshold_mm_month": candidate.threshold,
                "fold": fold_name,
                "validation_start": start,
                "validation_end": end,
                **metrics,
                **prefix_metric_dict(raw_skill, "raw_"),
                **prefix_metric_dict(corrected_skill, "corrected_"),
                **skill_improvement,
            })

            if not detail.empty:
                detail.insert(0, "model", model)
                detail.insert(1, "scenario", scenario_name)
                detail.insert(2, "parameter", parameter)
                detail.insert(3, "candidate", candidate.label)
                detail.insert(4, "fold", fold_name)
                detail_parts.append(detail)

    return pd.DataFrame(summary_rows), pd.concat(detail_parts, ignore_index=True) if detail_parts else pd.DataFrame()


def choose_best_candidate(cv_summary: pd.DataFrame) -> pd.DataFrame:
    if cv_summary.empty:
        return pd.DataFrame()

    averaged = (
        cv_summary.groupby(
            ["model", "scenario", "parameter", "candidate", "method", "reference_threshold_mm_month"],
            dropna=False,
            as_index=False,
        )
        .mean(numeric_only=True)
    )

    selections: list[pd.DataFrame] = []
    for (model, scenario, parameter), group in averaged.groupby(["model", "scenario", "parameter"]):
        g = group.copy()
        metric_cols = [c for c in g.columns if c.endswith("_error")]

        if parameter == "precipitation":
            weights = {
                "dry_frequency_error": 0.25,
                "wet_mean_error": 0.15,
                "wet_std_error": 0.10,
                "wet_q50_error": 0.10,
                "wet_q90_error": 0.15,
                "wet_q95_error": 0.15,
                "wasserstein_error": 0.10,
            }
        else:
            weights = {
                "mean_error": 0.20,
                "std_error": 0.15,
                "q50_error": 0.15,
                "q90_error": 0.15,
                "q95_error": 0.15,
                "wasserstein_error": 0.20,
            }

        g["selection_score"] = 0.0
        total_weight = 0.0
        for metric in metric_cols:
            weight = weights.get(metric, 0.0)
            if weight <= 0:
                continue
            finite_metric = pd.to_numeric(g[metric], errors="coerce").replace([np.inf, -np.inf], np.nan)
            ranked = finite_metric.rank(method="average", ascending=True)
            # Missing wet-intensity metrics occur in very dry point-month
            # groups. Penalize them rather than allowing a NaN score to win.
            worst_rank = float(ranked.max()) + 1.0 if ranked.notna().any() else float(len(g) + 1)
            g[f"rank_{metric}"] = ranked.fillna(worst_rank)
            g["selection_score"] += weight * g[f"rank_{metric}"]
            total_weight += weight
        if total_weight > 0:
            g["selection_score"] /= total_weight

        g = g.sort_values(["selection_score", "wasserstein_error", "mean_error"]).reset_index(drop=True)
        g["selected"] = False
        g.loc[0, "selected"] = True
        selections.append(g)

    return pd.concat(selections, ignore_index=True)


# =============================================================================
# 10. FINAL FITTING, INDEPENDENT VALIDATION, FUTURE APPLICATION
# =============================================================================


def candidate_from_selection(row: pd.Series) -> Candidate:
    threshold = row.get("reference_threshold_mm_month", np.nan)
    return Candidate(
        method=str(row["method"]),
        threshold=None if pd.isna(threshold) else float(threshold),
    )


def save_long_and_wide(
    df: pd.DataFrame,
    output_csv: Path,
    output_xlsx: Path | None,
    value_col: str = "corrected_value",
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    if not SAVE_WIDE_EXCEL or output_xlsx is None:
        return
    output_xlsx.parent.mkdir(parents=True, exist_ok=True)
    temp = df.copy()
    temp["month_column"] = temp.apply(lambda r: f"{MONTH_ABBR[int(r['month'])]}_{int(r['year'])}", axis=1)
    wide = temp.pivot_table(index="point_id", columns="month_column", values=value_col, aggfunc="mean")
    date_order = sorted(
        wide.columns,
        key=lambda c: (int(str(c).split("_")[-1]), MONTH_MAP[str(c).split("_")[0].lower()]),
    )
    wide = wide[date_order].reset_index()
    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        wide.to_excel(writer, sheet_name="downscaled", index=False)


def climate_change_signal_report(
    model_hist_raw: pd.DataFrame,
    corrected_hist: pd.DataFrame,
    future_raw: pd.DataFrame,
    future_corrected: pd.DataFrame,
    parameter: str,
    model: str,
    scenario: str,
    period_name: str,
) -> pd.DataFrame:
    rows: list[dict] = []
    for point_id in sorted(set(model_hist_raw["point_id"]).intersection(future_raw["point_id"])):
        hraw = finite(model_hist_raw[model_hist_raw["point_id"] == point_id]["value"])
        hcor = finite(corrected_hist[corrected_hist["point_id"] == point_id]["corrected_value"])
        fraw = finite(future_raw[future_raw["point_id"] == point_id]["value"])
        fcor = finite(future_corrected[future_corrected["point_id"] == point_id]["corrected_value"])
        if min(len(hraw), len(hcor), len(fraw), len(fcor)) < 3:
            continue

        for q in [0.50, 0.90, 0.95]:
            hraw_q, hcor_q = np.quantile(hraw, q), np.quantile(hcor, q)
            fraw_q, fcor_q = np.quantile(fraw, q), np.quantile(fcor, q)
            if parameter.startswith("temperature_") or parameter == "soil_moisture":
                raw_change = fraw_q - hraw_q
                corrected_change = fcor_q - hcor_q
            else:
                raw_change = fraw_q / max(abs(hraw_q), EPS)
                corrected_change = fcor_q / max(abs(hcor_q), EPS)
            rows.append({
                "model": model, "scenario": scenario, "parameter": parameter,
                "period": period_name, "point_id": point_id, "quantile": q,
                "raw_change_signal": raw_change,
                "corrected_change_signal": corrected_change,
                "absolute_signal_difference": abs(corrected_change - raw_change),
            })
    return pd.DataFrame(rows)


# =============================================================================
# 11. MAIN WORKFLOW
# =============================================================================


def main() -> None:
    print("=" * 96)
    print("MIROC6 CLASSICAL METHOD SELECTION WITH PREVIOUS SPLIT EXCEL INPUT PATHS")
    print("=" * 96)
    print(f"Reference:  {REFERENCE_FOLDER}")
    print(f"Calibration: {CALIBRATION_EXCEL_DIR} | {CALIBRATION_START}-{CALIBRATION_END}")
    print(f"Validation:  {VALIDATION_EXCEL_DIR} | {VALIDATION_START}-{VALIDATION_END}")
    print(f"Future:      {FUTURE_EXCEL_DIR} | {FUTURE_START}-{FUTURE_END}")
    print("Method selection uses blocked CV only within calibration 1981-2005.")
    print("Validation 2006-2025 is used once after method selection.")
    print("=" * 96)

    print("Reading ERA5-Land reference data...")
    reference = read_reference_folder(REFERENCE_FOLDER)
    reference.to_csv(
        DIR_INVENTORY / "reference_long.csv",
        index=False,
        encoding="utf-8-sig",
    )
    reference_validation = validate_reference_dataset(reference)
    save_and_enforce_validation(reference_validation, "reference_data_validation.csv")

    run_summary: list[dict] = []
    validation_performance_rows: list[dict] = []
    all_candidate_performance_parts: list[pd.DataFrame] = []
    all_candidate_ranking_parts: list[pd.DataFrame] = []

    for model, period_roots in MODEL_PERIOD_ROOTS.items():
        print(f"\nReading split CMIP6 data for {model}...")
        cmip = read_model_root(model, period_roots)
        cmip.to_csv(
            DIR_INVENTORY / f"{normalize_text(model)}_cmip6_split_long.csv",
            index=False,
            encoding="utf-8-sig",
        )

        print(f"Validating {model} IDs, scenarios and period coverage...")
        model_validation = validate_model_dataset(cmip, reference, model)
        save_and_enforce_validation(
            model_validation,
            f"{normalize_text(model)}_data_validation.csv",
        )

        parameter_id_masks, id_alignment_summary, excluded_id_report = (
            build_id_alignment_for_model(reference, cmip, model)
        )
        id_alignment_summary.to_csv(
            DIR_INVENTORY / f"{normalize_text(model)}_ID_alignment_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )
        excluded_id_report.to_csv(
            DIR_INVENTORY / f"{normalize_text(model)}_excluded_IDs_by_parameter.csv",
            index=False,
            encoding="utf-8-sig",
        )

        scenarios = [
            scenario for scenario in REQUIRED_SCENARIOS
            if scenario in set(cmip["scenario"].dropna().astype(str))
        ]
        common_parameters = sorted(
            set(reference["parameter"])
            .intersection(cmip["parameter"])
            .intersection(SUPPORTED_PARAMETERS)
        )
        if not common_parameters:
            raise ValueError(f"No common supported parameters found for {model}.")

        all_cv_summary: list[pd.DataFrame] = []
        all_cv_detail: list[pd.DataFrame] = []
        all_selections: list[pd.DataFrame] = []

        # --------------------------------------------------------------
        # Candidate selection: separately for every SSP and parameter,
        # using only calibration data from 1981-2005.
        # --------------------------------------------------------------
        for scenario in scenarios:
            calibration_scenario = cmip[
                (cmip["period_type"] == "calibration")
                & (cmip["scenario"] == scenario)
                & cmip["year"].between(CALIBRATION_START, CALIBRATION_END)
            ].copy()

            for parameter in common_parameters:
                valid_ids = parameter_id_masks.get(parameter, set())
                if not valid_ids:
                    continue

                print(
                    f"  Blocked CV: {model} | {scenario} | {parameter} | "
                    f"{len(valid_ids)} common ID(s)"
                )
                reference_parameter = reference[
                    (reference["parameter"] == parameter)
                    & (reference["point_id"].astype(str).isin(valid_ids))
                    & reference["year"].between(CALIBRATION_START, CALIBRATION_END)
                ].copy()
                calibration_parameter = calibration_scenario[
                    (calibration_scenario["parameter"] == parameter)
                    & (calibration_scenario["point_id"].astype(str).isin(valid_ids))
                ].copy()

                aligned = align_training(
                    reference_parameter,
                    calibration_parameter,
                    parameter,
                )
                if aligned.empty:
                    continue

                cv_summary, cv_detail = run_blocked_cv(
                    model,
                    parameter,
                    aligned,
                    reference_parameter,
                )
                if cv_summary.empty:
                    continue

                selection = choose_best_candidate(cv_summary)
                all_cv_summary.append(cv_summary)
                if not cv_detail.empty:
                    all_cv_detail.append(cv_detail)
                all_selections.append(selection)

        if not all_cv_summary or not all_selections:
            raise ValueError(
                f"No classical candidate could be evaluated for {model}. "
                "Check calibration files, years and parameter names."
            )

        cv_summary_all = pd.concat(all_cv_summary, ignore_index=True)
        cv_detail_all = (
            pd.concat(all_cv_detail, ignore_index=True)
            if all_cv_detail else pd.DataFrame()
        )
        selection_all = pd.concat(all_selections, ignore_index=True)

        cv_summary_all.to_csv(
            DIR_CV / f"{normalize_text(model)}_cv_fold_metrics.csv",
            index=False,
            encoding="utf-8-sig",
        )
        if not cv_detail_all.empty:
            cv_detail_all.to_csv(
                DIR_CV / f"{normalize_text(model)}_cv_point_month_metrics.csv",
                index=False,
                encoding="utf-8-sig",
            )
        selection_all.to_csv(
            DIR_SELECTION / f"{normalize_text(model)}_candidate_ranking.csv",
            index=False,
            encoding="utf-8-sig",
        )

        performance_prefixes = ("raw_", "corrected_")
        performance_exact = {
            "model", "scenario", "parameter", "candidate", "method",
            "reference_threshold_mm_month", "selection_score", "selected",
            "rmse_improvement_percent", "mae_improvement_percent",
            "nrmse_improvement_percent", "nmae_improvement_percent",
            "kge_improvement", "r2_improvement", "pearson_r_improvement",
        }
        performance_cols = [
            col for col in selection_all.columns
            if col in performance_exact or col.startswith(performance_prefixes)
        ]
        candidate_performance = selection_all[performance_cols].copy()
        candidate_performance.to_csv(
            DIR_PERFORMANCE / f"{normalize_text(model)}_all_candidates_cv_performance.csv",
            index=False,
            encoding="utf-8-sig",
        )
        all_candidate_performance_parts.append(candidate_performance.copy())
        all_candidate_ranking_parts.append(selection_all.copy())

        selected = selection_all[selection_all["selected"].astype(bool)].copy()
        selected.to_csv(
            DIR_SELECTION / f"{normalize_text(model)}_selected_methods.csv",
            index=False,
            encoding="utf-8-sig",
        )

        # --------------------------------------------------------------
        # Final calibration fit, independent validation and future.
        # --------------------------------------------------------------
        for _, selected_row in selected.iterrows():
            scenario = str(selected_row["scenario"])
            parameter = str(selected_row["parameter"])
            candidate = candidate_from_selection(selected_row)
            valid_ids = parameter_id_masks.get(parameter, set())
            if not valid_ids:
                continue

            print(
                f"  Selected: {model} | {scenario} | {parameter} | "
                f"{candidate.label}"
            )

            reference_parameter = reference[
                (reference["parameter"] == parameter)
                & (reference["point_id"].astype(str).isin(valid_ids))
            ].copy()

            calibration_apply = cmip[
                (cmip["period_type"] == "calibration")
                & (cmip["scenario"] == scenario)
                & (cmip["parameter"] == parameter)
                & (cmip["point_id"].astype(str).isin(valid_ids))
                & cmip["year"].between(CALIBRATION_START, CALIBRATION_END)
            ].copy()
            reference_calibration = reference_parameter[
                reference_parameter["year"].between(
                    CALIBRATION_START, CALIBRATION_END
                )
            ].copy()
            aligned = align_training(
                reference_calibration,
                calibration_apply,
                parameter,
            )
            if aligned.empty:
                continue

            corrected_calibration = apply_candidate(
                calibration_apply,
                aligned,
                parameter,
                candidate,
            )
            calibration_csv = (
                DIR_CALIBRATION / normalize_text(model) / scenario / f"{parameter}.csv"
            )
            calibration_xlsx = (
                DIR_WIDE / normalize_text(model) / scenario
                / "calibration_1981_2005" / f"{parameter}.xlsx"
            )
            save_long_and_wide(
                corrected_calibration,
                calibration_csv,
                calibration_xlsx,
            )

            validation_apply = cmip[
                (cmip["period_type"] == "validation")
                & (cmip["scenario"] == scenario)
                & (cmip["parameter"] == parameter)
                & (cmip["point_id"].astype(str).isin(valid_ids))
                & cmip["year"].between(VALIDATION_START, VALIDATION_END)
            ].copy()
            corrected_validation = apply_candidate(
                validation_apply,
                aligned,
                parameter,
                candidate,
            )
            validation_csv = (
                DIR_VALIDATION / normalize_text(model) / scenario / f"{parameter}.csv"
            )
            validation_xlsx = (
                DIR_WIDE / normalize_text(model) / scenario
                / "validation_2006_2025" / f"{parameter}.xlsx"
            )
            save_long_and_wide(
                corrected_validation,
                validation_csv,
                validation_xlsx,
            )

            validation_observed = reference_parameter[
                reference_parameter["year"].between(
                    VALIDATION_START, VALIDATION_END
                )
            ].copy()
            validation_metrics, validation_detail = evaluate_distributionally(
                observed_df=validation_observed,
                corrected_df=corrected_validation,
                parameter=parameter,
                reference_threshold=candidate.threshold,
            )
            validation_raw_skill, _ = evaluate_paired_skill(
                observed_df=validation_observed,
                simulated_df=validation_apply,
                simulated_value_col="value",
                parameter=parameter,
            )
            validation_corrected_skill, _ = evaluate_paired_skill(
                observed_df=validation_observed,
                simulated_df=corrected_validation,
                simulated_value_col="corrected_value",
                parameter=parameter,
            )
            validation_improvement = paired_improvement_metrics(
                raw_metrics=validation_raw_skill,
                corrected_metrics=validation_corrected_skill,
            )

            validation_summary_row = {
                "model": model,
                "scenario": scenario,
                "parameter": parameter,
                "candidate": candidate.label,
                "calibration_start": CALIBRATION_START,
                "calibration_end": CALIBRATION_END,
                "validation_start": VALIDATION_START,
                "validation_end": VALIDATION_END,
                **validation_metrics,
                **prefix_metric_dict(validation_raw_skill, "raw_"),
                **prefix_metric_dict(validation_corrected_skill, "corrected_"),
                **validation_improvement,
            }
            pd.DataFrame([validation_summary_row]).to_csv(
                DIR_VALIDATION
                / f"{normalize_text(model)}__{scenario}__{parameter}__summary.csv",
                index=False,
                encoding="utf-8-sig",
            )
            validation_performance_rows.append(validation_summary_row)
            if not validation_detail.empty:
                validation_detail.to_csv(
                    DIR_VALIDATION
                    / f"{normalize_text(model)}__{scenario}__{parameter}__point_month.csv",
                    index=False,
                    encoding="utf-8-sig",
                )

            future_scenario = cmip[
                (cmip["period_type"] == "future")
                & (cmip["scenario"] == scenario)
                & (cmip["parameter"] == parameter)
                & (cmip["point_id"].astype(str).isin(valid_ids))
                & cmip["year"].between(FUTURE_START, FUTURE_END)
            ].copy()

            corrected_periods: list[pd.DataFrame] = []
            signal_parts: list[pd.DataFrame] = []
            for period_name, (start_year, end_year) in APPLICATION_PERIODS.items():
                apply_period = future_scenario[
                    future_scenario["year"].between(start_year, end_year)
                ].copy()
                if apply_period.empty:
                    continue
                corrected_period = apply_candidate(
                    apply_period,
                    aligned,
                    parameter,
                    candidate,
                )
                corrected_period["application_period"] = period_name
                corrected_periods.append(corrected_period)

                signal = climate_change_signal_report(
                    model_hist_raw=calibration_apply,
                    corrected_hist=corrected_calibration,
                    future_raw=apply_period,
                    future_corrected=corrected_period,
                    parameter=parameter,
                    model=model,
                    scenario=scenario,
                    period_name=period_name,
                )
                if not signal.empty:
                    signal_parts.append(signal)

            if not corrected_periods:
                continue

            corrected_future = pd.concat(corrected_periods, ignore_index=True)
            future_csv = (
                DIR_FUTURE / normalize_text(model) / scenario / f"{parameter}.csv"
            )
            future_xlsx = (
                DIR_WIDE / normalize_text(model) / scenario
                / "future_2026_2100" / f"{parameter}.xlsx"
            )
            save_long_and_wide(corrected_future, future_csv, future_xlsx)

            if signal_parts:
                pd.concat(signal_parts, ignore_index=True).to_csv(
                    DIR_SIGNAL / f"{normalize_text(model)}__{scenario}__{parameter}.csv",
                    index=False,
                    encoding="utf-8-sig",
                )

            run_summary.append({
                "model": model,
                "scenario": scenario,
                "parameter": parameter,
                "selected_candidate": candidate.label,
                "calibration_period": f"{CALIBRATION_START}-{CALIBRATION_END}",
                "validation_period": f"{VALIDATION_START}-{VALIDATION_END}",
                "future_period": f"{FUTURE_START}-{FUTURE_END}",
                "common_ID_count": len(valid_ids),
                "common_IDs": "; ".join(sorted(valid_ids)),
                "calibration_output_csv": str(calibration_csv),
                "validation_output_csv": str(validation_csv),
                "future_output_csv": str(future_csv),
            })

    run_summary_df = pd.DataFrame(run_summary)
    run_summary_df.to_csv(
        OUTPUT_ROOT / "run_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    validation_performance_df = pd.DataFrame(validation_performance_rows)
    if not validation_performance_df.empty:
        validation_performance_df.to_csv(
            DIR_PERFORMANCE
            / "miroc6_selected_methods_independent_validation_2006_2025.csv",
            index=False,
            encoding="utf-8-sig",
        )

    all_candidate_performance_df = (
        pd.concat(all_candidate_performance_parts, ignore_index=True)
        if all_candidate_performance_parts else pd.DataFrame()
    )
    all_candidate_ranking_df = (
        pd.concat(all_candidate_ranking_parts, ignore_index=True)
        if all_candidate_ranking_parts else pd.DataFrame()
    )
    selected_methods_df = (
        all_candidate_ranking_df[
            all_candidate_ranking_df["selected"].astype(bool)
        ].copy()
        if not all_candidate_ranking_df.empty else pd.DataFrame()
    )

    statistical_excel = (
        DIR_PERFORMANCE
        / "MIROC6_statistical_indices_cal1981_2005_val2006_2025.xlsx"
    )
    with pd.ExcelWriter(statistical_excel, engine="openpyxl") as writer:
        if not all_candidate_performance_df.empty:
            all_candidate_performance_df.to_excel(
                writer, sheet_name="All_Methods_CV", index=False
            )
        if not all_candidate_ranking_df.empty:
            all_candidate_ranking_df.to_excel(
                writer, sheet_name="Candidate_Ranking", index=False
            )
        if not selected_methods_df.empty:
            selected_methods_df.to_excel(
                writer, sheet_name="Selected_Methods", index=False
            )
        if not validation_performance_df.empty:
            validation_performance_df.to_excel(
                writer, sheet_name="Validation_2006_2025", index=False
            )
        if not run_summary_df.empty:
            run_summary_df.to_excel(writer, sheet_name="Run_Summary", index=False)

        guide_rows = [
            ["Column or metric", "Interpretation"],
            ["corrected_rmse", "Lower is better; physical unit of the parameter"],
            ["corrected_mae", "Lower is better; physical unit of the parameter"],
            ["corrected_r2", "Closer to 1 is better"],
            ["corrected_pearson_r", "Closer to 1 is better"],
            ["corrected_kge", "Closer to 1 is better; not calculated for Tmax/Tmin in degC"],
            ["corrected_pbias_percent", "Absolute value closer to 0 is better"],
            ["selection_score", "Lower is better; calculated only from calibration CV"],
            ["selected", "TRUE identifies the final method for that scenario and parameter"],
        ]
        pd.DataFrame(guide_rows[1:], columns=guide_rows[0]).to_excel(
            writer, sheet_name="Guide", index=False
        )

        for worksheet in writer.book.worksheets:
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions
            for column_cells in worksheet.columns:
                letter = column_cells[0].column_letter
                max_length = 0
                for cell in column_cells[:300]:
                    value = "" if cell.value is None else str(cell.value)
                    max_length = max(max_length, len(value))
                worksheet.column_dimensions[letter].width = min(
                    max(max_length + 2, 11), 32
                )

    metrics_guide = f"""PERFORMANCE METRICS GUIDE

Calibration and method selection: {CALIBRATION_START}-{CALIBRATION_END}
Independent validation: {VALIDATION_START}-{VALIDATION_END}
Future application: {FUTURE_START}-{FUTURE_END}

Candidate ranking is based only on blocked cross-validation inside calibration.
Validation data are not used to select a method or tune the dry threshold.

Main workbook:
{statistical_excel}
"""
    (DIR_PERFORMANCE / "performance_metrics_guide.txt").write_text(
        metrics_guide, encoding="utf-8"
    )

    settings = {
        "reference_period": [REFERENCE_START, REFERENCE_END],
        "calibration_method_selection_period": [CALIBRATION_START, CALIBRATION_END],
        "independent_validation_period": [VALIDATION_START, VALIDATION_END],
        "future_analysis": [FUTURE_START, FUTURE_END],
        "cv_blocks_inside_calibration": CV_BLOCKS,
        "future_application_periods": APPLICATION_PERIODS,
        "precipitation_reference_thresholds": PRECIP_REFERENCE_THRESHOLDS,
        "strict_data_validation": STRICT_DATA_VALIDATION,
        "ID_alignment_mode": ID_ALIGNMENT_MODE,
        "minimum_common_IDs": MIN_COMMON_IDS,
        "required_scenarios": REQUIRED_SCENARIOS,
        "target_units": TARGET_UNITS,
        "reference_input_units": REFERENCE_INPUT_UNITS,
        "model_input_units": MODEL_INPUT_UNITS,
        "reference_folder": str(REFERENCE_FOLDER),
        "cmip6_split_excel_root": str(CMIP6_SPLIT_EXCEL_ROOT),
        "model_period_roots": {
            model: {period: str(path) for period, path in roots.items()}
            for model, roots in MODEL_PERIOD_ROOTS.items()
        },
    }
    with open(OUTPUT_ROOT / "settings_used.json", "w", encoding="utf-8") as file:
        json.dump(settings, file, indent=2, ensure_ascii=False)

    print("\nWorkflow finished successfully.")
    print(f"Results: {OUTPUT_ROOT}")
    print(f"Statistical-index workbook: {statistical_excel}")


# =============================================================================
# 12. SMALL INTERNAL SELF-TEST
# =============================================================================


def _self_test() -> None:
    rng = np.random.default_rng(42)
    years = np.arange(CALIBRATION_START, CALIBRATION_END + 1)
    rows_ref = []
    rows_mod = []
    for point_id in ["1", "2"]:
        for year in years:
            for month in range(1, 13):
                base = max(0.0, 20 + 15 * math.cos((month - 1) / 12 * 2 * math.pi))
                obs = max(0.0, rng.gamma(1.5, max(base, 1) / 1.5) - 3)
                mod = max(0.0, obs * 0.75 + rng.normal(1.5, 2.0))
                date = pd.Timestamp(year, month, 1)
                rows_ref.append({"parameter": "precipitation", "point_id": point_id, "date": date, "year": year, "month": month, "value": obs, "lon": 0, "lat": 0})
                rows_mod.append({"model": "TEST", "scenario": "historical", "parameter": "precipitation", "point_id": point_id, "date": date, "year": year, "month": month, "value": mod, "lon": 0, "lat": 0})
    ref = pd.DataFrame(rows_ref)
    mod = pd.DataFrame(rows_mod)
    aligned = align_training(ref, mod, "precipitation")
    cv, _ = run_blocked_cv("TEST", "precipitation", aligned, ref)
    assert not cv.empty
    selected = choose_best_candidate(cv)
    assert selected["selected"].sum() == 1
    print("Self-test passed.")


if __name__ == "__main__":
    # Enable to evaluate the statistical routines using synthetic test inputs.
    RUN_SELF_TEST = False
    if RUN_SELF_TEST:
        _self_test()
    else:
        main()