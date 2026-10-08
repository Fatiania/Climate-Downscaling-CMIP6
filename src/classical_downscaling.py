# ============================================================
# MULTI-PARAMETER CMIP6 DOWNSCALING WORKFLOW FOR MOROCCO
# CLASSICAL MONTHLY METHODS ONLY
# MIROC6 ONLY + PET DERIVED FROM TEMPERATURE + ROBUST PET/SHORTWAVE NAME MATCHING
# IMPORTANT PET RULE: Reference file "Potential evaporation (mm)" and CMIP6 files named "potential_evaporation" are forced to the same parameter: potential_evaporation. Actual CMIP6 evaporation/evspsbl is ignored.
#
# Implemented downscaling approaches:
#   1) Precipitation/runoff: zero-inflated two-part monthly correction
#      - wet/dry occurrence correction
#      - wet-month intensity correction
#   2) Temperature variables: additive monthly delta correction
#   3) potential evaporation/soil moisture/groundwater/radiation/positive variables:
#      multiplicative monthly factor correction
#      IMPORTANT: negative reference evaporation values are converted to positive
#      before calibration/validation/future correction.
#
# Removed from the previous workflow:
#   - Quantile Mapping
#   - Quantile Delta Mapping
#   - Machine Learning models
#   - Extra generic scaling-method comparison
#
# Periods:
#   Calibration: 1981-2005
#   Validation:  2006-2025
#   Future:      2026-2100
#
# Important:
#   - CMIP6 CSV files are read using converted_value first, when available.
#   - Reference and CMIP6 are merged by point_id + year + month.
#   - All corrections are calculated separately for each point_id and calendar month.
# ============================================================

from pathlib import Path
import re
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================
# 1. INPUT PATHS
# ============================================================

REFERENCE_FOLDER = Path(
    r"D:\Morocco\morocco_data\Morocco excel data"
)

HISTORICAL_ROOT = Path(
    r"D:\Morocco\morocco_data\CMIP6_historical_1950_2025_393_points"
)

FUTURE_ROOT = Path(
    r"D:\Morocco\morocco_data\CMIP6_future_2015_2100_393_points"
)

HISTORICAL_MONTHLY_DIR = HISTORICAL_ROOT / "02_extracted_monthly_csv"
FUTURE_MONTHLY_DIR = FUTURE_ROOT / "02_extracted_monthly_csv"


# ============================================================
# 2. OUTPUT PATHS
# ============================================================

OUT_ROOT = Path(
    r"D:\Morocco\morocco_data\Downscaling_CLASSICAL_MIROC6_ONLY_DERIVED_PET"
)

STANDARD_DIR = OUT_ROOT / "01_standardized_inputs"
QUALITY_DIR = OUT_ROOT / "02_downscaling_quality"
FUTURE_DS_DIR = OUT_ROOT / "03_downscaled_future_monthly_csv"
CHANGE_DIR = OUT_ROOT / "04_future_change_analysis"
EXCEL_LIKE_DIR = OUT_ROOT / "05_excel_like_future_outputs"
LOG_DIR = OUT_ROOT / "06_logs"

for folder in [
    OUT_ROOT, STANDARD_DIR, QUALITY_DIR, FUTURE_DS_DIR,
    CHANGE_DIR, EXCEL_LIKE_DIR, LOG_DIR
]:
    folder.mkdir(parents=True, exist_ok=True)


# ============================================================
# 3. SCIENTIFIC SETTINGS
# ============================================================

REFERENCE_START_YEAR = 1981
REFERENCE_END_YEAR = 2025

COMMON_START_YEAR = 1981
COMMON_END_YEAR = 2025

CALIBRATION_START_YEAR = 1981
CALIBRATION_END_YEAR = 2005
VALIDATION_START_YEAR = 2006
VALIDATION_END_YEAR = 2025

# Baseline used for future-change analysis.
# This is intentionally the calibration/reference period, while 2006-2025
# remains an independent validation period.
BASELINE_START_YEAR = CALIBRATION_START_YEAR
BASELINE_END_YEAR = CALIBRATION_END_YEAR

FUTURE_START_YEAR = 2026
FUTURE_END_YEAR = 2100

EXPECTED_POINTS = 393

# Minimum values needed to calculate a correction for one point_id and one month.
# If fewer values exist, the code falls back to all points for that month.
MIN_VALUES_PER_POINT_MONTH = 10

# Minimum rows needed for calibration and validation.
MIN_CALIBRATION_ROWS = 1000
MIN_VALIDATION_ROWS = 1000

# Dry/wet thresholds for zero-inflated precipitation/runoff correction.
PRECIP_DRY_THRESHOLD_MM_MONTH = 1.0
RUNOFF_DRY_THRESHOLD_MM_MONTH = 0.01

# If a point-month is always dry in observations, normal future model noise is set to zero.
# If the future model value is larger than the maximum historical model value for that same
# point-month, it can be retained as an extreme future signal.
DRY_MONTH_POLICY = "allow_extreme_future"  # options: "always_zero", "allow_extreme_future"

# Numerical safety.
EPS = 1e-12

# Models to run. The final main() runs each model separately and saves each model in its own folder.
# The code normalizes names, so MIROC6 and miroc6 are treated the same.
SELECTED_MODELS = {"miroc6"}

# Explicit filename-to-variable mapping:
#   Reference Excel/file/sheet: Potential evaporation (mm)
#   CMIP6 CSV/Excel files:    potential_evaporation__historical__cnrm_cm6_1__...
#                            potential_evaporation__ssp...__cnrm_cm6_1__...
# Both are standardized internally as: potential_evaporation
FORCE_REFERENCE_PET_NAMES = {
    "potential_evaporation",
    "potential_evaporation_mm",
    "potential_evaporation_millimeter",
    "potential_evaporation_millimeters",
    "potential_evaporation_mm_month",
    "potential_evaporation_mm_monthly",
}
FORCE_CMIP6_PET_NAMES = {
    "potential_evaporation",
    "potential_evaporation_mm",
    "potential_evapotranspiration",
    "pet",
    "eto",
    "et0",
}


def normalize_model_name(model_name):
    """Normalize CMIP6 model names for safe matching."""
    return normalize_text(str(model_name))


def clean_numeric_text(value):
    """
    Convert messy numeric values to float.

    IMPORTANT for this Morocco workflow:
    commas are treated as thousands separators, NOT as European decimal commas.
    Examples:
      '-1,103.336' -> -1103.336
      '1,234'      -> 1234
      '12.5'       -> 12.5
      '(4.5)'      -> -4.5
      '−2.1'       -> -2.1

    This intentionally does NOT convert '12,5' to 12.5. If such values exist,
    they will be interpreted as 125 because comma is assumed to be a thousands separator.
    """
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    s = str(value).strip()
    if s == "":
        return np.nan

    s = s.replace("−", "-").replace("–", "-").replace("—", "-")
    s = s.replace("\u00a0", " ").strip()

    # Accounting-style negative numbers: (12.3) -> -12.3
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1]

    # Remove common missing-value strings.
    if s.lower() in {"nan", "none", "null", "na", "n/a", "missing", "--", "-"}:
        return np.nan

    # User-confirmed rule: commas are not European decimals; remove all commas.
    s = s.replace(",", "")

    # Keep only the first numeric token after cleaning.
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    if not m:
        return np.nan

    try:
        out = float(m.group(0))
        return -out if neg else out
    except Exception:
        return np.nan

def to_numeric_clean_series(series):
    """Robust numeric conversion for all reference and CMIP6 values."""
    return series.apply(clean_numeric_text)

# Save Excel-like outputs in the old wide monthly sheet format.
SAVE_EXCEL_LIKE_OUTPUTS = True

# Future periods for change analysis.
FUTURE_PERIODS = {
    "near_future_2026_2040": (2026, 2040),
    "mid_future_2041_2060": (2041, 2060),
    "far_future_2061_2080": (2061, 2080),
    "end_century_2081_2100": (2081, 2100),
}


# ============================================================
# 4. PARAMETER MAPPING AND METHOD CHOICE
# ============================================================

PARAMETER_RULES = {
    "precipitation": {
        "excel_keywords": ["precip", "rain", "rainfall"],
        "cmip6_aliases": ["precipitation", "pr", "rainfall", "rain"],
        "bounds": {"min": 0.0, "max": None},
    },
    "runoff_total": {
        "excel_keywords": ["runoff", "mrro", "total runoff"],
        "cmip6_aliases": ["runoff_total", "runoff", "mrro", "total_runoff"],
        "bounds": {"min": 0.0, "max": None},
    },
    "soil_moisture": {
        "excel_keywords": ["soil", "moisture", "mrsos", "ssm", "sm"],
        "cmip6_aliases": [
            "soil_moisture", "soil_moisture_upper", "soil_moisture(-)",
            "mrsos", "ssm", "sm", "surface_soil_moisture_m3_m3"
        ],
        "bounds": {"min": 0.0, "max": 1.0},
    },
    "temperature_max": {
        "excel_keywords": ["temperature max", "temperature_max", "tasmax", "tmax", "max temp"],
        "cmip6_aliases": ["temperature_max", "tasmax", "tmax"],
        "bounds": {"min": None, "max": None},
    },
    "temperature_min": {
        "excel_keywords": ["temperature min", "temperature_min", "tasmin", "tmin", "min temp"],
        "cmip6_aliases": ["temperature_min", "tasmin", "tmin"],
        "bounds": {"min": None, "max": None},
    },
    "temperature_mean": {
        "excel_keywords": ["temperature mean", "temperature_mean", "tas", "tmean", "mean temp"],
        "cmip6_aliases": ["temperature_mean", "tas", "tmean", "temperature"],
        "bounds": {"min": None, "max": None},
    },
    "shortwave_radiation": {
        # Reference names such as "Shortwave radiation (J.m-2)" are treated as shortwave_radiation.
        "excel_keywords": [
            "shortwave radiation", "Shortwave radiation (J.m-2)", "shortwave_radiation",
            "short wave radiation", "short_wave_radiation", "surface shortwave radiation",
            "surface_downwelling_shortwave_radiation", "downwelling shortwave",
            "solar radiation", "surface solar radiation", "rsds", "radiation"
        ],
        "cmip6_aliases": [
            "shortwave_radiation", "surface_downwelling_shortwave_radiation",
            "downwelling_shortwave_radiation", "surface_downwelling_shortwave",
            "surface_solar_radiation", "solar_radiation", "rsds"
        ],
        "bounds": {"min": 0.0, "max": None},
    },
    "potential_evaporation": {
        # The reference file may be named PET, potential evaporation, or even evaporation.
        # In this workflow it is treated as PET/potential evaporation.
        "excel_keywords": [
            "potential evaporation", "Potential evaporation (mm)", "potential_evaporation",
            "potential_evaporation_mm", "potential evaporation mm", "potential evap",
            "potential evapotranspiration", "potential_evapotranspiration",
            "pet", "eto", "et0", "reference evaporation",
            "reference_evaporation", "reference evapotranspiration",
            # Dataset convention: generic evaporation labels represent potential evaporation.
            # This mapping is specific to the configured reference dataset.
            "evaporation", "evapor"
        ],
        # IMPORTANT: do not include actual CMIP6 evaporation aliases such as evspsbl here.
        # CMIP6 PET must come from files derived previously, e.g. potential_evaporation__ssp...csv.
        "cmip6_aliases": [
            "potential_evaporation", "potential_evaporation_mm", "potential_evapotranspiration", "pet", "eto", "et0",
            "derived_pet", "derived_pet_hargreaves", "derived_from_existing_temperature_files"
        ],
        "bounds": {"min": 0.0, "max": None},
    },
    "groundwater": {
        "excel_keywords": ["groundwater", "ground water", "gwl", "water table"],
        "cmip6_aliases": ["groundwater", "ground_water", "gwl", "water_table"],
        "bounds": {"min": None, "max": None},
    },
}


ZERO_INFLATED_PARAMETERS = {"precipitation", "runoff_total"}
ADDITIVE_PARAMETERS = {"temperature_max", "temperature_min", "temperature_mean"}
MULTIPLICATIVE_PARAMETERS = {
    "soil_moisture", "shortwave_radiation", "potential_evaporation", "groundwater"
}


def get_downscaling_method(parameter):
    if parameter in ZERO_INFLATED_PARAMETERS:
        return "zero_inflated_two_part"
    if parameter in ADDITIVE_PARAMETERS:
        return "additive_monthly_delta"
    if parameter in MULTIPLICATIVE_PARAMETERS:
        return "multiplicative_monthly_factor"
    # Safe default for unknown positive hydrological variables.
    return "multiplicative_monthly_factor"


def get_dry_threshold(parameter):
    if parameter == "precipitation":
        return PRECIP_DRY_THRESHOLD_MM_MONTH
    if parameter == "runoff_total":
        return RUNOFF_DRY_THRESHOLD_MM_MONTH
    return 0.0


# ============================================================
# 5. LOGGING
# ============================================================

LOG_FILE = LOG_DIR / "classical_downscaling_log.csv"
log_rows = []


def now_text():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def print_info(message):
    print(f"[{now_text()}] {message}")


def add_log(step, status, message="", parameter="", model="", scenario="", method="", file_path=""):
    row = {
        "time": now_text(),
        "step": step,
        "status": status,
        "parameter": parameter,
        "model": model,
        "scenario": scenario,
        "method": method,
        "message": str(message),
        "file_path": str(file_path),
    }
    log_rows.append(row)
    pd.DataFrame(log_rows).to_csv(LOG_FILE, index=False, encoding="utf-8-sig")


# ============================================================
# ID COVERAGE REPORTS
# ============================================================

ID_COVERAGE_FILE = LOG_DIR / "id_coverage_report_by_stage.csv"
MISSING_ID_FILE = LOG_DIR / "missing_id_details_by_stage.csv"
id_coverage_rows = []
missing_id_rows = []


def save_id_reports():
    if id_coverage_rows:
        pd.DataFrame(id_coverage_rows).to_csv(ID_COVERAGE_FILE, index=False, encoding="utf-8-sig")
    if missing_id_rows:
        pd.DataFrame(missing_id_rows).to_csv(MISSING_ID_FILE, index=False, encoding="utf-8-sig")


def record_id_coverage_stage(parameter, model, scenario, method, stage, reference_ids, data_ids, data_rows=0):
    """Save point-ID coverage information for one workflow stage."""
    ref_set = set(clean_id(x) for x in reference_ids if clean_id(x) != "")
    data_set = set(clean_id(x) for x in data_ids if clean_id(x) != "")

    common_set = ref_set.intersection(data_set)
    missing_from_data = sorted(ref_set - data_set, key=lambda x: (len(str(x)), str(x)))
    extra_in_data = sorted(data_set - ref_set, key=lambda x: (len(str(x)), str(x)))

    id_coverage_rows.append({
        "time": now_text(),
        "parameter": parameter,
        "model": model,
        "scenario": scenario,
        "method": method,
        "stage": stage,
        "reference_id_count": len(ref_set),
        "data_id_count": len(data_set),
        "common_id_count": len(common_set),
        "data_rows": int(data_rows) if pd.notna(data_rows) else data_rows,
        "missing_reference_ids_in_data_count": len(missing_from_data),
        "extra_data_ids_not_in_reference_count": len(extra_in_data),
        "complete_expected_points": len(common_set) == EXPECTED_POINTS,
        "first_missing_reference_ids": ";".join(missing_from_data[:30]),
        "first_extra_data_ids": ";".join(extra_in_data[:30]),
    })

    for mid in missing_from_data:
        missing_id_rows.append({
            "time": now_text(),
            "parameter": parameter,
            "model": model,
            "scenario": scenario,
            "method": method,
            "stage": stage,
            "missing_type": "reference_id_missing_in_data",
            "point_id": mid,
        })

    for eid in extra_in_data:
        missing_id_rows.append({
            "time": now_text(),
            "parameter": parameter,
            "model": model,
            "scenario": scenario,
            "method": method,
            "stage": stage,
            "missing_type": "data_id_not_in_reference",
            "point_id": eid,
        })

    save_id_reports()



# ============================================================
# 6. MONTH AND NAME HELPERS
# ============================================================

MONTH_ABBR_TO_NUM = {
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

MONTH_NUM_TO_ABBR = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
    5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
    9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}


def normalize_text(text):
    text = str(text).lower().strip()
    text = text.replace("-", "_")
    text = text.replace(" ", "_")
    text = re.sub(r"[^a-z0-9_]+", "", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")



def normalize_parameter_name(name):
    """Return the standard internal parameter name.

    This fixes name matching for reference/CMIP6 files with different styles, e.g.:
      Potential evaporation (mm) -> potential_evaporation
      potential_evaporation      -> potential_evaporation
      Shortwave radiation (J.m-2)-> shortwave_radiation
    It also prevents actual evaporation / evspsbl from being treated as PET.
    """
    raw = str(name)
    n = normalize_text(raw)

    # Potential evaporation / PET from reference or CMIP6-derived files.
    if (
        n in {
            "potential_evaporation", "potential_evaporation_mm", "potential_evaporation_m",
            "potential_evaporation_j", "potential_evaporation_jm2", "potential_evaporation_j_m2",
            "potential_evapotranspiration", "potential_evapotranspiration_mm",
            "pet", "eto", "et0"
        }
        or ("potential" in n and ("evaporation" in n or "evapotranspiration" in n or "evap" in n))
    ):
        return "potential_evaporation"

    # Shortwave radiation names.
    if (
        n in {
            "shortwave_radiation", "shortwave_radiation_jm2", "shortwave_radiation_j_m2",
            "surface_downwelling_shortwave_radiation", "downwelling_shortwave_radiation",
            "surface_solar_radiation", "solar_radiation", "rsds"
        }
        or ("shortwave" in n and "radiation" in n)
        or ("solar" in n and "radiation" in n)
    ):
        return "shortwave_radiation"

    # Actual evaporation is NOT PET. Keep it separate so it can be ignored.
    if n in {"evaporation", "evspsbl", "actual_evaporation", "evaporation_including_sublimation_and_transpiration"}:
        return "evaporation"

    if n in {"precipitation", "precip", "rain", "rainfall", "pr"} or "precip" in n:
        return "precipitation"
    if n in {"runoff_total", "runoff", "total_runoff", "mrro"} or "runoff" in n:
        return "runoff_total"
    if ("soil" in n and "moist" in n) or n in {"soil_moisture", "soil_moisture_upper", "mrsos", "sm", "ssm", "surface_soil_moisture_m3_m3"}:
        return "soil_moisture"
    if n in {"temperature_max", "tasmax", "tmax", "max_temperature", "maximum_temperature"} or (("temp" in n or "temperature" in n) and "max" in n):
        return "temperature_max"
    if n in {"temperature_min", "tasmin", "tmin", "min_temperature", "minimum_temperature"} or (("temp" in n or "temperature" in n) and "min" in n):
        return "temperature_min"
    if n in {"temperature_mean", "tas", "tmean", "mean_temperature", "temperature"} or (("temp" in n or "temperature" in n) and "mean" in n):
        return "temperature_mean"
    if "ground" in n and "water" in n:
        return "groundwater"

    return n

def safe_name(name):
    name = str(name)
    name = re.sub(r'[\\/*?:"<>|]', "_", name)
    name = name.strip()
    return name if name else "output"


def month_col_name(year, month):
    return f"{MONTH_NUM_TO_ABBR[int(month)]}_{int(year)}"


def parse_month_column(col):
    """
    Detect monthly columns like:
      Jan_1981, January 1981, Jan-1981, 1981-01, 1981_01
    Also accepts Excel/pandas datetime columns.
    """
    if isinstance(col, (pd.Timestamp, datetime)):
        return int(col.year), int(col.month)

    text = str(col).strip()

    m = re.match(r"^([A-Za-z]+)[_\s\-]+(\d{4})$", text)
    if m:
        month_text = m.group(1).lower()
        year = int(m.group(2))
        if month_text in MONTH_ABBR_TO_NUM:
            return year, MONTH_ABBR_TO_NUM[month_text]

    m = re.match(r"^(\d{4})[_\s\-/]+(\d{1,2})$", text)
    if m:
        year = int(m.group(1))
        month = int(m.group(2))
        if 1 <= month <= 12:
            return year, month

    return None


def make_month_columns(start_year, end_year):
    return [
        month_col_name(year, month)
        for year in range(start_year, end_year + 1)
        for month in range(1, 13)
    ]


FUTURE_MONTH_COLUMNS = make_month_columns(FUTURE_START_YEAR, FUTURE_END_YEAR)


def clean_id(value):
    if pd.isna(value):
        return ""
    try:
        x = float(value)
        if x.is_integer():
            return str(int(x))
    except Exception:
        pass
    return str(value).strip()


def find_column(df, possible_names):
    lower_map = {str(c).strip().lower(): c for c in df.columns}
    for name in possible_names:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return None


def infer_reference_parameter_from_excel_name(excel_path):
    stem = normalize_text(excel_path.stem)

    if "soil" in stem and "moist" in stem:
        return "soil_moisture"
    if ("temperature" in stem or "temp" in stem) and ("max" in stem or "tasmax" in stem or "tmax" in stem):
        return "temperature_max"
    if ("temperature" in stem or "temp" in stem) and ("min" in stem or "tasmin" in stem or "tmin" in stem):
        return "temperature_min"
    if ("temperature" in stem or "temp" in stem) and ("mean" in stem or "tas" in stem or "tmean" in stem):
        return "temperature_mean"
    if (
        "shortwave" in stem
        or "short_wave" in stem
        or "rsds" in stem
        or ("solar" in stem and "radiation" in stem)
        or ("radiation" in stem and "longwave" not in stem and "long_wave" not in stem)
    ):
        # This catches reference names such as: Shortwave radiation (J.m-2)
        return "shortwave_radiation"
    if "ground" in stem and "water" in stem:
        return "groundwater"

    # PET / potential evaporation MUST be detected before generic evaporation words.
    # In this dataset, the field labeled "Evaporation" represents potential evaporation.
    if (
        stem == "pet"
        or "potential_evaporation" in stem
        or "potential_evapotranspiration" in stem
        or ("potential" in stem and "evap" in stem)
        or "reference_evaporation" in stem
        or "reference_evapotranspiration" in stem
        or "eto" in stem
        or "et0" in stem
        or "evaporation" in stem
        or "evapor" in stem
    ):
        return "potential_evaporation"

    for parameter, rule in PARAMETER_RULES.items():
        for keyword in rule["excel_keywords"]:
            if normalize_text(keyword) in stem:
                return parameter

    return None


def parameter_matches_cmip6(reference_parameter, cmip6_parameter):
    p = normalize_text(cmip6_parameter)

    # Actual CMIP6 evaporation (evaporation/evspsbl) is NOT PET.
    # It must not be used for potential_evaporation downscaling.
    actual_evap_names = {"evaporation", "evap", "evspsbl", "evapotranspiration"}

    if reference_parameter == "potential_evaporation":
        if p in actual_evap_names or p.startswith("evaporation") or p.startswith("evspsbl"):
            return False
        return (
            p in {"potential_evaporation", "potential_evapotranspiration", "pet", "eto", "et0"}
            or "potential_evap" in p
            or "derived_pet" in p
        )

    if reference_parameter not in PARAMETER_RULES:
        return p == normalize_text(reference_parameter)

    aliases = [normalize_text(a) for a in PARAMETER_RULES[reference_parameter]["cmip6_aliases"]]
    if p in aliases:
        return True

    if reference_parameter == "soil_moisture":
        return "soil" in p and "moist" in p
    if reference_parameter == "precipitation":
        return "precip" in p or p in {"pr", "rain", "rainfall"}
    if reference_parameter == "runoff_total":
        return "runoff" in p or p == "mrro"
    if reference_parameter == "shortwave_radiation":
        return (
            p == "rsds"
            or "shortwave" in p
            or "short_wave" in p
            or "surface_downwelling_shortwave" in p
            or "downwelling_shortwave" in p
            or ("solar" in p and "radiation" in p)
            or p in {"solar_radiation", "surface_solar_radiation"}
        )
    if reference_parameter == "temperature_max":
        return "max" in p or p == "tasmax" or p == "tmax"
    if reference_parameter == "temperature_min":
        return "min" in p or p == "tasmin" or p == "tmin"
    if reference_parameter == "temperature_mean":
        return p in {"tas", "tmean", "temperature", "temperature_mean"}
    if reference_parameter == "groundwater":
        return "ground" in p and "water" in p

    return False


def clip_parameter_values(parameter, values):
    values = np.asarray(values, dtype=float)
    rule = PARAMETER_RULES.get(parameter)
    if not rule:
        return values

    bounds = rule["bounds"]
    vmin = bounds.get("min")
    vmax = bounds.get("max")

    if vmin is not None:
        values = np.maximum(values, vmin)
    if vmax is not None:
        values = np.minimum(values, vmax)

    return values


def extract_year_range_from_text(text):
    m = re.search(r"(\d{4})_(\d{4})", str(text))
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


# ============================================================
# 7. READ REFERENCE EXCEL FILES
# ============================================================

def apply_reference_value_conventions(long_df, parameter, excel_path=None, sheet_name=None):
    """
    Applies parameter-specific conventions to reference Excel values only.

    PET note:
    Some reference potential evaporation / PET products store evaporation as negative fluxes.
    In this workflow PET is downscaled as a positive water-loss amount,
    so negative reference PET values are converted to positive using abs().

    CMIP6 values are not modified here. CMIP6 PET must already be derived
    from temperature files and saved as potential_evaporation, not evspsbl.
    """
    if long_df.empty:
        return long_df

    if parameter == "potential_evaporation":
        values = to_numeric_clean_series(long_df["reference_value"])
        negative_count = int((values < 0).sum())
        positive_count = int((values > 0).sum())
        zero_count = int((values == 0).sum())

        long_df = long_df.copy()
        long_df["reference_value_original_sign"] = values
        long_df["reference_value"] = values.abs()

        print_info(
            f"Converted reference potential evaporation/PET to positive values using abs() | "
            f"file={Path(excel_path).name if excel_path is not None else ''} | "
            f"sheet={sheet_name if sheet_name is not None else ''} | "
            f"negative={negative_count}, positive={positive_count}, zero={zero_count}"
        )
        add_log(
            step="reference_evaporation_sign_conversion",
            status="done",
            parameter=parameter,
            message=(
                f"Converted reference potential evaporation/PET to positive with abs(); "
                f"negative={negative_count}; positive={positive_count}; zero={zero_count}"
            ),
            file_path=excel_path if excel_path is not None else "",
        )

    return long_df

def read_one_reference_excel(excel_path, parameter):
    """
    Converts one reference Excel file from wide format to long format.

    Expected shape:
      ID | optional metadata/coordinates | Jan_1981 | Feb_1981 | ... | Dec_2025

    Output:
      parameter | point_id | date | year | month | reference_value
    """
    xls = pd.ExcelFile(excel_path)
    long_parts = []
    template_sheets = {}

    for sheet_name in xls.sheet_names:
        if str(sheet_name).lower() == "summary":
            continue

        df = pd.read_excel(excel_path, sheet_name=sheet_name)
        if df.empty or df.shape[1] < 2:
            continue

        month_cols = []
        month_info = {}
        for col in df.columns:
            parsed = parse_month_column(col)
            if parsed is None:
                continue
            year, month = parsed
            if REFERENCE_START_YEAR <= year <= REFERENCE_END_YEAR:
                month_cols.append(col)
                month_info[col] = (year, month)

        if not month_cols:
            print_info(f"No monthly columns detected in {excel_path.name} | sheet {sheet_name}")
            continue

        first_month_index = min(df.columns.get_loc(c) for c in month_cols)
        meta_cols = list(df.columns[:first_month_index])
        if not meta_cols:
            meta_cols = [df.columns[0]]

        id_col = meta_cols[0]

        template = df[meta_cols].copy()
        template["__point_id"] = template[id_col].apply(clean_id)
        template_sheets[sheet_name] = template

        temp = df[[id_col] + month_cols].copy()
        temp = temp.rename(columns={id_col: "point_id"})
        temp["point_id"] = temp["point_id"].apply(clean_id)

        long = temp.melt(
            id_vars=["point_id"],
            value_vars=month_cols,
            var_name="month_column",
            value_name="reference_value",
        )

        long["year"] = long["month_column"].map(lambda c: month_info[c][0])
        long["month"] = long["month_column"].map(lambda c: month_info[c][1])
        long["date"] = pd.to_datetime(
            long["year"].astype(str) + "-" + long["month"].astype(str).str.zfill(2) + "-01"
        )
        long["reference_value"] = to_numeric_clean_series(long["reference_value"])
        long["parameter"] = parameter
        long["sheet_name"] = sheet_name

        # Parameter-specific reference conventions.
        # For potential evaporation/PET, negative reference values are converted to positive
        # before calibration and validation.
        long = apply_reference_value_conventions(
            long_df=long,
            parameter=parameter,
            excel_path=excel_path,
            sheet_name=sheet_name,
        )

        long = long.dropna(subset=["point_id", "date", "reference_value"])

        long_parts.append(long[[
            "parameter", "sheet_name", "point_id", "date", "year", "month", "reference_value"
        ]])

    if not long_parts:
        return pd.DataFrame(), template_sheets

    long_all = pd.concat(long_parts, ignore_index=True)
    long_all = (
        long_all
        .groupby(["parameter", "point_id", "date", "year", "month"], as_index=False)["reference_value"]
        .mean()
    )
    return long_all, template_sheets


def read_all_reference_excels():
    if not REFERENCE_FOLDER.exists():
        raise FileNotFoundError(f"Reference folder not found: {REFERENCE_FOLDER}")

    excel_files = sorted([
        p for p in list(REFERENCE_FOLDER.glob("*.xlsx")) + list(REFERENCE_FOLDER.glob("*.xls"))
        if not p.name.startswith("~$")
    ])

    if not excel_files:
        raise FileNotFoundError(f"No Excel files found in: {REFERENCE_FOLDER}")

    all_reference_parts = []
    all_templates = {}
    inventory_rows = []

    for excel_path in excel_files:
        parameter = infer_reference_parameter_from_excel_name(excel_path)
        if parameter is None:
            print_info(f"Skipping unknown reference file: {excel_path.name}")
            add_log("reference_read", "skipped", "Could not infer parameter from Excel file name.", file_path=excel_path)
            continue

        print_info(f"Reading reference file: {excel_path.name} as parameter={parameter}")
        ref_long, template_sheets = read_one_reference_excel(excel_path, parameter)

        if ref_long.empty:
            print_info(f"No usable reference data found in: {excel_path.name}")
            continue

        all_reference_parts.append(ref_long)
        all_templates[parameter] = {
            "excel_path": excel_path,
            "template_sheets": template_sheets,
        }

        inventory_rows.append({
            "parameter": parameter,
            "excel_file": excel_path.name,
            "rows_long": len(ref_long),
            "points": ref_long["point_id"].nunique(),
            "year_min": int(ref_long["year"].min()),
            "year_max": int(ref_long["year"].max()),
            "sheets": ", ".join(map(str, template_sheets.keys())),
            "selected_method": get_downscaling_method(parameter),
        })

    if not all_reference_parts:
        raise ValueError("No reference datasets were prepared from the Excel folder.")

    reference_all = pd.concat(all_reference_parts, ignore_index=True)
    reference_all = (
        reference_all
        .groupby(["parameter", "point_id", "date", "year", "month"], as_index=False)["reference_value"]
        .mean()
    )

    inventory = pd.DataFrame(inventory_rows)
    reference_path = STANDARD_DIR / "reference_all_parameters_long_1981_2025.csv"
    inventory_path = STANDARD_DIR / "reference_excel_inventory.csv"

    reference_all.to_csv(reference_path, index=False, encoding="utf-8-sig")
    inventory.to_csv(inventory_path, index=False, encoding="utf-8-sig")

    print_info(f"Saved reference long file: {reference_path}")
    print_info(f"Saved reference inventory: {inventory_path}")

    return reference_all, all_templates, inventory


# ============================================================
# 8. READ CMIP6 FILES
# ============================================================

def parse_cmip6_filename(path):
    """
    Expected style:
      precipitation__historical__miroc6__1950_2014__monthly_points.csv
      runoff_total__ssp2_4_5__mpi_esm1_2_lr__2015_2100__monthly_points.csv
    """
    stem = Path(path).stem
    parts = stem.split("__")

    parameter = parts[0] if len(parts) > 0 else "unknown"
    scenario = parts[1] if len(parts) > 1 else "unknown"
    model = parts[2] if len(parts) > 2 else "unknown"
    years = parts[3] if len(parts) > 3 else "unknown"
    start_year, end_year = extract_year_range_from_text(years)

    return {
        "cmip6_parameter": parameter,
        "scenario": scenario,
        "model": model,
        "years": years,
        "start_year": start_year,
        "end_year": end_year,
        "file_name": Path(path).name,
        "file_path": str(Path(path)),
    }


def index_cmip6_folder(folder):
    if not folder.exists():
        raise FileNotFoundError(f"CMIP6 folder not found: {folder}")

    records = [parse_cmip6_filename(path) for path in sorted(list(folder.glob("*.csv")) + list(folder.glob("*.xlsx")) + list(folder.glob("*.xls")))]
    df = pd.DataFrame(records)
    if df.empty:
        raise FileNotFoundError(f"No CSV files found in: {folder}")
    return df


def detect_value_column(df):
    preferred = [
        "converted_value", "value_converted", "cmip6_converted_value",
        "unit_converted_value", "converted", "convertedval",
        "converted_value_mm", "converted_value_c", "converted_value_m3_m3",
        "value", "raw_value", "cmip6_value", "model_value", "extracted_value",
        "precipitation", "runoff_total", "soil_moisture", "temperature_max",
        "temperature_min", "temperature_mean", "shortwave_radiation",
        "surface_downwelling_shortwave_radiation", "solar_radiation",
        "potential_evaporation", "potential_evaporation_mm",
        "pet", "groundwater", "surface_soil_moisture_m3_m3",
        "pr", "mrro", "mrsos", "mrso", "tasmax", "tasmin", "tas", "rsds",
    ]

    lower_map = {str(c).strip().lower(): c for c in df.columns}
    for name in preferred:
        if name.lower() in lower_map:
            return lower_map[name.lower()]

    bad_names = {
        "point_id", "points", "point", "id", "number",
        "lon", "longitude", "x", "lat", "latitude", "y",
        "date", "time", "year", "month",
        "nearest_grid_lon", "nearest_grid_lat",
        "model", "scenario", "parameter", "source_file",
    }

    numeric_candidates = []
    scored_candidates = []
    for col in df.columns:
        low = str(col).strip().lower()
        if low in bad_names:
            continue
        if "grid" in low or "bnds" in low or "bounds" in low:
            continue

        converted = to_numeric_clean_series(df[col])
        valid_ratio = float(converted.notna().mean()) if len(converted) else 0.0
        unique_count = int(converted.dropna().nunique())

        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_candidates.append(col)
        if valid_ratio >= 0.50 and unique_count > 1:
            scored_candidates.append((valid_ratio, unique_count, col))

    if len(numeric_candidates) == 1:
        return numeric_candidates[0]
    if numeric_candidates:
        return numeric_candidates[-1]
    if scored_candidates:
        scored_candidates.sort(key=lambda x: (x[0], x[1]))
        return scored_candidates[-1][2]

    raise ValueError("Could not detect CMIP6 value column after robust numeric conversion.")


def read_one_cmip6_csv(path):
    """
    Standardizes one CMIP6 CSV to:
      point_id, lon, lat, date, year, month, raw_value

    raw_value is the internal column name. If converted_value exists,
    converted_value is selected first and renamed internally to raw_value.
    """
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    id_col = find_column(df, ["point_id", "points", "point", "id", "number"])
    lon_col = find_column(df, ["lon", "longitude", "x"])
    lat_col = find_column(df, ["lat", "latitude", "y"])
    date_col = find_column(df, ["date", "time"])
    year_col = find_column(df, ["year"])
    month_col = find_column(df, ["month"])

    if id_col is None:
        id_col = df.columns[0]

    # Case A: long format.
    if date_col is not None or (year_col is not None and month_col is not None):
        value_col = detect_value_column(df)
        print_info(f"Selected CMIP6 value column in {Path(path).name}: {value_col}")

        out = pd.DataFrame()
        out["point_id"] = df[id_col].apply(clean_id)
        out["lon"] = to_numeric_clean_series(df[lon_col]) if lon_col is not None else np.nan
        out["lat"] = to_numeric_clean_series(df[lat_col]) if lat_col is not None else np.nan

        if date_col is not None:
            out["date"] = pd.to_datetime(df[date_col], errors="coerce")
            out["year"] = out["date"].dt.year
            out["month"] = out["date"].dt.month
        else:
            out["year"] = to_numeric_clean_series(df[year_col]).astype("Int64")
            out["month"] = to_numeric_clean_series(df[month_col]).astype("Int64")
            out["date"] = pd.to_datetime(
                out["year"].astype(str) + "-" + out["month"].astype(str).str.zfill(2) + "-01",
                errors="coerce",
            )

        out["raw_value"] = to_numeric_clean_series(df[value_col])

    # Case B: wide monthly format.
    else:
        month_cols = []
        month_info = {}
        for col in df.columns:
            parsed = parse_month_column(col)
            if parsed is not None:
                month_cols.append(col)
                month_info[col] = parsed

        if not month_cols:
            raise ValueError(f"No date/year/month or monthly columns found in {Path(path).name}")

        meta = pd.DataFrame()
        meta["point_id"] = df[id_col].apply(clean_id)
        meta["lon"] = to_numeric_clean_series(df[lon_col]) if lon_col is not None else np.nan
        meta["lat"] = to_numeric_clean_series(df[lat_col]) if lat_col is not None else np.nan

        wide = df[[id_col] + month_cols].copy()
        wide = wide.rename(columns={id_col: "point_id"})
        wide["point_id"] = wide["point_id"].apply(clean_id)

        long = wide.melt(
            id_vars=["point_id"],
            value_vars=month_cols,
            var_name="month_column",
            value_name="raw_value",
        )
        long["year"] = long["month_column"].map(lambda c: month_info[c][0])
        long["month"] = long["month_column"].map(lambda c: month_info[c][1])
        long["date"] = pd.to_datetime(
            long["year"].astype(str) + "-" + long["month"].astype(str).str.zfill(2) + "-01",
            errors="coerce",
        )
        long["raw_value"] = to_numeric_clean_series(long["raw_value"])
        out = long.merge(meta, on="point_id", how="left")

    out = out.dropna(subset=["point_id", "date", "year", "month", "raw_value"])
    out = (
        out
        .groupby(["point_id", "date", "year", "month"], as_index=False)
        .agg(
            raw_value=("raw_value", "mean"),
            lon=("lon", "first"),
            lat=("lat", "first"),
        )
    )
    return out[["point_id", "lon", "lat", "date", "year", "month", "raw_value"]]


def read_cmip6_records(records_df, reference_parameter):
    parts = []
    for _, r in records_df.iterrows():
        path = Path(r["file_path"])
        print_info(f"Reading CMIP6: {path.name}")
        df = read_one_cmip6_csv(path)
        df["reference_parameter"] = reference_parameter
        df["cmip6_parameter"] = r["cmip6_parameter"]
        df["scenario"] = r["scenario"]
        df["model"] = r["model"]
        df["source_file"] = path.name
        parts.append(df)

    if not parts:
        return pd.DataFrame()

    out = pd.concat(parts, ignore_index=True)
    out = (
        out
        .groupby([
            "reference_parameter", "model", "scenario",
            "point_id", "date", "year", "month"
        ], as_index=False)
        .agg(
            raw_value=("raw_value", "mean"),
            lon=("lon", "first"),
            lat=("lat", "first"),
            cmip6_parameter=("cmip6_parameter", "first"),
            source_file=("source_file", "first"),
        )
    )
    return out


def collapse_common_raw(common_raw):
    """
    Removes scenario/file duplication for the common training period by averaging
    rows with the same point_id, year and month. This avoids duplicate matches
    if 2015-2025 exists in both historical/overlap and future files.
    """
    if common_raw.empty:
        return common_raw
    return (
        common_raw
        .groupby(["point_id", "date", "year", "month"], as_index=False)
        .agg(
            raw_value=("raw_value", "mean"),
            lon=("lon", "first"),
            lat=("lat", "first"),
        )
    )


# ============================================================
# 9. ALIGNMENT TO REFERENCE IDS
# ============================================================

def filter_to_reference_ids(df, reference_df, parameter="", model="", scenario="", method="", stage=""):
    """
    Keeps only point IDs available in the reference Excel data.
    This keeps calibration, validation, future downscaling and change analysis
    aligned to the same reference IDs.
    """
    if df is None or df.empty:
        return df

    ref_ids = reference_df[["point_id"]].drop_duplicates().copy()
    ref_ids["point_id"] = ref_ids["point_id"].apply(clean_id)
    ref_ids = ref_ids[ref_ids["point_id"] != ""].copy()
    ref_ids["reference_order"] = range(1, len(ref_ids) + 1)

    before_rows = len(df)
    before_points = df["point_id"].nunique()

    out = df.copy()
    out["point_id"] = out["point_id"].apply(clean_id)
    out = out.merge(ref_ids, on="point_id", how="inner")

    sort_cols = ["reference_order"]
    if "year" in out.columns:
        sort_cols.append("year")
    if "month" in out.columns:
        sort_cols.append("month")
    out = out.sort_values(sort_cols).reset_index(drop=True)

    after_rows = len(out)
    after_points = out["point_id"].nunique()

    missing_in_cmip6 = sorted(set(ref_ids["point_id"]) - set(out["point_id"]))

    record_id_coverage_stage(
        parameter=parameter,
        model=model,
        scenario=scenario,
        method=method,
        stage=stage,
        reference_ids=ref_ids["point_id"],
        data_ids=out["point_id"],
        data_rows=after_rows,
    )

    print_info(
        f"Aligned {stage} to reference IDs for {parameter} {model} {scenario}: "
        f"rows {before_rows}->{after_rows}; points {before_points}->{after_points}; "
        f"reference IDs={len(ref_ids)}"
    )

    if missing_in_cmip6:
        print_info(
            f"WARNING: {len(missing_in_cmip6)} reference IDs were not found in {stage} CMIP6 data. "
            f"First missing IDs: {missing_in_cmip6[:20]}"
        )

    add_log(
        step="align_reference_ids",
        status="done",
        message=(
            f"{stage}: rows {before_rows}->{after_rows}; "
            f"points {before_points}->{after_points}; reference_points={len(ref_ids)}; "
            f"missing_reference_ids_in_cmip6={len(missing_in_cmip6)}"
        ),
        parameter=parameter,
        model=model,
        scenario=scenario,
    )

    return out


# ============================================================
# 10. CLASSICAL MONTHLY DOWNSCALING METHODS
# ============================================================

def prepare_training_groups(train_df):
    train_clean = train_df.dropna(subset=["raw_value", "reference_value", "point_id", "month"]).copy()

    exact_groups = {}
    for key, g in train_clean.groupby(["point_id", "month"]):
        if len(g) >= MIN_VALUES_PER_POINT_MONTH:
            exact_groups[key] = g

    month_groups = {}
    for month, g in train_clean.groupby("month"):
        if len(g) >= MIN_VALUES_PER_POINT_MONTH:
            month_groups[month] = g

    return exact_groups, month_groups, train_clean


def select_training_group(point_id, month, exact_groups, month_groups, all_train):
    g_train = exact_groups.get((point_id, month), None)
    if g_train is None or len(g_train) < MIN_VALUES_PER_POINT_MONTH:
        g_train = month_groups.get(month, all_train)
    return g_train


def additive_monthly_delta(apply_df, train_df, parameter):
    """
    Temperature correction.
    For each point_id and calendar month:
      corrected = raw_model + mean(reference - historical_model)
    """
    result = pd.Series(index=apply_df.index, dtype=float)
    exact_groups, month_groups, all_train = prepare_training_groups(train_df)

    for (point_id, month), g_apply in apply_df.groupby(["point_id", "month"]):
        idx = g_apply.index
        g_train = select_training_group(point_id, month, exact_groups, month_groups, all_train)

        x_train = to_numeric_clean_series(g_train["raw_value"]).values.astype(float)
        y_train = to_numeric_clean_series(g_train["reference_value"]).values.astype(float)
        x_apply = to_numeric_clean_series(g_apply["raw_value"]).values.astype(float)

        mask = np.isfinite(x_train) & np.isfinite(y_train)
        corrected = np.full_like(x_apply, np.nan, dtype=float)
        apply_mask = np.isfinite(x_apply)

        if mask.sum() >= 2:
            delta = np.nanmean(y_train[mask] - x_train[mask])
            corrected[apply_mask] = x_apply[apply_mask] + delta

        result.loc[idx] = clip_parameter_values(parameter, corrected)

    return result


def multiplicative_monthly_factor(apply_df, train_df, parameter):
    """
    Positive-variable correction.
    For each point_id and calendar month:
      factor = mean(reference) / mean(historical_model)
      corrected = raw_model * factor

    For potential evaporation/PET, the reference Excel values have already been converted
    from negative to positive before this method is called.

    If historical model mean is too close to zero, the code safely falls back to
    additive bias correction for that group.
    """
    result = pd.Series(index=apply_df.index, dtype=float)
    exact_groups, month_groups, all_train = prepare_training_groups(train_df)

    for (point_id, month), g_apply in apply_df.groupby(["point_id", "month"]):
        idx = g_apply.index
        g_train = select_training_group(point_id, month, exact_groups, month_groups, all_train)

        x_train = to_numeric_clean_series(g_train["raw_value"]).values.astype(float)
        y_train = to_numeric_clean_series(g_train["reference_value"]).values.astype(float)
        x_apply = to_numeric_clean_series(g_apply["raw_value"]).values.astype(float)

        mask = np.isfinite(x_train) & np.isfinite(y_train)
        corrected = np.full_like(x_apply, np.nan, dtype=float)
        apply_mask = np.isfinite(x_apply)

        if mask.sum() >= 2:
            mean_x = np.nanmean(x_train[mask])
            mean_y = np.nanmean(y_train[mask])

            if np.isfinite(mean_x) and abs(mean_x) > EPS:
                factor = mean_y / mean_x
                corrected[apply_mask] = x_apply[apply_mask] * factor
            else:
                bias = mean_y - mean_x
                corrected[apply_mask] = x_apply[apply_mask] + bias

        result.loc[idx] = clip_parameter_values(parameter, corrected)

    return result


def zero_inflated_two_part_monthly_correction(apply_df, train_df, parameter):
    """
    Precipitation/runoff correction.

    For each point_id and calendar month:
      Step 1: wet/dry occurrence correction
      Step 2: wet-month intensity correction

    Zeros or near-zero values are not removed. They are used to estimate wet/dry
    frequency. Only wet months are used to estimate the intensity factor.
    """
    dry_threshold = get_dry_threshold(parameter)
    result = pd.Series(index=apply_df.index, dtype=float)
    exact_groups, month_groups, all_train = prepare_training_groups(train_df)

    for (point_id, month), g_apply in apply_df.groupby(["point_id", "month"]):
        idx = g_apply.index
        g_train = select_training_group(point_id, month, exact_groups, month_groups, all_train)

        x_train = to_numeric_clean_series(g_train["raw_value"]).values.astype(float)
        y_train = to_numeric_clean_series(g_train["reference_value"]).values.astype(float)
        x_apply = to_numeric_clean_series(g_apply["raw_value"]).values.astype(float)

        train_mask = np.isfinite(x_train) & np.isfinite(y_train)
        x_train = x_train[train_mask]
        y_train = y_train[train_mask]

        corrected = np.full_like(x_apply, np.nan, dtype=float)
        apply_mask = np.isfinite(x_apply)

        if len(x_train) < 2 or len(y_train) < 2:
            result.loc[idx] = corrected
            continue

        obs_wet = y_train > dry_threshold
        obs_wet_frequency = float(np.mean(obs_wet))

        # Case 1: historically always dry in the observed/reference data.
        if obs_wet_frequency <= EPS:
            corrected[apply_mask] = 0.0

            if DRY_MONTH_POLICY == "allow_extreme_future":
                historical_model_max = np.nanmax(x_train)
                # Keep only values that exceed the maximum historical model value
                # for the same point/month or fallback group.
                future_extreme = apply_mask & (x_apply > historical_model_max)
                corrected[future_extreme] = x_apply[future_extreme]

            result.loc[idx] = clip_parameter_values(parameter, corrected)
            continue

        # Case 2: use observed wet frequency to define the model wet threshold.
        # Example: if observed wet frequency is 8%, only model values above the
        # 92nd percentile are treated as wet.
        if obs_wet_frequency >= 1.0 - EPS:
            model_wet_threshold = np.nanmin(x_train) - EPS
        else:
            model_wet_threshold = np.nanquantile(x_train, 1.0 - obs_wet_frequency)

        model_wet_train = x_train > model_wet_threshold
        model_wet_apply = apply_mask & (x_apply > model_wet_threshold)

        obs_wet_values = y_train[obs_wet]
        model_wet_values = x_train[model_wet_train]

        # Wet intensity factor.
        if len(model_wet_values) >= 1 and np.nanmean(model_wet_values) > EPS:
            intensity_factor = np.nanmean(obs_wet_values) / np.nanmean(model_wet_values)
        else:
            # Safe fallback: keep wet model intensity unchanged.
            intensity_factor = 1.0

        corrected[apply_mask] = 0.0
        corrected[model_wet_apply] = x_apply[model_wet_apply] * intensity_factor

        result.loc[idx] = clip_parameter_values(parameter, corrected)

    return result


def apply_selected_classical_method(apply_df, train_df, parameter, method):
    if method == "zero_inflated_two_part":
        return zero_inflated_two_part_monthly_correction(apply_df, train_df, parameter)
    if method == "additive_monthly_delta":
        return additive_monthly_delta(apply_df, train_df, parameter)
    if method == "multiplicative_monthly_factor":
        return multiplicative_monthly_factor(apply_df, train_df, parameter)
    raise ValueError(f"Unknown classical method: {method}")


# ============================================================
# 11. QUALITY METRICS
# ============================================================

def calc_metrics(obs, sim):
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)

    mask = np.isfinite(obs) & np.isfinite(sim)
    obs = obs[mask]
    sim = sim[mask]

    n = len(obs)
    if n < 2:
        return {
            "n": n, "r": np.nan, "r2": np.nan, "rmse": np.nan,
            "mae": np.nan, "bias": np.nan, "pbias_percent": np.nan,
            "nse": np.nan, "kge": np.nan,
        }

    error = sim - obs
    rmse = float(np.sqrt(np.mean(error ** 2)))
    mae = float(np.mean(np.abs(error)))
    bias = float(np.mean(error))

    obs_mean = np.mean(obs)
    sim_mean = np.mean(sim)

    if np.std(obs) == 0 or np.std(sim) == 0:
        r = np.nan
    else:
        r = float(np.corrcoef(obs, sim)[0, 1])

    r2 = float(r ** 2) if np.isfinite(r) else np.nan
    denom = np.sum((obs - obs_mean) ** 2)
    nse = float(1 - np.sum(error ** 2) / denom) if denom != 0 else np.nan
    pbias = float(100 * np.sum(error) / np.sum(obs)) if np.sum(obs) != 0 else np.nan

    alpha = np.std(sim) / np.std(obs) if np.std(obs) != 0 else np.nan
    beta = sim_mean / obs_mean if obs_mean != 0 else np.nan

    if np.isfinite(r) and np.isfinite(alpha) and np.isfinite(beta):
        kge = float(1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))
    else:
        kge = np.nan

    return {
        "n": n, "r": r, "r2": r2, "rmse": rmse,
        "mae": mae, "bias": bias, "pbias_percent": pbias,
        "nse": nse, "kge": kge,
    }


def metric_row(parameter, model, scenario, method, evaluation_period, obs, sim):
    row = {
        "parameter": parameter,
        "model": model,
        "scenario": scenario,
        "method": method,
        "evaluation_period": evaluation_period,
    }
    row.update(calc_metrics(obs, sim))
    return row


# ============================================================
# 12. FUTURE CHANGE ANALYSIS
# ============================================================

def calculate_future_change(parameter, model, scenario, method, future_df, reference_df):
    """Calculate future changes relative to the calibration baseline period.

    Academic choice used here:
      baseline = 1981-2005

    Reason:
      1981-2005 is the calibration/reference period used to estimate the
      monthly correction parameters, while 2006-2025 remains independent
      validation. Future changes are therefore reported relative to the same
      historical period used for calibration.
    """
    baseline_label = f"baseline_{BASELINE_START_YEAR}_{BASELINE_END_YEAR}_mean"

    baseline_source = reference_df[
        (reference_df["year"] >= BASELINE_START_YEAR)
        & (reference_df["year"] <= BASELINE_END_YEAR)
    ].copy()

    if baseline_source.empty:
        print_info(
            f"WARNING: baseline reference period {BASELINE_START_YEAR}-{BASELINE_END_YEAR} is empty. "
            "Falling back to all available reference years."
        )
        baseline_source = reference_df.copy()

    baseline = (
        baseline_source
        .groupby("point_id", as_index=False)["reference_value"]
        .mean()
        .rename(columns={"reference_value": baseline_label})
    )
    baseline["baseline_start_year"] = BASELINE_START_YEAR
    baseline["baseline_end_year"] = BASELINE_END_YEAR

    coords = (
        future_df
        .groupby("point_id", as_index=False)
        .agg(lon=("lon", "first"), lat=("lat", "first"))
    )
    baseline = baseline.merge(coords, on="point_id", how="left")

    parts = []
    for period_name, (start_year, end_year) in FUTURE_PERIODS.items():
        temp = future_df[(future_df["year"] >= start_year) & (future_df["year"] <= end_year)].copy()
        if temp.empty:
            continue

        period_mean = (
            temp
            .groupby("point_id", as_index=False)["downscaled_value"]
            .mean()
            .rename(columns={"downscaled_value": "period_mean"})
        )

        merged = period_mean.merge(baseline, on="point_id", how="left")
        merged["baseline_mean"] = merged[baseline_label]
        merged["absolute_change"] = merged["period_mean"] - merged["baseline_mean"]
        merged["percentage_change"] = np.where(
            np.abs(merged["baseline_mean"]) > EPS,
            100 * merged["absolute_change"] / merged["baseline_mean"],
            np.nan,
        )
        merged["parameter"] = parameter
        merged["model"] = model
        merged["scenario"] = scenario
        merged["method"] = method
        merged["period"] = period_name
        merged["period_start"] = start_year
        merged["period_end"] = end_year
        parts.append(merged)

    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)

def summarize_changes(change_df):
    if change_df.empty:
        return pd.DataFrame()

    return (
        change_df
        .groupby(["parameter", "method", "scenario", "model", "period"], as_index=False)
        .agg(
            points=("point_id", "nunique"),
            mean_baseline=("baseline_mean", "mean"),
            mean_period_value=("period_mean", "mean"),
            mean_absolute_change=("absolute_change", "mean"),
            median_absolute_change=("absolute_change", "median"),
            min_absolute_change=("absolute_change", "min"),
            max_absolute_change=("absolute_change", "max"),
            mean_percentage_change=("percentage_change", "mean"),
            median_percentage_change=("percentage_change", "median"),
            min_percentage_change=("percentage_change", "min"),
            max_percentage_change=("percentage_change", "max"),
        )
    )


def multi_model_summary(change_df):
    if change_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    point_summary = (
        change_df
        .groupby(["parameter", "method", "scenario", "period", "point_id"], as_index=False)
        .agg(
            lon=("lon", "first"),
            lat=("lat", "first"),
            models=("model", "nunique"),
            multi_model_period_mean=("period_mean", "mean"),
            multi_model_absolute_change=("absolute_change", "mean"),
            multi_model_percentage_change=("percentage_change", "mean"),
            model_sd_period_mean=("period_mean", "std"),
            model_sd_absolute_change=("absolute_change", "std"),
            model_sd_percentage_change=("percentage_change", "std"),
        )
    )

    period_summary = (
        point_summary
        .groupby(["parameter", "method", "scenario", "period"], as_index=False)
        .agg(
            points=("point_id", "nunique"),
            mean_multi_model_period_mean=("multi_model_period_mean", "mean"),
            mean_multi_model_absolute_change=("multi_model_absolute_change", "mean"),
            mean_multi_model_percentage_change=("multi_model_percentage_change", "mean"),
            mean_model_sd_absolute_change=("model_sd_absolute_change", "mean"),
            mean_model_sd_percentage_change=("model_sd_percentage_change", "mean"),
        )
    )

    return point_summary, period_summary


# ============================================================
# 13. EXCEL-LIKE OUTPUTS
# ============================================================

def long_to_wide_monthly(df, value_col, start_year, end_year):
    temp = df.copy()
    temp["month_column"] = temp.apply(
        lambda r: month_col_name(int(r["year"]), int(r["month"])),
        axis=1,
    )

    month_cols = make_month_columns(start_year, end_year)
    wide = temp.pivot_table(
        index="point_id",
        columns="month_column",
        values=value_col,
        aggfunc="mean",
    ).reset_index()

    for col in month_cols:
        if col not in wide.columns:
            wide[col] = np.nan

    wide = wide[["point_id"] + month_cols]
    wide["point_id"] = wide["point_id"].apply(clean_id)
    return wide


def save_excel_like_output(parameter, model, scenario, method, future_df, template_sheets):
    if not template_sheets:
        return None

    wide = long_to_wide_monthly(
        future_df,
        value_col="downscaled_value",
        start_year=FUTURE_START_YEAR,
        end_year=FUTURE_END_YEAR,
    )

    out_file = EXCEL_LIKE_DIR / safe_name(
        f"{parameter}__downscaled__{method}__{scenario}__{model}__2026_2100_excel_like.xlsx"
    )

    with pd.ExcelWriter(out_file, engine="openpyxl") as writer:
        summary_rows = []

        for sheet_name, template in template_sheets.items():
            meta_cols = list(template.columns)
            if "__point_id" in meta_cols:
                meta_cols.remove("__point_id")

            out_sheet = template.merge(
                wide,
                left_on="__point_id",
                right_on="point_id",
                how="left",
            )
            out_sheet = out_sheet[meta_cols + FUTURE_MONTH_COLUMNS]
            out_sheet.to_excel(writer, sheet_name=str(sheet_name)[:31], index=False)

            rows_with_values = out_sheet[FUTURE_MONTH_COLUMNS].notna().any(axis=1).sum()
            summary_rows.append({
                "sheet_name": sheet_name,
                "rows": len(out_sheet),
                "rows_with_values": int(rows_with_values),
                "monthly_columns": len(FUTURE_MONTH_COLUMNS),
            })

        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="summary", index=False)

    return out_file


# ============================================================
# 14. MAIN WORKFLOW
# ============================================================

def run_workflow_once():
    print_info("=" * 120)
    print_info("CLASSICAL MULTI-PARAMETER CMIP6 DOWNSCALING WORKFLOW STARTED")
    print_info("=" * 120)
    add_log("start", "started", "Classical workflow started.")

    reference_all, templates_by_parameter, reference_inventory = read_all_reference_excels()

    hist_index = index_cmip6_folder(HISTORICAL_MONTHLY_DIR)
    fut_index = index_cmip6_folder(FUTURE_MONTHLY_DIR)

    # Keep only the selected model(s), here MIROC6 / miroc6.
    selected_norm = {normalize_model_name(m) for m in SELECTED_MODELS}
    hist_index = hist_index[hist_index["model"].apply(normalize_model_name).isin(selected_norm)].copy()
    fut_index = fut_index[fut_index["model"].apply(normalize_model_name).isin(selected_norm)].copy()

    print_info(f"Selected model filter: {sorted(SELECTED_MODELS)}")
    print_info(f"Historical files after model filter: {len(hist_index)}")
    print_info(f"Future files after model filter: {len(fut_index)}")

    hist_index.to_csv(STANDARD_DIR / "cmip6_historical_file_index.csv", index=False, encoding="utf-8-sig")
    fut_index.to_csv(STANDARD_DIR / "cmip6_future_file_index.csv", index=False, encoding="utf-8-sig")

    metrics_rows = []
    all_change_parts = []

    parameters = sorted(reference_all["parameter"].unique())
    print_info(f"Reference parameters detected: {parameters}")

    for parameter in parameters:
        method = get_downscaling_method(parameter)

        print_info("=" * 120)
        print_info(f"PROCESSING PARAMETER: {parameter} | METHOD: {method}")
        print_info("=" * 120)

        ref_param = reference_all[reference_all["parameter"] == parameter].copy()

        hist_param_index = hist_index[
            hist_index["cmip6_parameter"].apply(lambda p: parameter_matches_cmip6(parameter, p))
        ].copy()

        fut_param_index = fut_index[
            fut_index["cmip6_parameter"].apply(lambda p: parameter_matches_cmip6(parameter, p))
        ].copy()

        if hist_param_index.empty:
            msg = f"No matching historical CMIP6 files found for {parameter}."
            print_info(msg)
            add_log("parameter", "skipped", msg, parameter=parameter, method=method)
            continue

        if fut_param_index.empty:
            msg = f"No matching future CMIP6 files found for {parameter}."
            print_info(msg)
            add_log("parameter", "skipped", msg, parameter=parameter, method=method)
            continue

        combinations = (
            fut_param_index[["model", "scenario"]]
            .drop_duplicates()
            .sort_values(["scenario", "model"])
            .reset_index(drop=True)
        )

        print_info(f"Future model-scenario combinations for {parameter}: {len(combinations)}")
        print(combinations.to_string(index=False))

        for _, combo in combinations.iterrows():
            model = combo["model"]
            scenario = combo["scenario"]

            print_info("-" * 120)
            print_info(f"Parameter={parameter} | Model={model} | Scenario={scenario} | Method={method}")
            print_info("-" * 120)

            try:
                hist_files = hist_param_index[
                    (hist_param_index["model"].apply(normalize_model_name) == normalize_model_name(model))
                    & (hist_param_index["scenario"].astype(str).str.lower() == "historical")
                ]

                # Optional overlap files in the historical folder, if they exist.
                overlap_files = hist_param_index[
                    (hist_param_index["model"].apply(normalize_model_name) == normalize_model_name(model))
                    & (hist_param_index["scenario"] == scenario)
                ]

                future_files = fut_param_index[
                    (fut_param_index["model"].apply(normalize_model_name) == normalize_model_name(model))
                    & (fut_param_index["scenario"] == scenario)
                ]

                if hist_files.empty:
                    msg = "No historical CMIP6 file for this model."
                    print_info(msg)
                    add_log("combination", "skipped", msg, parameter=parameter, model=model, scenario=scenario, method=method)
                    continue

                if future_files.empty:
                    msg = "No future CMIP6 file for this model-scenario."
                    print_info(msg)
                    add_log("combination", "skipped", msg, parameter=parameter, model=model, scenario=scenario, method=method)
                    continue

                # Use historical files plus any scenario overlap files.
                # Also include future files because many CMIP6 future files start in 2015,
                # which is needed for validation through 2025.
                common_file_records = pd.concat([hist_files, overlap_files, future_files], ignore_index=True)
                common_raw = read_cmip6_records(common_file_records, parameter)
                common_raw = common_raw[
                    (common_raw["year"] >= COMMON_START_YEAR)
                    & (common_raw["year"] <= COMMON_END_YEAR)
                ].copy()
                common_raw = collapse_common_raw(common_raw)
                common_raw = filter_to_reference_ids(
                    common_raw, ref_param, parameter=parameter, model=model, scenario=scenario, method=method, stage="common"
                )

                future_raw = read_cmip6_records(future_files, parameter)
                future_raw = future_raw[
                    (future_raw["year"] >= FUTURE_START_YEAR)
                    & (future_raw["year"] <= FUTURE_END_YEAR)
                ].copy()
                future_raw = filter_to_reference_ids(
                    future_raw, ref_param, parameter=parameter, model=model, scenario=scenario, method=method, stage="future"
                )

                if common_raw.empty or future_raw.empty:
                    msg = "Common or future raw data is empty after filtering."
                    print_info(msg)
                    add_log("combination", "skipped", msg, parameter=parameter, model=model, scenario=scenario, method=method)
                    continue

                common = common_raw.merge(
                    ref_param[["point_id", "year", "month", "reference_value"]],
                    on=["point_id", "year", "month"],
                    how="inner",
                )
                common = common.dropna(subset=["raw_value", "reference_value"])

                record_id_coverage_stage(
                    parameter=parameter,
                    model=model,
                    scenario=scenario,
                    method=method,
                    stage="common_after_reference_month_merge",
                    reference_ids=ref_param["point_id"],
                    data_ids=common["point_id"] if not common.empty else [],
                    data_rows=len(common),
                )

                if common.empty:
                    msg = "No point-year-month overlap between reference and CMIP6 common period."
                    print_info(msg)
                    add_log("combination", "skipped", msg, parameter=parameter, model=model, scenario=scenario, method=method)
                    continue

                calibration = common[
                    (common["year"] >= CALIBRATION_START_YEAR)
                    & (common["year"] <= CALIBRATION_END_YEAR)
                ].copy()

                validation = common[
                    (common["year"] >= VALIDATION_START_YEAR)
                    & (common["year"] <= VALIDATION_END_YEAR)
                ].copy()

                if len(calibration) < MIN_CALIBRATION_ROWS or len(validation) < MIN_VALIDATION_ROWS:
                    msg = (
                        f"Not enough calibration/validation rows. "
                        f"Calibration rows={len(calibration)}; validation rows={len(validation)}."
                    )
                    print_info(msg)
                    add_log("combination", "skipped", msg, parameter=parameter, model=model, scenario=scenario, method=method)
                    continue

                evaluation_period = f"validation_{VALIDATION_START_YEAR}_{VALIDATION_END_YEAR}"

                print_info(
                    f"Common rows={len(common)} | calibration={CALIBRATION_START_YEAR}-{CALIBRATION_END_YEAR} "
                    f"rows={len(calibration)} | validation={VALIDATION_START_YEAR}-{VALIDATION_END_YEAR} "
                    f"rows={len(validation)} | points={common['point_id'].nunique()}"
                )

                add_log(
                    "calibration_validation_split",
                    "done",
                    f"Calibration={CALIBRATION_START_YEAR}-{CALIBRATION_END_YEAR}; "
                    f"Validation={VALIDATION_START_YEAR}-{VALIDATION_END_YEAR}; "
                    f"calibration_rows={len(calibration)}; validation_rows={len(validation)}",
                    parameter=parameter,
                    model=model,
                    scenario=scenario,
                    method=method,
                )

                # Raw quality before downscaling.
                metrics_rows.append(metric_row(
                    parameter, model, scenario,
                    "converted_cmip6_before_downscaling",
                    evaluation_period,
                    validation["reference_value"].values,
                    validation["raw_value"].values,
                ))

                raw_quality_file = QUALITY_DIR / safe_name(
                    f"{parameter}__common_converted_cmip6__{scenario}__{model}.csv"
                )
                common.to_csv(raw_quality_file, index=False, encoding="utf-8-sig")

                # Validation with selected method only.
                val_downscaled = validation.copy()
                val_downscaled["downscaled_value"] = apply_selected_classical_method(
                    apply_df=validation,
                    train_df=calibration,
                    parameter=parameter,
                    method=method,
                )

                val_downscaled["downscaled_value"] = clip_parameter_values(
                    parameter,
                    val_downscaled["downscaled_value"].values,
                )

                metrics_rows.append(metric_row(
                    parameter, model, scenario,
                    method,
                    evaluation_period,
                    val_downscaled["reference_value"].values,
                    val_downscaled["downscaled_value"].values,
                ))

                val_file = QUALITY_DIR / safe_name(
                    f"{parameter}__validation_downscaled__{method}__{scenario}__{model}.csv"
                )
                val_downscaled.to_csv(val_file, index=False, encoding="utf-8-sig")

                # Future correction uses all common years 1981-2025 after validation.
                future_downscaled = future_raw.copy()
                future_downscaled["downscaled_value"] = apply_selected_classical_method(
                    apply_df=future_raw,
                    train_df=common,
                    parameter=parameter,
                    method=method,
                )
                future_downscaled["downscaled_value"] = clip_parameter_values(
                    parameter,
                    future_downscaled["downscaled_value"].values,
                )
                future_downscaled["parameter"] = parameter
                future_downscaled["method"] = method
                future_downscaled["model"] = model
                future_downscaled["scenario"] = scenario

                future_out = FUTURE_DS_DIR / safe_name(
                    f"{parameter}__downscaled__{method}__{scenario}__{model}__2026_2100_monthly.csv"
                )
                future_downscaled.to_csv(future_out, index=False, encoding="utf-8-sig")
                print_info(f"Saved future downscaled CSV: {future_out}")

                change_df = calculate_future_change(
                    parameter=parameter,
                    model=model,
                    scenario=scenario,
                    method=method,
                    future_df=future_downscaled,
                    reference_df=ref_param,
                )

                if not change_df.empty:
                    change_out = CHANGE_DIR / safe_name(
                        f"{parameter}__future_changes_by_point__{method}__{scenario}__{model}.csv"
                    )
                    change_df.to_csv(change_out, index=False, encoding="utf-8-sig")
                    all_change_parts.append(change_df)

                if SAVE_EXCEL_LIKE_OUTPUTS and parameter in templates_by_parameter:
                    excel_file = save_excel_like_output(
                        parameter=parameter,
                        model=model,
                        scenario=scenario,
                        method=method,
                        future_df=future_downscaled,
                        template_sheets=templates_by_parameter[parameter]["template_sheets"],
                    )
                    if excel_file is not None:
                        print_info(f"Saved Excel-like future file: {excel_file}")

                add_log(
                    "combination",
                    "success",
                    "Finished this parameter/model/scenario with selected classical method.",
                    parameter=parameter,
                    model=model,
                    scenario=scenario,
                    method=method,
                )

            except Exception as e:
                msg = f"{type(e).__name__}: {e}"
                print_info(f"FAILED: parameter={parameter}, model={model}, scenario={scenario} | {msg}")
                add_log("combination", "failed", msg, parameter=parameter, model=model, scenario=scenario, method=method)

    # ========================================================
    # SAVE FINAL QUALITY METRICS
    # ========================================================

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_file = QUALITY_DIR / "downscaling_quality_metrics_classical_only_PET_SHORTWAVE_IDCHECK.csv"
    metrics_df.to_csv(metrics_file, index=False, encoding="utf-8-sig")

    # Method mapping file.
    method_map = []
    for parameter in parameters:
        method_map.append({
            "parameter": parameter,
            "selected_method": get_downscaling_method(parameter),
            "reason": (
                "zero-inflated wet/dry plus wet-intensity correction"
                if get_downscaling_method(parameter) == "zero_inflated_two_part"
                else "additive monthly delta correction"
                if get_downscaling_method(parameter) == "additive_monthly_delta"
                else "multiplicative monthly factor correction"
            ),
        })
    method_map_df = pd.DataFrame(method_map)
    method_map_file = QUALITY_DIR / "selected_classical_method_by_parameter.csv"
    method_map_df.to_csv(method_map_file, index=False, encoding="utf-8-sig")

    # ========================================================
    # SAVE FINAL FUTURE CHANGE SUMMARIES
    # ========================================================

    final_excel = OUT_ROOT / "classical_downscaling_quality_and_future_change_report.xlsx"

    with pd.ExcelWriter(final_excel, engine="openpyxl") as writer:
        reference_inventory.to_excel(writer, sheet_name="reference_inventory", index=False)
        method_map_df.to_excel(writer, sheet_name="selected_methods", index=False)
        metrics_df.to_excel(writer, sheet_name="quality_metrics", index=False)

        if all_change_parts:
            all_changes = pd.concat(all_change_parts, ignore_index=True)
            all_changes_file = CHANGE_DIR / "future_changes_by_point_all_parameters_classical_only.csv"
            all_changes.to_csv(all_changes_file, index=False, encoding="utf-8-sig")

            change_summary = summarize_changes(all_changes)
            change_summary_file = CHANGE_DIR / "future_changes_summary_by_parameter_method_model_scenario_period.csv"
            change_summary.to_csv(change_summary_file, index=False, encoding="utf-8-sig")

            mm_point, mm_period = multi_model_summary(all_changes)
            mm_point_file = CHANGE_DIR / "model_future_changes_by_point.csv"
            mm_period_file = CHANGE_DIR / "model_future_changes_period_summary.csv"
            mm_point.to_csv(mm_point_file, index=False, encoding="utf-8-sig")
            mm_period.to_csv(mm_period_file, index=False, encoding="utf-8-sig")

            change_summary.to_excel(writer, sheet_name="future_change_summary", index=False)
            mm_period.to_excel(writer, sheet_name="model_summary", index=False)
            all_changes.head(900000).to_excel(writer, sheet_name="changes_by_point_sample", index=False)

    add_log("finish", "success", "Classical workflow finished.")

    print_info("=" * 120)
    print_info("CLASSICAL WORKFLOW FINISHED")
    print_info("=" * 120)
    print_info(f"Output root: {OUT_ROOT}")
    print_info(f"Quality metrics: {metrics_file}")
    print_info(f"Selected method map: {method_map_file}")
    print_info(f"Future change analysis folder: {CHANGE_DIR}")
    print_info(f"Excel-like future outputs folder: {EXCEL_LIKE_DIR}")
    print_info(f"Final Excel report: {final_excel}")
    print_info(f"Log file: {LOG_FILE}")

    print("")
    print("IMPORTANT OUTPUTS")
    print("-" * 90)
    print(f"1) Quality metrics: {metrics_file}")
    print(f"2) Selected methods: {method_map_file}")
    print(f"3) Future change summary: {CHANGE_DIR / 'future_changes_summary_by_parameter_method_model_scenario_period.csv'}")
    print(f"4) Model future-change summary: {CHANGE_DIR / 'model_future_changes_period_summary.csv'}")
    print(f"5) Final Excel report: {final_excel}")
    print(f"6) ID coverage report: {ID_COVERAGE_FILE}")
    print(f"7) Missing ID details: {MISSING_ID_FILE}")




# ============================================================
# FINAL ROBUST OVERRIDES FOR PET + SHORTWAVE + NUMERIC VALUES
# ============================================================
# These functions intentionally override earlier functions above.
# They solve the specific issue:
#   Reference file name:  Potential evaporation (mm)
#   CMIP6 file name:      potential_evaporation__...
# and also make shortwave and numeric parsing more robust.

DEBUG_REFERENCE_DETECTION_FILE = LOG_DIR / "reference_parameter_detection_debug.csv"
DEBUG_CMIP6_DETECTION_FILE = LOG_DIR / "cmip6_parameter_detection_debug.csv"
DEBUG_MATCHING_FILE = LOG_DIR / "pet_shortwave_matching_debug.csv"


def infer_parameter_from_any_text(text):
    """Return the canonical parameter name from a filename, sheet name, or column text."""
    n = normalize_text(text)

    # Explicit PET mapping for reference and climate-model filenames.
    # Example: Potential evaporation (mm) -> potential_evaporation_mm -> potential_evaporation
    # Example: potential_evaporation -> potential_evaporation
    if n in FORCE_REFERENCE_PET_NAMES or n in FORCE_CMIP6_PET_NAMES:
        return "potential_evaporation"

    # PET must be checked BEFORE generic evaporation words.
    if (
        n == "pet"
        or n in {"eto", "et0"}
        or "potential_evaporation" in n
        or "potential_evapotranspiration" in n
        or ("potential" in n and "evap" in n)
        or "reference_evaporation" in n
        or "reference_evapotranspiration" in n
    ):
        return "potential_evaporation"

    # The reference variable name normalizes to potential_evaporation_mm.
    if n == "potential_evaporation_mm":
        return "potential_evaporation"

    if (
        "shortwave" in n
        or "short_wave" in n
        or "rsds" in n
        or ("solar" in n and "radiation" in n)
        or ("radiation" in n and "longwave" not in n and "long_wave" not in n)
    ):
        return "shortwave_radiation"

    if "soil" in n and "moist" in n:
        return "soil_moisture"
    if ("temperature" in n or "temp" in n) and ("max" in n or "tasmax" in n or "tmax" in n):
        return "temperature_max"
    if ("temperature" in n or "temp" in n) and ("min" in n or "tasmin" in n or "tmin" in n):
        return "temperature_min"
    if ("temperature" in n or "temp" in n) and ("mean" in n or n == "tas" or "tmean" in n):
        return "temperature_mean"
    if "precip" in n or "rainfall" in n or n == "rain" or n == "pr":
        return "precipitation"
    if "runoff" in n or n == "mrro" or "total_runoff" in n:
        return "runoff_total"
    if "ground" in n and "water" in n:
        return "groundwater"

    # IMPORTANT: generic evaporation in reference is treated as PET only for reference data.
    # Actual CMIP6 evaporation is NOT treated as PET unless its filename contains potential/PET.
    if "evaporation" in n or "evapor" in n:
        return "potential_evaporation"

    return None


def infer_reference_parameter_from_excel_name(excel_path):
    """Override: robust reference filename matching, including Potential evaporation (mm)."""
    return infer_parameter_from_any_text(Path(excel_path).stem)


def parameter_matches_cmip6(reference_parameter, cmip6_parameter):
    """Override: robust matching between canonical reference parameter and CMIP6 filename parameter."""
    ref = normalize_text(reference_parameter)
    cmip_raw = str(cmip6_parameter)
    cmip = normalize_text(cmip_raw)

    # Never allow actual CMIP6 evaporation/evspsbl to match PET.
    if ref == "potential_evaporation":
        if cmip in {"evaporation", "evspsbl", "actual_evaporation"}:
            return False
        if cmip in FORCE_CMIP6_PET_NAMES:
            return True
        return (
            "potential_evaporation" in cmip
            or "potential_evapotranspiration" in cmip
            or cmip in {"pet", "eto", "et0"}
            or ("potential" in cmip and "evap" in cmip)
        )

    if ref == "shortwave_radiation":
        return (
            "shortwave" in cmip
            or "short_wave" in cmip
            or cmip == "rsds"
            or "rsds" in cmip
            or ("solar" in cmip and "radiation" in cmip)
            or "surface_downwelling_shortwave" in cmip
        )

    canonical = infer_parameter_from_any_text(cmip_raw)
    if canonical is not None:
        return canonical == ref

    if reference_parameter not in PARAMETER_RULES:
        return cmip == ref

    aliases = [normalize_text(a) for a in PARAMETER_RULES[reference_parameter]["cmip6_aliases"]]
    return cmip in aliases


def parse_cmip6_filename(path):
    """Override: parse CMIP6 filename robustly even if parameter contains underscores."""
    p = Path(path)
    stem = p.stem
    parts = stem.split("__")

    scenario_idx = None
    scenario = "unknown"
    for i, part in enumerate(parts):
        pn = normalize_text(part)
        if pn == "historical" or pn.startswith("ssp"):
            scenario_idx = i
            scenario = pn
            break

    if scenario_idx is None:
        parameter_text = parts[0] if parts else stem
        model = "unknown"
        years = "unknown"
    else:
        parameter_text = "__".join(parts[:scenario_idx]) if scenario_idx > 0 else parts[0]
        model = parts[scenario_idx + 1] if len(parts) > scenario_idx + 1 else "unknown"
        years = "unknown"
        for part in parts[scenario_idx + 2:]:
            sy, ey = extract_year_range_from_text(part)
            if sy is not None:
                years = part
                break

    canonical = infer_parameter_from_any_text(parameter_text)
    if canonical is None:
        canonical = normalize_text(parameter_text)

    start_year, end_year = extract_year_range_from_text(years)

    return {
        "cmip6_parameter": canonical,
        "cmip6_parameter_original": parameter_text,
        "scenario": scenario,
        "model": normalize_text(model),
        "model_original": model,
        "years": years,
        "start_year": start_year,
        "end_year": end_year,
        "file_name": p.name,
        "file_path": str(p),
    }


def index_cmip6_folder(folder):
    """Index CSV/XLSX inputs in monthly and parent directories; record diagnostics."""
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"CMIP6 folder not found: {folder}")

    roots = [folder]
    # Also search the root folder because PET/shortwave products may have been saved in another subfolder.
    if folder.name.lower() == "02_extracted_monthly_csv" and folder.parent.exists():
        roots.append(folder.parent)

    candidates = []
    seen = set()
    for root in roots:
        for ext in ("*.csv", "*.xlsx", "*.xls", "*.xlsm"):
            for path in root.rglob(ext):
                if path.name.startswith("~$"):
                    continue
                key = str(path.resolve())
                if key in seen:
                    continue
                seen.add(key)
                name_norm = normalize_text(path.name)
                # Keep monthly point files only; avoid annual/period/report files.
                if "monthly" not in name_norm:
                    continue
                if any(x in name_norm for x in ["annual", "period", "summary", "quality", "change", "report", "downscaled"]):
                    continue
                candidates.append(path)

    records = [parse_cmip6_filename(path) for path in sorted(candidates)]
    df = pd.DataFrame(records)
    if df.empty:
        raise FileNotFoundError(f"No monthly CSV/Excel CMIP6 files found in: {folder} or parent root")

    # Save a diagnostic file showing PET and shortwave data availability.
    try:
        tag = "historical" if "historical" in normalize_text(str(folder)) else "future"
        debug_path = LOG_DIR / f"cmip6_file_detection_debug_{tag}.csv"
        df.to_csv(debug_path, index=False, encoding="utf-8-sig")
    except Exception:
        pass

    return df


def detect_value_column(df):
    """Override: include potential_evaporation and shortwave aliases in value-column detection."""
    preferred = [
        "converted_value", "value_converted", "cmip6_converted_value",
        "unit_converted_value", "converted", "convertedval",
        "converted_value_mm", "converted_value_c", "converted_value_m3_m3",
        "potential_evaporation", "potential_evapotranspiration", "pet", "eto", "et0",
        "shortwave_radiation", "surface_downwelling_shortwave_radiation", "rsds", "solar_radiation",
        "value", "raw_value", "cmip6_value", "model_value", "extracted_value",
        "precipitation", "runoff_total", "soil_moisture", "temperature_max",
        "temperature_min", "temperature_mean", "groundwater", "surface_soil_moisture_m3_m3",
        "pr", "mrro", "mrsos", "mrso", "tasmax", "tasmin", "tas",
    ]

    lower_map = {str(c).strip().lower(): c for c in df.columns}
    for name in preferred:
        if name.lower() in lower_map:
            return lower_map[name.lower()]

    bad_names = {
        "point_id", "points", "point", "id", "number",
        "lon", "longitude", "x", "lat", "latitude", "y",
        "date", "time", "year", "month",
        "nearest_grid_lon", "nearest_grid_lat",
        "model", "scenario", "parameter", "source_file",
    }

    numeric_candidates = []
    for col in df.columns:
        low = str(col).strip().lower()
        if low in bad_names:
            continue
        if "grid" in low or "bnds" in low or "bounds" in low:
            continue
        temp_numeric = to_numeric_clean_series(df[col])
        if temp_numeric.notna().sum() > max(5, 0.1 * len(df)):
            numeric_candidates.append(col)

    if numeric_candidates:
        return numeric_candidates[-1]

    raise ValueError("Could not detect CMIP6 value column.")


def _standardize_one_cmip6_dataframe(df, path_label=""):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    id_col = find_column(df, ["point_id", "points", "point", "id", "number"])
    lon_col = find_column(df, ["lon", "longitude", "x", "point_lon"])
    lat_col = find_column(df, ["lat", "latitude", "y", "point_lat"])
    date_col = find_column(df, ["date", "time"])
    year_col = find_column(df, ["year"])
    month_col = find_column(df, ["month"])

    if id_col is None:
        id_col = df.columns[0]

    if date_col is not None or (year_col is not None and month_col is not None):
        value_col = detect_value_column(df)
        print_info(f"Selected CMIP6 value column in {Path(path_label).name}: {value_col}")

        out = pd.DataFrame()
        out["point_id"] = df[id_col].apply(clean_id)
        out["lon"] = to_numeric_clean_series(df[lon_col]) if lon_col is not None else np.nan
        out["lat"] = to_numeric_clean_series(df[lat_col]) if lat_col is not None else np.nan

        if date_col is not None:
            out["date"] = pd.to_datetime(df[date_col], errors="coerce")
            out["year"] = out["date"].dt.year
            out["month"] = out["date"].dt.month
        else:
            out["year"] = to_numeric_clean_series(df[year_col]).astype("Int64")
            out["month"] = to_numeric_clean_series(df[month_col]).astype("Int64")
            out["date"] = pd.to_datetime(
                out["year"].astype(str) + "-" + out["month"].astype(str).str.zfill(2) + "-01",
                errors="coerce",
            )

        out["raw_value"] = to_numeric_clean_series(df[value_col])

    else:
        month_cols = []
        month_info = {}
        for col in df.columns:
            parsed = parse_month_column(col)
            if parsed is not None:
                month_cols.append(col)
                month_info[col] = parsed

        if not month_cols:
            raise ValueError(f"No date/year/month or monthly columns found in {Path(path_label).name}")

        meta = pd.DataFrame()
        meta["point_id"] = df[id_col].apply(clean_id)
        meta["lon"] = to_numeric_clean_series(df[lon_col]) if lon_col is not None else np.nan
        meta["lat"] = to_numeric_clean_series(df[lat_col]) if lat_col is not None else np.nan

        wide = df[[id_col] + month_cols].copy()
        wide = wide.rename(columns={id_col: "point_id"})
        wide["point_id"] = wide["point_id"].apply(clean_id)

        long = wide.melt(
            id_vars=["point_id"],
            value_vars=month_cols,
            var_name="month_column",
            value_name="raw_value",
        )
        long["year"] = long["month_column"].map(lambda c: month_info[c][0])
        long["month"] = long["month_column"].map(lambda c: month_info[c][1])
        long["date"] = pd.to_datetime(
            long["year"].astype(str) + "-" + long["month"].astype(str).str.zfill(2) + "-01",
            errors="coerce",
        )
        long["raw_value"] = to_numeric_clean_series(long["raw_value"])
        out = long.merge(meta, on="point_id", how="left")

    out = out.dropna(subset=["point_id", "date", "year", "month", "raw_value"])
    out = (
        out
        .groupby(["point_id", "date", "year", "month"], as_index=False)
        .agg(
            raw_value=("raw_value", "mean"),
            lon=("lon", "first"),
            lat=("lat", "first"),
        )
    )
    return out[["point_id", "lon", "lat", "date", "year", "month", "raw_value"]]


def read_one_cmip6_csv(path):
    """Override: read CMIP6 CSV or Excel, long or wide, and robustly convert numeric values."""
    path = Path(path)
    parts = []
    if path.suffix.lower() == ".csv":
        sheets = {"csv": pd.read_csv(path)}
    elif path.suffix.lower() in {".xlsx", ".xls", ".xlsm"}:
        sheets = pd.read_excel(path, sheet_name=None)
    else:
        raise ValueError(f"Unsupported CMIP6 file type: {path}")

    for sheet_name, df in sheets.items():
        if df is None or df.empty:
            continue
        if str(sheet_name).lower() in {"summary", "readme", "metadata"}:
            continue
        try:
            parts.append(_standardize_one_cmip6_dataframe(df, path_label=f"{path.name}:{sheet_name}"))
        except Exception as e:
            print_info(f"WARNING: skipped sheet {sheet_name} in {path.name}: {type(e).__name__}: {e}")

    if not parts:
        raise ValueError(f"Could not read CMIP6 values from: {path}")

    out = pd.concat(parts, ignore_index=True)
    out = (
        out
        .groupby(["point_id", "date", "year", "month"], as_index=False)
        .agg(
            raw_value=("raw_value", "mean"),
            lon=("lon", "first"),
            lat=("lat", "first"),
        )
    )
    return out[["point_id", "lon", "lat", "date", "year", "month", "raw_value"]]


def _read_reference_dataframe_to_long(df, parameter, excel_path, sheet_name):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    month_cols = []
    month_info = {}
    for col in df.columns:
        parsed = parse_month_column(col)
        if parsed is None:
            continue
        year, month = parsed
        if REFERENCE_START_YEAR <= year <= REFERENCE_END_YEAR:
            month_cols.append(col)
            month_info[col] = (year, month)

    if not month_cols:
        return pd.DataFrame(), None

    first_month_index = min(df.columns.get_loc(c) for c in month_cols)
    meta_cols = list(df.columns[:first_month_index])
    if not meta_cols:
        meta_cols = [df.columns[0]]
    id_col = meta_cols[0]

    template = df[meta_cols].copy()
    template["__point_id"] = template[id_col].apply(clean_id)

    temp = df[[id_col] + month_cols].copy()
    temp = temp.rename(columns={id_col: "point_id"})
    temp["point_id"] = temp["point_id"].apply(clean_id)

    long = temp.melt(
        id_vars=["point_id"],
        value_vars=month_cols,
        var_name="month_column",
        value_name="reference_value",
    )
    long["year"] = long["month_column"].map(lambda c: month_info[c][0])
    long["month"] = long["month_column"].map(lambda c: month_info[c][1])
    long["date"] = pd.to_datetime(
        long["year"].astype(str) + "-" + long["month"].astype(str).str.zfill(2) + "-01"
    )
    long["reference_value"] = to_numeric_clean_series(long["reference_value"])
    long["parameter"] = parameter
    long["sheet_name"] = sheet_name
    long["source_reference_file"] = Path(excel_path).name
    long = long.dropna(subset=["point_id", "date", "reference_value"])

    if parameter == "potential_evaporation":
        # Reference PET must be positive. This also handles old datasets where evaporation sign is negative.
        long["reference_value"] = long["reference_value"].abs()

    return long[[
        "parameter", "sheet_name", "point_id", "date", "year", "month", "reference_value", "source_reference_file"
    ]], template


def read_all_reference_excels():
    """Override: read xlsx/xls/xlsm/csv reference files and detect parameter from file OR sheet name."""
    if not REFERENCE_FOLDER.exists():
        raise FileNotFoundError(f"Reference folder not found: {REFERENCE_FOLDER}")

    candidates = []
    for ext in ("*.xlsx", "*.xls", "*.xlsm", "*.csv"):
        candidates.extend([p for p in REFERENCE_FOLDER.glob(ext) if not p.name.startswith("~$")])
    candidates = sorted(candidates)

    if not candidates:
        raise FileNotFoundError(f"No Excel/CSV reference files found in: {REFERENCE_FOLDER}")

    all_reference_parts = []
    all_templates = {}
    inventory_rows = []
    debug_rows = []

    for excel_path in candidates:
        file_parameter = infer_parameter_from_any_text(excel_path.stem)

        if excel_path.suffix.lower() == ".csv":
            sheets = {"csv": pd.read_csv(excel_path)}
        else:
            sheets = pd.read_excel(excel_path, sheet_name=None)

        for sheet_name, df in sheets.items():
            if df is None or df.empty:
                continue
            if str(sheet_name).lower() in {"summary", "readme", "metadata"}:
                continue

            sheet_parameter = infer_parameter_from_any_text(sheet_name)
            parameter = sheet_parameter or file_parameter

            debug_rows.append({
                "reference_file": excel_path.name,
                "sheet_name": sheet_name,
                "file_parameter_detected": file_parameter,
                "sheet_parameter_detected": sheet_parameter,
                "final_parameter_used": parameter,
            })

            if parameter is None:
                print_info(f"Skipping unknown reference: {excel_path.name} | sheet {sheet_name}")
                continue

            print_info(f"Reading reference: {excel_path.name} | sheet={sheet_name} as parameter={parameter}")
            ref_long, template = _read_reference_dataframe_to_long(df, parameter, excel_path, sheet_name)

            if ref_long.empty:
                print_info(f"No usable monthly reference data: {excel_path.name} | sheet {sheet_name}")
                continue

            all_reference_parts.append(ref_long)

            all_templates.setdefault(parameter, {"excel_path": excel_path, "template_sheets": {}})
            safe_sheet_key = f"{excel_path.stem}_{sheet_name}"[:31]
            all_templates[parameter]["template_sheets"][safe_sheet_key] = template

            inventory_rows.append({
                "parameter": parameter,
                "excel_file": excel_path.name,
                "sheet_name": sheet_name,
                "rows_long": len(ref_long),
                "points": ref_long["point_id"].nunique(),
                "year_min": int(ref_long["year"].min()),
                "year_max": int(ref_long["year"].max()),
                "selected_method": get_downscaling_method(parameter),
            })

    pd.DataFrame(debug_rows).to_csv(DEBUG_REFERENCE_DETECTION_FILE, index=False, encoding="utf-8-sig")

    if not all_reference_parts:
        raise ValueError("No reference datasets were prepared from the reference folder.")

    reference_all = pd.concat(all_reference_parts, ignore_index=True)
    reference_all = (
        reference_all
        .groupby(["parameter", "point_id", "date", "year", "month"], as_index=False)["reference_value"]
        .mean()
    )

    inventory = pd.DataFrame(inventory_rows)
    reference_path = STANDARD_DIR / "reference_all_parameters_long_1981_2025.csv"
    inventory_path = STANDARD_DIR / "reference_excel_inventory.csv"

    reference_all.to_csv(reference_path, index=False, encoding="utf-8-sig")
    inventory.to_csv(inventory_path, index=False, encoding="utf-8-sig")

    print_info(f"Saved reference long file: {reference_path}")
    print_info(f"Saved reference inventory: {inventory_path}")
    print_info(f"Saved reference detection debug: {DEBUG_REFERENCE_DETECTION_FILE}")

    return reference_all, all_templates, inventory


def read_cmip6_records(records_df, reference_parameter):
    """Override: read CMIP6 records and preserve original file info for diagnostics."""
    parts = []
    for _, r in records_df.iterrows():
        path = Path(r["file_path"])
        print_info(f"Reading CMIP6: {path.name}")
        df = read_one_cmip6_csv(path)
        df["reference_parameter"] = reference_parameter
        df["cmip6_parameter"] = r["cmip6_parameter"]
        df["scenario"] = r["scenario"]
        df["model"] = r["model"]
        df["source_file"] = path.name
        parts.append(df)

    if not parts:
        return pd.DataFrame()

    out = pd.concat(parts, ignore_index=True)
    out = (
        out
        .groupby([
            "reference_parameter", "model", "scenario",
            "point_id", "date", "year", "month"
        ], as_index=False)
        .agg(
            raw_value=("raw_value", "mean"),
            lon=("lon", "first"),
            lat=("lat", "first"),
            cmip6_parameter=("cmip6_parameter", "first"),
            source_file=("source_file", "first"),
        )
    )
    return out



# ============================================================
# 15. PET DERIVED DIRECTLY FROM TEMPERATURE FILES
# ============================================================
# This block forces potential_evaporation to be calculated from CMIP6 temperature_mean,
# temperature_max and temperature_min. It does NOT need CMIP6 potential_evaporation files.
# It also does NOT use actual evaporation / evspsbl.

PET_DERIVE_FROM_TEMPERATURE = True
PET_SOURCE_DESCRIPTION = "derived_inside_downscaling_from_temperature_mean_max_min_hargreaves_samani"
PET_TEMPERATURE_PARAMETERS = ["temperature_mean", "temperature_max", "temperature_min"]


def is_temperature_parameter_for_pet(cmip6_parameter):
    p = normalize_text(cmip6_parameter)
    if p in {"temperature_mean", "temperature_max", "temperature_min", "tas", "tasmax", "tasmin", "tmean", "tmax", "tmin"}:
        return True
    if parameter_matches_cmip6("temperature_mean", p):
        return True
    if parameter_matches_cmip6("temperature_max", p):
        return True
    if parameter_matches_cmip6("temperature_min", p):
        return True
    return False


def canonical_temperature_parameter(cmip6_parameter):
    p = normalize_text(cmip6_parameter)
    if p in {"temperature_mean", "tas", "tmean", "mean_temperature", "temperature"}:
        return "temperature_mean"
    if p in {"temperature_max", "tasmax", "tmax", "max_temperature", "maximum_temperature"}:
        return "temperature_max"
    if p in {"temperature_min", "tasmin", "tmin", "min_temperature", "minimum_temperature"}:
        return "temperature_min"
    if parameter_matches_cmip6("temperature_mean", p):
        return "temperature_mean"
    if parameter_matches_cmip6("temperature_max", p):
        return "temperature_max"
    if parameter_matches_cmip6("temperature_min", p):
        return "temperature_min"
    return None


_BASE_parameter_matches_cmip6_FOR_PET_TEMPERATURE = parameter_matches_cmip6


def parameter_matches_cmip6(reference_parameter, cmip6_parameter):
    """Override: for potential_evaporation, accept temperature files as input source.

    This lets the downscaling workflow create CMIP6 PET internally from temperature_mean,
    temperature_max and temperature_min, instead of requiring separate PET CMIP6 files.
    """
    ref = normalize_parameter_name(reference_parameter)
    cmip = normalize_text(cmip6_parameter)

    if ref == "potential_evaporation":
        # Never use actual evaporation / evspsbl as PET.
        if cmip in {"evaporation", "evspsbl", "actual_evaporation", "evaporation_including_sublimation_and_transpiration"}:
            return False
        # Allow already-derived PET files if they exist, but temperature files are preferred in read_cmip6_records.
        if cmip in {"potential_evaporation", "potential_evapotranspiration", "pet", "eto", "et0"}:
            return True
        # Important: allow temperature files so PET can be derived internally.
        if cmip in {"temperature_mean", "temperature_max", "temperature_min", "tas", "tasmax", "tasmin", "tmean", "tmax", "tmin"}:
            return True
        return False

    return _BASE_parameter_matches_cmip6_FOR_PET_TEMPERATURE(reference_parameter, cmip6_parameter)


def days_in_month_array(years, months):
    return np.array([
        pd.Timestamp(year=int(y), month=int(m), day=1).days_in_month
        for y, m in zip(years, months)
    ], dtype=float)


def mid_month_day_of_year_array(years, months):
    return np.array([
        pd.Timestamp(year=int(y), month=int(m), day=15).dayofyear
        for y, m in zip(years, months)
    ], dtype=float)


def extraterrestrial_radiation_mm_day(latitude_deg, day_of_year):
    """FAO-56 extraterrestrial radiation converted to mm/day water equivalent."""
    lat_rad = np.deg2rad(np.asarray(latitude_deg, dtype=float))
    j = np.asarray(day_of_year, dtype=float)

    solar_constant = 0.0820  # MJ m-2 min-1
    inverse_relative_distance = 1.0 + 0.033 * np.cos(2.0 * np.pi * j / 365.0)
    solar_declination = 0.409 * np.sin(2.0 * np.pi * j / 365.0 - 1.39)

    x = -np.tan(lat_rad) * np.tan(solar_declination)
    x = np.clip(x, -1.0, 1.0)
    sunset_hour_angle = np.arccos(x)

    ra_mj_m2_day = (
        (24.0 * 60.0 / np.pi)
        * solar_constant
        * inverse_relative_distance
        * (
            sunset_hour_angle * np.sin(lat_rad) * np.sin(solar_declination)
            + np.cos(lat_rad) * np.cos(solar_declination) * np.sin(sunset_hour_angle)
        )
    )
    return 0.408 * ra_mj_m2_day


def calculate_pet_hargreaves_from_temperature(df):
    """Monthly PET in mm/month from Celsius temperature_mean/max/min and latitude."""
    tmean = pd.to_numeric(df["temperature_mean"], errors="coerce").astype(float).values
    tmax = pd.to_numeric(df["temperature_max"], errors="coerce").astype(float).values
    tmin = pd.to_numeric(df["temperature_min"], errors="coerce").astype(float).values
    lat = pd.to_numeric(df["lat"], errors="coerce").astype(float).values
    years = df["year"].astype(int).values
    months = df["month"].astype(int).values

    days = days_in_month_array(years, months)
    doy = mid_month_day_of_year_array(years, months)
    ra = extraterrestrial_radiation_mm_day(lat, doy)

    temp_range = np.maximum(tmax - tmin, 0.0)
    pet_day = 0.0023 * ra * (tmean + 17.8) * np.sqrt(temp_range)
    pet_month = pet_day * days
    return np.maximum(pet_month, 0.0)


def derive_pet_from_temperature_records(records_df, target_reference_parameter="potential_evaporation"):
    """Build a CMIP6-like PET dataframe from temperature_mean/max/min records.

    Input is a subset of the CMIP6 file index that may contain temperature and/or PET files.
    This function prefers temperature files and ignores actual evaporation.
    Output columns match read_cmip6_records().
    """
    if records_df is None or records_df.empty:
        return pd.DataFrame()

    records_df = records_df.copy()
    records_df["canonical_temperature_parameter"] = records_df["cmip6_parameter"].apply(canonical_temperature_parameter)
    temp_records = records_df[records_df["canonical_temperature_parameter"].isin(PET_TEMPERATURE_PARAMETERS)].copy()

    # Prefer temperature-derived PET. Direct PET files are only fallback if temperature triplet is unavailable.
    if not temp_records.empty:
        parts_by_param = []
        for temp_parameter in PET_TEMPERATURE_PARAMETERS:
            recs = temp_records[temp_records["canonical_temperature_parameter"] == temp_parameter].copy()
            if recs.empty:
                print_info(f"PET derivation: missing {temp_parameter} file(s), cannot derive PET from temperature for this group.")
                return pd.DataFrame()

            # Read each file using the standard CMIP6 reader, then collapse duplicates across historical/future overlap.
            df_temp = _BASE_read_cmip6_records_FOR_PET_TEMPERATURE(recs, temp_parameter)
            if df_temp.empty:
                return pd.DataFrame()
            df_temp = (
                df_temp
                .groupby(["point_id", "date", "year", "month"], as_index=False)
                .agg(
                    temp_value=("raw_value", "mean"),
                    lon=("lon", "first"),
                    lat=("lat", "first"),
                    model=("model", "first"),
                    scenario=("scenario", "first"),
                    source_file=("source_file", lambda s: ";".join(sorted(set(map(str, s)))))
                )
            )
            df_temp = df_temp.rename(columns={"temp_value": temp_parameter})
            parts_by_param.append(df_temp)

        merged = parts_by_param[0]
        for df_temp in parts_by_param[1:]:
            keep_cols = ["point_id", "date", "year", "month", df_temp.columns[df_temp.columns.str.startswith("temperature_")][0]]
            # Preserve lon/lat from the first temperature file, usually temperature_mean.
            merged = merged.merge(df_temp[keep_cols], on=["point_id", "date", "year", "month"], how="inner")

        if merged.empty:
            return pd.DataFrame()

        if "lat" not in merged.columns or merged["lat"].isna().all():
            raise ValueError("Cannot derive PET: latitude is missing from temperature files.")

        merged["raw_value"] = calculate_pet_hargreaves_from_temperature(merged)
        merged["reference_parameter"] = target_reference_parameter
        merged["cmip6_parameter"] = "potential_evaporation"
        merged["source_file"] = PET_SOURCE_DESCRIPTION + " | " + merged["source_file"].astype(str)

        # Force model/scenario from records where possible.
        if "model" not in merged.columns or merged["model"].isna().all():
            merged["model"] = records_df["model"].iloc[0]
        if "scenario" not in merged.columns or merged["scenario"].isna().all():
            merged["scenario"] = records_df["scenario"].iloc[0]

        return merged[[
            "reference_parameter", "model", "scenario", "point_id", "date", "year", "month",
            "raw_value", "lon", "lat", "cmip6_parameter", "source_file"
        ]]

    # Fallback: if no temperature files are provided, read direct PET files if available.
    pet_records = records_df[records_df["cmip6_parameter"].apply(lambda p: normalize_text(p) in {"potential_evaporation", "potential_evapotranspiration", "pet", "eto", "et0"})].copy()
    if not pet_records.empty:
        return _BASE_read_cmip6_records_FOR_PET_TEMPERATURE(pet_records, target_reference_parameter)

    return pd.DataFrame()


_BASE_read_cmip6_records_FOR_PET_TEMPERATURE = read_cmip6_records


def read_cmip6_records(records_df, reference_parameter):
    """Alternative method: derive potential_evaporation from temperature data when enabled."""
    ref = normalize_parameter_name(reference_parameter)
    if ref == "potential_evaporation" and PET_DERIVE_FROM_TEMPERATURE:
        print_info("Potential evaporation: deriving CMIP6 PET from temperature_mean/max/min inside downscaling code.")
        return derive_pet_from_temperature_records(records_df, target_reference_parameter="potential_evaporation")
    return _BASE_read_cmip6_records_FOR_PET_TEMPERATURE(records_df, reference_parameter)


# ============================================================
# 15. RUN SELECTED MODELS IN COMPLETELY SEPARATE OUTPUT FOLDERS
# ============================================================

BASE_MODEL_OUTPUT_ROOT = Path(
    r"D:\Morocco\morocco_data\Downscaling_CLASSICAL_MIROC6_ONLY_DERIVED_PET"
)

MODEL_OUTPUT_FOLDER_NAMES = {
    "cnrm_cm6_1": "CNRM_CM6_1",
    "miroc6": "MIROC6",
}


def configure_output_for_single_model(model_name):
    """Reset all output folders/log files so each model is saved completely separately."""
    global OUT_ROOT, STANDARD_DIR, QUALITY_DIR, FUTURE_DS_DIR, CHANGE_DIR, EXCEL_LIKE_DIR, LOG_DIR
    global LOG_FILE, ID_COVERAGE_FILE, MISSING_ID_FILE
    global DEBUG_REFERENCE_DETECTION_FILE, DEBUG_CMIP6_DETECTION_FILE, DEBUG_MATCHING_FILE
    global log_rows, id_coverage_rows, missing_id_rows, SELECTED_MODELS

    model_norm = normalize_model_name(model_name)
    folder_name = MODEL_OUTPUT_FOLDER_NAMES.get(model_norm, safe_name(model_norm).upper())

    OUT_ROOT = BASE_MODEL_OUTPUT_ROOT / folder_name
    STANDARD_DIR = OUT_ROOT / "01_standardized_inputs"
    QUALITY_DIR = OUT_ROOT / "02_downscaling_quality"
    FUTURE_DS_DIR = OUT_ROOT / "03_downscaled_future_monthly_csv"
    CHANGE_DIR = OUT_ROOT / "04_future_change_analysis"
    EXCEL_LIKE_DIR = OUT_ROOT / "05_excel_like_future_outputs"
    LOG_DIR = OUT_ROOT / "06_logs"

    for folder in [
        OUT_ROOT, STANDARD_DIR, QUALITY_DIR, FUTURE_DS_DIR,
        CHANGE_DIR, EXCEL_LIKE_DIR, LOG_DIR
    ]:
        folder.mkdir(parents=True, exist_ok=True)

    LOG_FILE = LOG_DIR / "classical_downscaling_log.csv"
    ID_COVERAGE_FILE = LOG_DIR / "id_coverage_report_by_stage.csv"
    MISSING_ID_FILE = LOG_DIR / "missing_id_details_by_stage.csv"

    DEBUG_REFERENCE_DETECTION_FILE = LOG_DIR / "reference_parameter_detection_debug.csv"
    DEBUG_CMIP6_DETECTION_FILE = LOG_DIR / "cmip6_parameter_detection_debug.csv"
    DEBUG_MATCHING_FILE = LOG_DIR / "pet_shortwave_matching_debug.csv"

    log_rows = []
    id_coverage_rows = []
    missing_id_rows = []
    SELECTED_MODELS = {model_norm}

    print_info("=" * 120)
    print_info(f"CONFIGURED SEPARATE OUTPUT FOR MODEL: {model_norm}")
    print_info(f"Output root: {OUT_ROOT}")
    print_info("=" * 120)


def main():
    """Run CNRM-CM6-1 and MIROC6 separately, with no mixed outputs."""
    models_to_run = ["miroc6"]
    for model_name in models_to_run:
        configure_output_for_single_model(model_name)
        run_workflow_once()

    print("\nALL SELECTED MODELS FINISHED.")
    print(f"Main output root: {BASE_MODEL_OUTPUT_ROOT}")
    print(f"MIROC6 results: {BASE_MODEL_OUTPUT_ROOT / 'MIROC6'}")


if __name__ == "__main__":
    main()