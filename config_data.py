"""Configuration for blood transfusion prediction data preprocessing."""
from typing import Literal
from general_config import DATA_BALANCE, DATA_DIR as OUTPUT_DIR
# Paths
DATA_DIR = "./"  # folder with the extracted raw data
OUTPUT_DIR.mkdir(exist_ok=True)
PROJECT_NAME = "blood_fl"
# Databases to process
DATABASES = ["hirid", "mimic", "uka", "eicu", "umc"]
# Observation window settings (hours before event)
OBSERVATION_START = 6  # Start of observation window (hours before event)
OBSERVATION_END = 2    # End of observation window (hours before event)
MIN_OBSERVATION_LENGTH = 3  # Minimum required observations per patient
# Derived values
OBSERVATION_LENGTH = OBSERVATION_START - OBSERVATION_END
PREDICTION_WINDOW = OBSERVATION_END
# Data quality settings
COMPLETENESS_THRESHOLD = 30.0  # Minimum % of non-null values per row
# Class balance, derived from general_config.DATA_BALANCE ("natural" or e.g. 0.2)
POSITIVE_RATIO: float | Literal["natural"] = "natural" if DATA_BALANCE == "natural" else int(DATA_BALANCE) / 100
# Feature definitions
VITAL_SIGNS = [
'Hemoglobin [Mass/volume] in Blood',
'Potassium [Moles/volume] in Blood',
'Bicarbonate [Moles/volume] in Blood',
'Platelets [#/volume] in Blood',
'Glucose [Moles/volume] in Blood',
'Body temperature',
'Albumin [Mass/volume] in Body fluid',
'Mean blood pressure',
'Heart rate',
'White blood cell count',
'Respiratory rate',
'Creatinine [Moles/volume] in Blood',
'Systolic blood pressure',
]
STATIC_FEATURES = [
'age', 'weight', 'height', 'gender_concept_id',
'Glasgow coma score eye opening',
'Glasgow coma score motor',
'Glasgow coma score verbal'
]
CATEGORICAL_FEATURES = [
'gender_concept_id',
'Glasgow coma score eye opening',
'Glasgow coma score motor',
'Glasgow coma score verbal'
]
def get_ratio_suffix(positive_ratio: float | Literal["natural"]) -> str:
    """Generate filename suffix based on positive ratio setting."""
    if positive_ratio == "natural":
        return "natural"
    return f"balanced_morevariables_{int(positive_ratio * 100)}pct"

def validate_config():
    """Validate configuration and print summary."""
    assert OBSERVATION_START > OBSERVATION_END, \
    "OBSERVATION_START must be greater than OBSERVATION_END"
    assert 0 <= COMPLETENESS_THRESHOLD <= 100, \
    "COMPLETENESS_THRESHOLD must be between 0 and 100"
    assert 1 <= MIN_OBSERVATION_LENGTH <= OBSERVATION_LENGTH, \
    f"MIN_OBSERVATION_LENGTH must be between 1 and {OBSERVATION_LENGTH}"
    if isinstance(POSITIVE_RATIO, float):
        assert 0 < POSITIVE_RATIO < 1, "POSITIVE_RATIO must be between 0 and 1"
    print("=" * 60)
    print("CONFIGURATION SUMMARY")
    print("=" * 60)
    print(f"Observation window: {OBSERVATION_START}h to {OBSERVATION_END}h before event")
    print(f"  → Observation length: {OBSERVATION_LENGTH}h (timepoints: _1 to _{OBSERVATION_LENGTH})")
    print(f"  → Min observation length: {MIN_OBSERVATION_LENGTH}h")
    print(f"  → Prediction window: {PREDICTION_WINDOW}h")
    print(f"Completeness threshold: {COMPLETENESS_THRESHOLD}%")
    print(f"Positive ratio: {POSITIVE_RATIO}")
    print(f"Databases: {DATABASES}")
    print("=" * 60)