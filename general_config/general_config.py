from pathlib import Path
POOL_SIZE = 5


# Data
# Cohort used by every script: "20" (balanced to 20% transfusions) or "natural".
# This is the only place where it is set.
DATA_BALANCE = "natural"
CLIENT_NAMES = {1: "eicu", 2: "mimic", 3: "umc", 4: "hirid", 5: "uka"}
RANDOM_STATE = 42

# Paths (relative to the repository root, so scripts work from any directory)
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "FLdata"
RESULTS_DIR = ROOT / "FLresults"
HYPERPARAMS_DIR = ROOT / "FLhyperparams"
RESULTS_DIR.mkdir(exist_ok=True)
HYPERPARAMS_DIR.mkdir(exist_ok=True)
THEORETICAL_KAPPA_FILE_PATTERN = HYPERPARAMS_DIR / f"theoretical_kappa_{{client}}_{DATA_BALANCE}.json"
FL_METRICS_CSV_PATTERN = RESULTS_DIR / f"FL_metrics_{{modelname}}_client_{{client_id}}_{DATA_BALANCE}.csv"


def get_file_pattern():
    if DATA_BALANCE == "natural":
        return f"X_{{client}}_natural.parquet", f"y_{{client}}_natural.parquet"
    return f"X_{{client}}_balanced_morevariables_{DATA_BALANCE}pct.parquet", \
           f"y_{{client}}_balanced_morevariables_{DATA_BALANCE}pct.parquet"

X_PATTERN, Y_PATTERN = get_file_pattern()




# Local Learning Settings
OPTUNA_TRIALS = 50
USE_EXISTING_HYPERPARAMS = False

# Federated Learning
FL_NUM_ROUNDS = 600
FL_LOCAL_EPOCHS = 20 #only as backup, will be tuned during HPO phase
FL_BATCH_SIZE = 32

# HPO Settings 
FL_HPO_FACTOR = 0.8
FL_HPO_RATE = 10

# Theoretical Kappa Settings
MAX_FEATURES = 8
N_BINS = 16
N_ITERATIONS  = 5

# Federated kappa_max, written by federated/kappa/server.py after the kappa round.
FEDERATED_KAPPA_FILE = RESULTS_DIR / f"federated_kappa_{DATA_BALANCE}.json"

