"""Configuration for XGBoost with HPO."""
from general_config import DATA_BALANCE, RESULTS_DIR, HYPERPARAMS_DIR, RANDOM_STATE
MODEL = "XGB"

#For FL only
XGB_NUM_BINS = 256
XGB_BOOST_TREES = 2


MLFLOW_EXPERIMENT = "XGB"
MLFLOW_SUFFIX = f"{DATA_BALANCE}_xgb"

XGBOOST_FIXED = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbosity": 0,
    "n_estimators": 1000,        # cap only; early stopping picks the count
    "scale_pos_weight": 1.0,     # threshold tuning already handles imbalance
}

XGB_EARLY_STOPPING_ROUNDS = 10   

XGB_HPO_SPACE = {
    "learning_rate":    (0.01, 0.1, "log"),
    "max_depth":        (2, 4, "int"),
    "min_child_weight": (10, 50, "int"),
    "gamma":            (0.0, 5.0, "float"),
    "subsample":        (0.4, 0.7, "float"),
    "colsample_bytree": (0.5, 0.9, "float"),
    "reg_alpha":        (0.0, 1.0, "float"),
    "reg_lambda":       (1.0, 50.0, "log"),
    "threshold":        (0.3, 0.7, "float"), 
    "max_bin": (16,64, "int"),
}

# Output files
HYPERPARAMS_FILE_PATTERN = HYPERPARAMS_DIR / f"hyperparams_xgb_{{client}}_{DATA_BALANCE}.json"
FL_HYPERPARAMS_FILE = HYPERPARAMS_DIR / f"hyperparams_xgb_FL_global_{DATA_BALANCE}.json"
FL_BOOST_HYPERPARAMS_FILE_PATTERN = HYPERPARAMS_DIR / f"hyperparams_xgb_FL_boost_{{client}}_{DATA_BALANCE}.json"
LL_MODEL_PATH_PATTERN = RESULTS_DIR / "models" / "xgb_model_client_{client_id}_{balance}.json"
METRICS_CSV = RESULTS_DIR / f"local_metrics_xgb_{DATA_BALANCE}.csv"
FORMATTED_CSV = RESULTS_DIR / f"local_metrics_formatted_xgb_{DATA_BALANCE}.csv"
