"""Configuration for Federated and Local Learning with Logistic Regression and HPO."""
from general_config import DATA_BALANCE, RESULTS_DIR, HYPERPARAMS_DIR
MODEL = "LR"

# MLflow
MLFLOW_EXPERIMENT = "LR"
MLFLOW_SUFFIX = f"{DATA_BALANCE}_lr"

# Logistic Regression HPO Space (shared between FL and local learning)
LR_HPO_SPACE = {
    # Optimizer
    "learning_rate": (0.001, 0.3, "log"),
    "weight_decay": (1e-5, 1e-2, "log"),
    
    # Training
    "local_epochs": (2, 150, "int"),
    "batch_size": (16, 128, "int"),
    
    # FedProx
    "proximal_mu": (0.0, 1.0, "float"),
}



# Output files

HYPERPARAMS_FILE_PATTERN = HYPERPARAMS_DIR / f"hyperparams_lr_{{client}}_{DATA_BALANCE}.json"
FL_HYPERPARAMS_FILE = HYPERPARAMS_DIR / f"hyperparams_lr_FL_global_{DATA_BALANCE}.json"
FL_BOOST_HYPERPARAMS_FILE_PATTERN = HYPERPARAMS_DIR / f"hyperparams_lr_FL_boost_{{client}}_{DATA_BALANCE}.json"
LL_MODEL_PATH_PATTERN = RESULTS_DIR / "models" / "lr_model_client_{client_id}.pt"
METRICS_CSV = RESULTS_DIR / f"local_metrics_lr_{DATA_BALANCE}.csv"
FORMATTED_CSV = RESULTS_DIR / f"local_metrics_formatted_lr_{DATA_BALANCE}.csv"