"""Configuration for Federated and Local Learning with Neural Network and HPO."""
from general_config import DATA_BALANCE, RESULTS_DIR, HYPERPARAMS_DIR
MODEL = "NN"

# MLflow
MLFLOW_EXPERIMENT = "MLP"
MLFLOW_SUFFIX = f"{DATA_BALANCE}_mlp"

# Neural Network HPO Space
NN_HPO_SPACE = {
    # Architecture
    "hidden_dim": (16, 256, "int"),
    "dropout": (0.1, 0.4, "float"),
    
    # Optimizer
    "learning_rate": (0.001, 0.3, "log"),
    "weight_decay": (1e-5, 1e-3, "log"),
    "adam_beta1": (0.75, 0.95, "float"),
    "adam_beta2": (0.89, 0.9999, "float"),
    
    # Training
    "local_epochs": (2, 150, "int"),
    "batch_size": (16, 128, "int"),
    "grad_clip_norm": (0.5, 5.0, "float"),
    
    # Regularization
    "label_smoothing": (0.0, 0.7, "float"),
    "lr_warmup_epochs": (0, 10, "int"),
}

# Output files
HYPERPARAMS_FILE_PATTERN = HYPERPARAMS_DIR / f"hyperparams_mlp_{{client}}_{DATA_BALANCE}.json"
FL_HYPERPARAMS_FILE = HYPERPARAMS_DIR / f"hyperparams_mlp_FL_global_{DATA_BALANCE}.json"
FL_BOOST_HYPERPARAMS_FILE_PATTERN = HYPERPARAMS_DIR / f"hyperparams_mlp_FL_boost_{{client}}_{DATA_BALANCE}.json"
LL_MODEL_PATH_PATTERN = RESULTS_DIR / "models" / "mlp_model_client_{client_id}.pt"
METRICS_CSV = RESULTS_DIR / f"local_metrics_mlp_{DATA_BALANCE}.csv"
FORMATTED_CSV = RESULTS_DIR / f"local_metrics_formatted_mlp_{DATA_BALANCE}.csv"
