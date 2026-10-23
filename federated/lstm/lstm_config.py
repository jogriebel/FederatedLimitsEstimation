"""Configuration for Federated and Local Learning with LSTM and HPO."""
from general_config import DATA_BALANCE, RESULTS_DIR, HYPERPARAMS_DIR
MODEL = "LSTM"

# MLflow
MLFLOW_EXPERIMENT = "LSTM"
MLFLOW_SUFFIX = f"{DATA_BALANCE}_LSTM"


# LSTM HPO Space (for temporal models)
LSTM_HPO_SPACE = {
    # Architecture
    "hidden_dim": (32, 128, "int"),
    "num_layers": (1, 3, "int"),
    "dropout": (0.1, 0.5, "float"),
    
    # Optimizer
    "learning_rate": (0.0001, 0.01, "log"),
    "weight_decay": (1e-6, 1e-3, "log"),
    
    # Training
    "local_epochs": (10, 200, "int"),
    "batch_size": (16, 128, "int"),
}

# Output files
HYPERPARAMS_FILE_PATTERN = HYPERPARAMS_DIR / f"hyperparams_lstm_{{client}}_{DATA_BALANCE}.json"
FL_HYPERPARAMS_FILE = HYPERPARAMS_DIR / f"hyperparams_lstm_FL_global_{DATA_BALANCE}.json"
FL_BOOST_HYPERPARAMS_FILE_PATTERN = HYPERPARAMS_DIR / f"hyperparams_lstm_FL_boost_{{client}}_{DATA_BALANCE}.json"
LL_MODEL_PATH_PATTERN = RESULTS_DIR / "models" / "lstm_model_client_{client_id}.pt"
METRICS_CSV = RESULTS_DIR / f"local_metrics_lstm_{DATA_BALANCE}.csv"
FORMATTED_CSV = RESULTS_DIR / f"local_metrics_formatted_lstm_{DATA_BALANCE}.csv"