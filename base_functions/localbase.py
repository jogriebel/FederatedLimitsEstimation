import pandas as pd
import logging
from sklearn.metrics import (
    roc_auc_score, average_precision_score, recall_score, precision_score,
    f1_score, brier_score_loss, confusion_matrix, log_loss
)
import numpy as np
import json
import mlflow

from kappa_limits import compute_actual_kappa, compute_theoretical_kappa
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log_to_mlflow_local(results, MODEL, OPTUNA_TRIALS, DATA_BALANCE):
    """Log results to MLflow."""
    # Log parameters
    for k, v in results.items():
        if k.startswith("param_"):
            mlflow.log_param(k.replace("param_", ""), v)
    
    mlflow.log_param("optuna_trials", OPTUNA_TRIALS)
    mlflow.log_param("client_id", results["client_id"])
    mlflow.log_param("client_name", results["client_name"])
    mlflow.log_param("data_balance", DATA_BALANCE)
    mlflow.log_param("model_type", MODEL)
    
    # Log metrics
    for k, v in results.items():
        if k.startswith(("test_", "theoretical", "optuna")):
            if isinstance(v, (int, float)) and v is not None and not np.isnan(v):
                mlflow.log_metric(k, v)

def save_results(all_results, METRICS_CSV, RESULTS_DIR, FORMATTED_CSV):
    """Save results to CSV files."""
    RESULTS_DIR.mkdir(exist_ok=True)
    
    # Comprehensive metrics
    df = pd.DataFrame(all_results)
    df.to_csv(METRICS_CSV, index=False)
    logger.info(f"Saved comprehensive metrics to {METRICS_CSV}")
    
    # Formatted summary
    summary_cols = [
        "client_id", "client_name", "test_AUC", "test_AUPRC", "test_F1",
        "test_Precision", "test_Recall", "test_Sensitivity", "test_Specificity",
        "test_Brier", "test_log_loss", "test_actual_kappa", "theoretical_kappa", 
        "bayes_error", "jsd_before", "jsd_after"
    ]
    summary_cols += [f"{c}_ci_{b}" for c in summary_cols if c.startswith("test_") for b in ("low", "high")]
    summary = df[[c for c in summary_cols if c in df.columns]]
    summary.to_csv(FORMATTED_CSV, index=False)
    logger.info(f"Saved formatted metrics to {FORMATTED_CSV}")
    
    
def compute_all_metrics(y_true: np.ndarray, y_prob: np.ndarray, 
                       threshold: float, n_boot: int = 1000,
                       alpha: float = 0.05, seed: int = 42, ref=None) -> dict:
    """Compute all classification metrics with given threshold."""
    y_pred = (y_prob > threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0
    
    results = {
        "AUC": roc_auc_score(y_true, y_prob),
        "AUPRC": average_precision_score(y_true, y_prob),
        "F1": f1_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred),
        "Sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0,
        "Specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
        "Brier": brier_score_loss(y_true, y_prob),
        "log_loss": log_loss(y_true, y_prob),
        "actual_kappa": compute_actual_kappa(y_true, y_pred),
        "threshold": threshold
    }

    # Class-stratified bootstrap percentile CIs (model and threshold held fixed)
    if n_boot > 0:
        y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
        rng = np.random.default_rng(seed)
        pos, neg = np.where(y_true == 1)[0], np.where(y_true != 1)[0]
        boots, refs = [], []
        for _ in range(n_boot):
            idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
            boots.append(compute_all_metrics(y_true[idx], y_prob[idx], threshold, n_boot=0))
            if ref is not None:
                refs.append(compute_all_metrics(y_true[idx], np.asarray(ref[0])[idx], ref[1], n_boot=0))
        for k in [k for k in results if k != "threshold"]:
            vals = np.array([b[k] for b in boots], dtype=float)
            results[f"{k}_ci_low"], results[f"{k}_ci_high"] = np.nanpercentile(
                vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
            if refs:
                diffs = vals - np.array([r[k] for r in refs], dtype=float)
                results[f"{k}_diff_ci_low"], results[f"{k}_diff_ci_high"] = np.nanpercentile(
                    diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    if ref is not None:
        base = compute_all_metrics(y_true, ref[0], ref[1], n_boot=0)
        results.update({f"{k}_diff": results[k] - base[k] for k in base if k != "threshold"})
    return results
    
# =============================================================================
# Hyperparameter Persistence
# =============================================================================

def get_hyperparams_path(client_name, HYPERPARAMS_FILE_PATTERN, DATA_BALANCE) -> str:
    """Get path for hyperparameters file."""
    return str(HYPERPARAMS_FILE_PATTERN).format(client=client_name, balance=DATA_BALANCE)


def save_hyperparams(client_name, params, HYPERPARAMS_FILE_PATTERN, DATA_BALANCE):
    """Save hyperparameters to JSON file."""
    path = get_hyperparams_path(client_name, HYPERPARAMS_FILE_PATTERN, DATA_BALANCE)
    with open(path, 'w') as f:
        json.dump(params, f, indent=2)
    logger.info(f"Saved hyperparameters to {path}")


def load_hyperparams(client_name, HYPERPARAMS_FILE_PATTERN, DATA_BALANCE) -> dict:
    """Load hyperparameters from JSON file if exists."""
    path = get_hyperparams_path(client_name, HYPERPARAMS_FILE_PATTERN, DATA_BALANCE)
    try:
        with open(path, 'r') as f:
            params = json.load(f)
        logger.info(f"Loaded hyperparameters from {path}")
        return params
    except FileNotFoundError:
        return None


def get_theoretical_kappa_path(client_name, THEORETICAL_KAPPA_FILE_PATTERN, DATA_BALANCE) -> str:
    """Get path for theoretical kappa file."""
    return str(THEORETICAL_KAPPA_FILE_PATTERN).format(client=client_name, balance=DATA_BALANCE)


def save_theoretical_kappa(client_name, kappa, HYPERPARAMS_DIR, THEORETICAL_KAPPA_FILE_PATTERN, DATA_BALANCE):
    """Save theoretical kappa to JSON file."""
    HYPERPARAMS_DIR.mkdir(exist_ok=True)
    path = get_theoretical_kappa_path(client_name, THEORETICAL_KAPPA_FILE_PATTERN, DATA_BALANCE)
    with open(path, 'w') as f:
        json.dump({"theoretical_kappa": kappa}, f, indent=2)
    logger.info(f"Saved theoretical kappa to {path}")


def load_theoretical_kappa(client_name,THEORETICAL_KAPPA_FILE_PATTERN, DATA_BALANCE) -> dict:
    """Load theoretical kappa from JSON file if exists."""
    path = get_theoretical_kappa_path(client_name, THEORETICAL_KAPPA_FILE_PATTERN, DATA_BALANCE)
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        logger.info(f"Loaded theoretical kappa from {path}")
        return data
    except FileNotFoundError:
        return None

def check_or_compute_kappa(compute_limits, client_name, THEORETICAL_KAPPA_FILE_PATTERN, DATA_BALANCE, X_test_df, X_train_df, X_val_df, y_test_np, y_train_np, y_val_np, HYPERPARAMS_DIR):
    # Compute theoretical limits
    theoretical_kappa= 0.0
    if compute_limits:
        existing_kappa = load_theoretical_kappa(client_name, THEORETICAL_KAPPA_FILE_PATTERN, DATA_BALANCE)
        if existing_kappa:
            theoretical_kappa = existing_kappa["theoretical_kappa"]
            logger.info(f"Using cached theoretical kappa: {theoretical_kappa:.4f}")
        else:
            logger.info("Computing theoretical limits...")
            # Flatten for theoretical limits computation
            X_all = np.concatenate([X_test_df.values, X_train_df.values, X_val_df.values])
            y_all = np.concatenate([y_test_np, y_train_np, y_val_np])
            kappa_results = compute_theoretical_kappa(X_all, y_all)
            theoretical_kappa = kappa_results.get("theoretical_kappa", 0.0)
            save_theoretical_kappa(client_name, theoretical_kappa, HYPERPARAMS_DIR,THEORETICAL_KAPPA_FILE_PATTERN, DATA_BALANCE)
    return theoretical_kappa