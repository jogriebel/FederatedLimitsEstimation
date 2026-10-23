"""Main training script with Optuna hyperparameter tuning and XGBoost."""
import logging
import numpy as np
import torch
import optuna
import mlflow
import xgboost as xgb
from sklearn.metrics import f1_score, log_loss, roc_auc_score, average_precision_score
#custom imports
from base_functions import (save_results, compute_all_metrics, 
                            load_hyperparams, save_hyperparams, 
                            check_or_compute_kappa, log_to_mlflow_local
                            )

from xgb_config import (
    MLFLOW_EXPERIMENT, MLFLOW_SUFFIX,
    METRICS_CSV, FORMATTED_CSV, 
    XGB_HPO_SPACE, HYPERPARAMS_FILE_PATTERN, MODEL, XGBOOST_FIXED, LL_MODEL_PATH_PATTERN,
    XGB_EARLY_STOPPING_ROUNDS
)
from general_config import (RANDOM_STATE, USE_EXISTING_HYPERPARAMS, THEORETICAL_KAPPA_FILE_PATTERN, 
                            OPTUNA_TRIALS, RESULTS_DIR, DATA_BALANCE, HYPERPARAMS_DIR)
from mydataloading import prepare_data, get_client_name

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Early stopping on validation log-loss
EARLY_STOPPING = {"eval_metric": "logloss", "early_stopping_rounds": XGB_EARLY_STOPPING_ROUNDS}


# =============================================================================
# MLflow helper (logs into the already active client run, no new runs)
# =============================================================================

def log_split_metrics(results: dict, prefixes=("train_", "val_")) -> None:
    """Log numeric train_/val_ entries of results as metrics on the active run.
    test_ entries are already logged by log_to_mlflow_local."""
    metrics = {}
    for k, v in results.items():
        if k.startswith(prefixes) and isinstance(v, (int, float, np.integer, np.floating)) \
                and not isinstance(v, bool) and np.isfinite(v):
            metrics[k] = float(v)
    if metrics:
        mlflow.log_metrics(metrics)


# =============================================================================
# Optuna Objective
# =============================================================================

def create_objective(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
    """Create Optuna objective function."""
    
    def objective(trial):
        # Sample hyperparameters from XGB_HPO_SPACE
        params = {**XGBOOST_FIXED, **EARLY_STOPPING}
        
        for param_name, (low, high, param_type) in XGB_HPO_SPACE.items():
            if param_name == "threshold":
                # Threshold is selected after tuning (not an XGBoost parameter)
                continue
            
            if param_type == "log":
                params[param_name] = trial.suggest_float(param_name, low, high, log=True)
            elif param_type == "int":
                params[param_name] = trial.suggest_int(param_name, low, high)
            elif param_type == "float":
                params[param_name] = trial.suggest_float(param_name, low, high)
        
        # Train on training split, score on validation split
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
        val_probs = model.predict_proba(X_val)[:, 1]
        train_probs = model.predict_proba(X_train)[:, 1]
        val_logloss = log_loss(y_val, val_probs)
        
        # Log this trial to the active client run; step = trial number
        trial_metrics = {
            "trial_val_logloss": val_logloss,
            "trial_val_AUC": roc_auc_score(y_val, val_probs),
            "trial_val_AUPRC": average_precision_score(y_val, val_probs),
            "trial_train_logloss": log_loss(y_train, train_probs),
            "trial_train_AUC": roc_auc_score(y_train, train_probs),
            "trial_train_AUPRC": average_precision_score(y_train, train_probs),
            "trial_best_iteration": float(model.best_iteration),
            **{f"trial_param_{k}": float(v) for k, v in trial.params.items()},
        }
        mlflow.log_metrics(trial_metrics, step=trial.number)
        
        return val_logloss
    
    return objective

# =============================================================================
# Training Pipeline
# =============================================================================

def train_client(client_id: int, compute_limits: bool = True) -> dict:
    """Train model for a single client with Optuna tuning on the validation split."""
    client_name = get_client_name(client_id)
    logger.info(f"Training client {client_id} ({client_name})")
    
    # Load data (70/10/20 split)
    X_train_df, y_train, X_test_df, y_test, X_val_df, y_val, feature_names = prepare_data(client_id)
    
    # Convert to numpy
    X_train_np = np.asarray(X_train_df, dtype=np.float32)
    X_val_np = np.asarray(X_val_df, dtype=np.float32)
    X_test_np = np.asarray(X_test_df, dtype=np.float32)
    y_train_np = np.asarray(y_train, dtype=np.float32).ravel()
    y_val_np = np.asarray(y_val, dtype=np.float32).ravel()
    y_test_np = np.asarray(y_test, dtype=np.float32).ravel()
    
    # Handle NaN/Inf
    X_train_np = np.nan_to_num(X_train_np, nan=0.0, posinf=0.0, neginf=0.0)
    X_val_np = np.nan_to_num(X_val_np, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_np = np.nan_to_num(X_test_np, nan=0.0, posinf=0.0, neginf=0.0)
    
    logger.info(f"Client {client_id}: {len(X_train_np)} train, {len(X_val_np)} val, {len(X_test_np)} test")
    
    # Compute theoretical limits (or load from file)
    theoretical_kappa = check_or_compute_kappa(compute_limits, client_name, THEORETICAL_KAPPA_FILE_PATTERN, DATA_BALANCE, X_test_df, X_train_df, X_val_df, y_test_np, y_train_np, y_val_np, HYPERPARAMS_DIR)
    # Get hyperparameters: load existing or run Optuna
    existing_params = load_hyperparams(client_name, HYPERPARAMS_FILE_PATTERN, DATA_BALANCE) if USE_EXISTING_HYPERPARAMS else None
    
    if existing_params:
        best_params = existing_params
        optuna_best_score = None
        logger.info("Using existing hyperparameters")
    else:
        logger.info(f"Running Optuna optimization ({OPTUNA_TRIALS} trials)...")
        study = optuna.create_study(
            direction="minimize", 
            sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
        )
        study.optimize(
            create_objective(X_train_np, y_train_np, X_val_np, y_val_np), 
            n_trials=OPTUNA_TRIALS, 
            show_progress_bar=True
        )
        
        best_params = study.best_params
        optuna_best_score = study.best_value
        logger.info(f"Best score: {optuna_best_score:.4f}")
        mlflow.log_metric("optuna_best_trial", float(study.best_trial.number))
        
        # Save hyperparameters
        save_hyperparams(client_name, study.best_params, HYPERPARAMS_FILE_PATTERN, DATA_BALANCE)
    
    # Prepare XGBoost params (drop any stored threshold)
    best_params.pop("threshold", None)
    xgb_params = {**XGBOOST_FIXED, **best_params, **EARLY_STOPPING}
    
    # Train final model on full training set
    logger.info("Training final model on full training set...")
    model = xgb.XGBClassifier(**xgb_params)
    model.fit(X_train_np, y_train_np, eval_set=[(X_val_np, y_val_np)], verbose=False)
    logger.info(f"Early stopping: best iteration {model.best_iteration} of {xgb_params['n_estimators']}")
    
    # Select threshold maximizing F1 on validation set
    val_probs = model.predict_proba(X_val_np)[:, 1]
    grid = np.linspace(0.05, 0.95, 91)
    threshold = float(grid[np.argmax([f1_score(y_val_np, (val_probs > t).astype(int), zero_division=0) for t in grid])])
    best_params["threshold"] = threshold
    
    # Train and validation metrics with the same threshold (diagnostics only)
    train_probs = model.predict_proba(X_train_np)[:, 1]
    train_metrics = compute_all_metrics(y_train_np, train_probs, threshold)
    val_metrics = compute_all_metrics(y_val_np, val_probs, threshold)
    
    # Final test set evaluation (only once on held-out test set)
    test_probs = model.predict_proba(X_test_np)[:, 1]
    test_metrics = compute_all_metrics(y_test_np, test_probs, threshold)
    
    # Save model
    model_dir = RESULTS_DIR / "models"
    model_dir.mkdir(exist_ok=True)
    model_path = str(LL_MODEL_PATH_PATTERN).format(client_id=client_id, balance=DATA_BALANCE)
    model.save_model(model_path)
    logger.info(f"Saved model to {model_path}")
    
    # Compile results
    results = {
        "client_id": client_id,
        "client_name": client_name,
        "theoretical_kappa": theoretical_kappa,
        "optuna_best_score": optuna_best_score,
        "best_iteration": model.best_iteration,
        **{f"train_{k}": v for k, v in train_metrics.items()},
        **{f"val_{k}": v for k, v in val_metrics.items()},
        **{f"test_{k}": v for k, v in test_metrics.items()},
        **{f"param_{k}": v for k, v in best_params.items()},
    }
    
    return results, model, test_probs



# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main execution function."""
    logger.info("Starting ML pipeline with XGBoost and Optuna tuning")
    
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    all_results = []
    
    for client_id in range(1, 6):
        run_name = f"client_xgb_{client_id}_{get_client_name(client_id)}_{MLFLOW_SUFFIX}"
        
        with mlflow.start_run(run_name=run_name):
            try:
                results, model, probs = train_client(client_id, compute_limits=True)
                log_to_mlflow_local(results, MODEL, OPTUNA_TRIALS, DATA_BALANCE)
                log_split_metrics(results)
                all_results.append(results)
                
                # Log efficiency
                if results["theoretical_kappa"] > 0:
                    efficiency = results["test_actual_kappa"] / results["theoretical_kappa"] * 100
                    logger.info(f"Client {client_id} - Efficiency: {efficiency:.1f}%")
                    mlflow.log_metric("kappa_efficiency", efficiency)
                    mlflow.log_metric("train_kappa_efficiency",
                                      results["train_actual_kappa"] / results["theoretical_kappa"] * 100)
                    mlflow.log_metric("val_kappa_efficiency",
                                      results["val_actual_kappa"] / results["theoretical_kappa"] * 100)
                
                logger.info(f"Client {client_id} completed: AUC={results['test_AUC']:.4f}, "
                           f"AUPRC={results['test_AUPRC']:.4f}")
                
            except Exception as e:
                logger.error(f"Error processing client {client_id}: {e}")
                raise
    
    save_results(all_results, METRICS_CSV, RESULTS_DIR, FORMATTED_CSV)
    logger.info("Pipeline completed successfully!")


if __name__ == "__main__":
    main()