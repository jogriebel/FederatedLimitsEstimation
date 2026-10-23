"""Main training script with Optuna hyperparameter tuning, XGBoost, and 5-fold CV."""
import logging
import numpy as np
import torch
import optuna
import mlflow
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import average_precision_score, f1_score
#custom imports
from base_functions import (save_results, compute_all_metrics, 
                            load_hyperparams, save_hyperparams, 
                            check_or_compute_kappa, log_to_mlflow_local
                            )

from xgb_config import (
    MLFLOW_EXPERIMENT, MLFLOW_SUFFIX,
    METRICS_CSV, FORMATTED_CSV, 
    XGB_HPO_SPACE, HYPERPARAMS_FILE_PATTERN, MODEL, XGBOOST_FIXED, LL_MODEL_PATH_PATTERN
)
from general_config import (RANDOM_STATE, USE_EXISTING_HYPERPARAMS, THEORETICAL_KAPPA_FILE_PATTERN, CV_FOLDS, 
                            OPTUNA_TRIALS, RESULTS_DIR, DATA_BALANCE, HYPERPARAMS_DIR)
from mydataloading import prepare_data, get_client_name

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# =============================================================================
# Optuna Objective
# =============================================================================

def create_objective(X: np.ndarray, y: np.ndarray):
    """Create Optuna objective function."""
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)
    
    def objective(trial):
        # Sample hyperparameters from XGB_HPO_SPACE
        params = {**XGBOOST_FIXED}
        threshold = None
        
        for param_name, (low, high, param_type) in XGB_HPO_SPACE.items():
            if param_name == "threshold":
                # Handle threshold separately (not an XGBoost parameter)
                threshold = trial.suggest_float(param_name, low, high)
                continue
            
            if param_type == "log":
                params[param_name] = trial.suggest_float(param_name, low, high, log=True)
            elif param_type == "int":
                params[param_name] = trial.suggest_int(param_name, low, high)
            elif param_type == "float":
                params[param_name] = trial.suggest_float(param_name, low, high)
        
        # Cross-validation
        model = xgb.XGBClassifier(**params)
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        
        cv_probs = cross_val_predict(model, X_arr, y_arr, cv=cv, method='predict_proba')[:, 1]
        
        auprc = average_precision_score(y_arr, cv_probs)
        y_pred = (cv_probs >= threshold).astype(int)
        f1 = f1_score(y_arr, y_pred)
        
        # Weighted combination
        return 0.5 * auprc + 0.5 * f1
    
    return objective

# =============================================================================
# Training Pipeline
# =============================================================================

def train_client(client_id: int, compute_limits: bool = True) -> dict:
    """Train model for a single client with Optuna tuning and 5-fold CV."""
    client_name = get_client_name(client_id)
    logger.info(f"Training client {client_id} ({client_name})")
    
    # Load data (80/20 split)
    X_train_df, y_train, X_test_df, y_test, feature_names = prepare_data(client_id, "local")
    
    # Convert to numpy
    X_train_np = np.asarray(X_train_df, dtype=np.float32)
    X_test_np = np.asarray(X_test_df, dtype=np.float32)
    y_train_np = np.asarray(y_train, dtype=np.float32).ravel()
    y_test_np = np.asarray(y_test, dtype=np.float32).ravel()
    
    # Handle NaN/Inf
    X_train_np = np.nan_to_num(X_train_np, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_np = np.nan_to_num(X_test_np, nan=0.0, posinf=0.0, neginf=0.0)
    
    logger.info(f"Client {client_id}: {len(X_train_np)} train, {len(X_test_np)} test")
    
    # Compute theoretical limits (or load from file)
    theoretical_kappa = check_or_compute_kappa(compute_limits, client_name, THEORETICAL_KAPPA_FILE_PATTERN, DATA_BALANCE, X_test_df, X_train_df, y_test_np, y_train_np, HYPERPARAMS_DIR)
    # Get hyperparameters: load existing or run Optuna
    existing_params = load_hyperparams(client_name, HYPERPARAMS_FILE_PATTERN, DATA_BALANCE) if USE_EXISTING_HYPERPARAMS else None
    
    if existing_params:
        best_params = existing_params
        optuna_best_score = None
        logger.info("Using existing hyperparameters")
    else:
        logger.info(f"Running Optuna optimization ({OPTUNA_TRIALS} trials)...")
        study = optuna.create_study(
            direction="maximize", 
            sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
        )
        study.optimize(
            create_objective(X_train_np, y_train_np), 
            n_trials=OPTUNA_TRIALS, 
            show_progress_bar=True
        )
        
        best_params = study.best_params
        optuna_best_score = study.best_value
        logger.info(f"Best score: {optuna_best_score:.4f}")
        
        # Save hyperparameters
        save_hyperparams(client_name, study.best_params, HYPERPARAMS_FILE_PATTERN, DATA_BALANCE)
    
    # Extract threshold and prepare XGBoost params
    threshold = best_params.pop("threshold", 0.5)
    xgb_params = {**XGBOOST_FIXED, **best_params}
    
    # Train final model on full training set
    logger.info("Training final model on full training set...")
    model = xgb.XGBClassifier(**xgb_params)
    model.fit(X_train_np, y_train_np, verbose=False)
    
    # 5-fold CV metrics on training data (for reporting)
    logger.info("Computing 5-fold CV metrics...")
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_probs = cross_val_predict(model, X_train_np, y_train_np, cv=cv, method='predict_proba')[:, 1]
    cv_metrics = compute_all_metrics(y_train_np, cv_probs, threshold)
    
    # Final test set evaluation (only once on held-out test set)
    test_probs = model.predict_proba(X_test_np)[:, 1]
    test_metrics = compute_all_metrics(y_test_np, test_probs, threshold)
    
    # Save model
    model_dir = RESULTS_DIR / "models"
    model_dir.mkdir(exist_ok=True)
    model_path = str(LL_MODEL_PATH_PATTERN).format(client_id=client_id, balance=DATA_BALANCE)
    model.save_model(model_path)
    logger.info(f"Saved model to {model_path}")
    
    # Add threshold back to best_params for logging
    best_params["threshold"] = threshold
    
    # Compile results
    results = {
        "client_id": client_id,
        "client_name": client_name,
        "theoretical_kappa": theoretical_kappa,
        "optuna_best_score": optuna_best_score,
        **{f"cv_{k}": v for k, v in cv_metrics.items()},
        **{f"test_{k}": v for k, v in test_metrics.items()},
        **{f"param_{k}": v for k, v in best_params.items()},
    }
    
    return results, model, test_probs



# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main execution function."""
    logger.info("Starting ML pipeline with XGBoost, Optuna tuning and 5-fold CV")
    
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    all_results = []
    
    for client_id in range(1, 6):
        run_name = f"client_xgb_{client_id}_{get_client_name(client_id)}_{MLFLOW_SUFFIX}"
        
        with mlflow.start_run(run_name=run_name):
            try:
                results, model, probs = train_client(client_id, compute_limits=True)
                log_to_mlflow_local(results, MODEL, CV_FOLDS, OPTUNA_TRIALS, DATA_BALANCE)
                all_results.append(results)
                
                # Log efficiency
                if results["theoretical_kappa"] > 0:
                    efficiency = results["test_actual_kappa"] / results["theoretical_kappa"] * 100
                    logger.info(f"Client {client_id} - Efficiency: {efficiency:.1f}%")
                    mlflow.log_metric("kappa_efficiency", efficiency)
                
                logger.info(f"Client {client_id} completed: AUC={results['test_AUC']:.4f}, "
                           f"AUPRC={results['test_AUPRC']:.4f}")
                
            except Exception as e:
                logger.error(f"Error processing client {client_id}: {e}")
                raise
    
    save_results(all_results, METRICS_CSV, RESULTS_DIR, FORMATTED_CSV)
    logger.info("Pipeline completed successfully!")


if __name__ == "__main__":
    main()