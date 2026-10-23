# """Main training script with Optuna hyperparameter tuning, Neural Network."""
# import logging
# import numpy as np
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader, TensorDataset
# import optuna
# import mlflow
# from sklearn.metrics import f1_score, log_loss


# #custom imports
# from base_functions import (save_results, compute_all_metrics, 
#                             load_hyperparams, save_hyperparams, 
#                             check_or_compute_kappa, log_to_mlflow_local
#                             )

# from mlp_config import (
#     MLFLOW_EXPERIMENT, MLFLOW_SUFFIX,
#     METRICS_CSV, FORMATTED_CSV, 
#     NN_HPO_SPACE, HYPERPARAMS_FILE_PATTERN, MODEL
# )
# from general_config import (RANDOM_STATE, USE_EXISTING_HYPERPARAMS, THEORETICAL_KAPPA_FILE_PATTERN, 
#                             OPTUNA_TRIALS, RESULTS_DIR, DATA_BALANCE, HYPERPARAMS_DIR)
# from mydataloading import prepare_data, get_client_name

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
# optuna.logging.set_verbosity(optuna.logging.WARNING)

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# # =============================================================================
# # Neural Network Model
# # =============================================================================

# class MLP(nn.Module):
#     """MLP for binary classification."""
    
#     def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.2):
#         super().__init__()
        
#         hidden2 = max(32, hidden_dim // 2)
        
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.BatchNorm1d(hidden_dim),
#             nn.LeakyReLU(0.1),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, hidden2),
#             nn.BatchNorm1d(hidden2),
#             nn.LeakyReLU(0.1),
#             nn.Dropout(dropout * 0.5),
#             nn.Linear(hidden2, 1),
#         )
        
#         self._init_weights()
    
#     def _init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0.01)
#             elif isinstance(m, nn.BatchNorm1d):
#                 nn.init.ones_(m.weight)
#                 nn.init.zeros_(m.bias)
    
#     def forward(self, x):
#         return self.net(x).squeeze(-1)




# # =============================================================================
# # Training Function
# # =============================================================================

# def train_model(model: nn.Module, X_train: torch.Tensor, y_train: torch.Tensor,
#                 params: dict, pos_weight: torch.Tensor) -> nn.Module:
#     """Train a single model with given hyperparameters."""
#     # Extract and clamp hyperparameters
#     lr = max(1e-5, min(0.3, params.get("learning_rate", 0.001)))
#     weight_decay = max(1e-6, min(1e-2, params.get("weight_decay", 1e-4)))
#     local_epochs = int(max(10, min(200, params.get("local_epochs", 100))))
#     batch_size = int(max(8, min(256, params.get("batch_size", 32))))
#     beta1 = max(0.75, min(0.99, params.get("adam_beta1", 0.9)))
#     beta2 = max(0.89, min(0.9999, params.get("adam_beta2", 0.999)))
#     grad_clip = max(0.1, min(10.0, params.get("grad_clip_norm", 1.0)))
#     warmup_epochs = int(max(0, min(20, params.get("lr_warmup_epochs", 0))))
#     label_smoothing = max(0.0, min(0.7, params.get("label_smoothing", 0.0)))
    
#     model.train()
#     optimizer = torch.optim.AdamW(
#         model.parameters(), 
#         lr=lr, 
#         weight_decay=weight_decay,
#         betas=(beta1, beta2)
#     )
    
#     # Scheduler with optional warmup
#     if warmup_epochs > 0 and local_epochs > warmup_epochs:
#         warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
#             optimizer, start_factor=0.1, total_iters=warmup_epochs
#         )
#         cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#             optimizer, T_max=local_epochs - warmup_epochs
#         )
#         scheduler = torch.optim.lr_scheduler.SequentialLR(
#             optimizer, 
#             schedulers=[warmup_scheduler, cosine_scheduler],
#             milestones=[warmup_epochs]
#         )
#     else:
#         scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#             optimizer, T_max=local_epochs
#         )
    
#     # Criterion with label smoothing
#     if label_smoothing > 0:
#         criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')
#     else:
#         criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
#     dataset = TensorDataset(X_train, y_train)
#     loader = DataLoader(
#         dataset, 
#         batch_size=batch_size, 
#         shuffle=True, 
#         drop_last=len(dataset) > batch_size
#     )
    
#     for epoch in range(local_epochs):
#         for X_batch, y_batch in loader:
#             X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            
#             optimizer.zero_grad()
#             logits = model(X_batch)
            
#             # Base loss
#             if label_smoothing > 0:
#                 y_smooth = y_batch * (1 - label_smoothing) + 0.5 * label_smoothing
#                 loss = criterion(logits, y_smooth).mean()
#             else:
#                 loss = criterion(logits, y_batch)
            
#             if torch.isnan(loss) or torch.isinf(loss):
#                 continue
            
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(
#                 model.parameters(), 
#                 max_norm=grad_clip
#             )
#             optimizer.step()
        
#         scheduler.step()
    
#     return model


# def predict_proba(model: nn.Module, X: torch.Tensor) -> np.ndarray:
#     """Get probability predictions."""
#     model.eval()
#     with torch.no_grad():
#         logits = model(X.to(DEVICE)).cpu()
#         y_prob = torch.sigmoid(logits).numpy()
#     return np.clip(y_prob, 1e-7, 1 - 1e-7)


# # =============================================================================
# # Optuna Objective
# # =============================================================================

# def create_objective(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
#                      pos_weight: torch.Tensor, input_dim: int):
#     """Create Optuna objective function."""
#     # Convert to tensors
#     X_train_tensor = torch.FloatTensor(X_train)
#     y_train_tensor = torch.FloatTensor(y_train)
#     X_val_tensor = torch.FloatTensor(X_val)
    
#     def objective(trial):
#         # Sample hyperparameters from NN_HPO_SPACE
#         params = {}
        
#         for param_name, (low, high, param_type) in NN_HPO_SPACE.items():
#             if param_name == "threshold":
#                 # Handle threshold separately (not a training parameter)
#                 continue
            
#             if param_type == "log":
#                 params[param_name] = trial.suggest_float(param_name, low, high, log=True)
#             elif param_type == "int":
#                 params[param_name] = trial.suggest_int(param_name, low, high)
#             elif param_type == "float":
#                 params[param_name] = trial.suggest_float(param_name, low, high)
        
#         # Extract architecture params
#         hidden_dim = params.get("hidden_dim", 128)
#         dropout = params.get("dropout", 0.2)
        
#         # Create and train model
#         model = MLP(input_dim, hidden_dim, dropout).to(DEVICE)
#         model = train_model(model, X_train_tensor, y_train_tensor, params, pos_weight)
        
#         # Predict on validation fold
#         return log_loss(y_val, predict_proba(model, X_val_tensor))
    
#     return objective



# # =============================================================================
# # Training Pipeline
# # =============================================================================

# def train_client(client_id: int, compute_limits: bool = True) -> dict:
#     """Train model for a single client with Optuna tuning and 5-fold CV."""
#     client_name = get_client_name(client_id)
#     logger.info(f"Training client {client_id} ({client_name})")
    
#     # Load data (70/10/20 split)
#     X_train_df, y_train, X_test_df, y_test, X_val_df, y_val, feature_names = prepare_data(client_id)
    
#     # Convert to numpy
#     X_train_np = np.asarray(X_train_df, dtype=np.float32)
#     X_val_np = np.asarray(X_val_df, dtype=np.float32)
#     X_test_np = np.asarray(X_test_df, dtype=np.float32)
#     y_train_np = np.asarray(y_train, dtype=np.float32).ravel()
#     y_val_np = np.asarray(y_val, dtype=np.float32).ravel()
#     y_test_np = np.asarray(y_test, dtype=np.float32).ravel()
    
#     # Handle NaN/Inf
#     X_train_np = np.nan_to_num(X_train_np, nan=0.0, posinf=0.0, neginf=0.0)
#     X_val_np = np.nan_to_num(X_val_np, nan=0.0, posinf=0.0, neginf=0.0)
#     X_test_np = np.nan_to_num(X_test_np, nan=0.0, posinf=0.0, neginf=0.0)
    
#     # Convert to tensors
#     X_train = torch.FloatTensor(X_train_np)
#     X_val = torch.FloatTensor(X_val_np)
#     X_test = torch.FloatTensor(X_test_np)
#     y_train_tensor = torch.FloatTensor(y_train_np)
    
#     # Compute class weights
#     pos_count = y_train_np.sum()
#     neg_count = len(y_train_np) - pos_count
#     pos_weight_val = neg_count / max(pos_count, 1)
#     pos_weight_val = np.clip(pos_weight_val, 0.5, 10.0)
#     pos_weight = torch.tensor([pos_weight_val]).to(DEVICE)
    
#     logger.info(f"Client {client_id}: {neg_count:.0f} neg, {pos_count:.0f} pos, weight={pos_weight_val:.2f}")
    
#     input_dim = X_train.shape[1]
    
#     # Compute theoretical limits (or load from file)
#     theoretical_kappa = check_or_compute_kappa(compute_limits, client_name, THEORETICAL_KAPPA_FILE_PATTERN, DATA_BALANCE, X_test_df, X_train_df, X_val_df, y_test_np, y_train_np, y_val_np, HYPERPARAMS_DIR)
   
#     # Get hyperparameters: load existing or run Optuna
#     existing_params = load_hyperparams(client_name, HYPERPARAMS_FILE_PATTERN, DATA_BALANCE) if USE_EXISTING_HYPERPARAMS else None
    
#     if existing_params:
#         best_params = existing_params
#         optuna_best_score = None
#         logger.info("Using existing hyperparameters")
#     else:
#         logger.info(f"Running Optuna optimization ({OPTUNA_TRIALS} trials)...")
#         study = optuna.create_study(
#             direction="minimize", 
#             sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
#         )
#         study.optimize(
#             create_objective(X_train_np, y_train_np, X_val_np, y_val_np, pos_weight, input_dim), 
#             n_trials=OPTUNA_TRIALS, 
#             show_progress_bar=True
#         )
        
#         best_params = study.best_params
#         optuna_best_score = study.best_value
#         logger.info(f"Best score: {optuna_best_score:.4f}")
        
#         # Save hyperparameters
#         save_hyperparams(client_name, study.best_params, HYPERPARAMS_FILE_PATTERN, DATA_BALANCE)
    
#     # Extract architecture params
#     hidden_dim = best_params.get("hidden_dim", 128)
#     dropout = best_params.get("dropout", 0.2)
    
#     # Train final model on full training set
#     logger.info("Training final model on full training set...")
#     model = MLP(input_dim, hidden_dim, dropout).to(DEVICE)
#     model = train_model(model, X_train, y_train_tensor, best_params, pos_weight)
    
#     val_probs = predict_proba(model, X_val)
#     grid = np.linspace(0.05, 0.95, 91)
#     threshold = float(grid[np.argmax([f1_score(y_val_np, (val_probs > t).astype(int), zero_division=0) for t in grid])])
#     best_params["threshold"] = threshold
    
#     # Final test set evaluation (only once on held-out test set)
#     test_probs = predict_proba(model, X_test)
#     test_metrics = compute_all_metrics(y_test_np, test_probs, threshold)
    
#     # Save model
#     model_dir = RESULTS_DIR / "models"
#     model_dir.mkdir(exist_ok=True)
#     model_path = model_dir / f"nn_model_client_{client_id}_{DATA_BALANCE}.pt"
#     torch.save({
#         'model_state_dict': model.state_dict(),
#         'input_dim': input_dim,
#         'hidden_dim': hidden_dim,
#         'dropout': dropout,
#         'threshold': threshold,
#         'hyperparameters': best_params
#     }, model_path)
#     logger.info(f"Saved model to {model_path}")
    
#     # Compile results
#     results = {
#         "client_id": client_id,
#         "client_name": client_name,
#         "theoretical_kappa": theoretical_kappa,
#         "optuna_best_score": optuna_best_score,
#         **{f"test_{k}": v for k, v in test_metrics.items()},
#         **{f"param_{k}": v for k, v in best_params.items()},
#     }
    
#     return results, model, test_probs


# # =============================================================================
# # Main Entry Point
# # =============================================================================

# def main():
#     """Main execution function."""
#     logger.info("Starting ML pipeline with Neural Network, Optuna tuning and 5-fold CV")
    
#     mlflow.set_experiment(MLFLOW_EXPERIMENT)
#     all_results = []
    
#     for client_id in range(1, 6):
#         run_name = f"client_nn_{client_id}_{get_client_name(client_id)}_{MLFLOW_SUFFIX}"
        
#         with mlflow.start_run(run_name=run_name):
#             try:
#                 results, model, probs = train_client(client_id, compute_limits=True)
#                 log_to_mlflow_local(results, MODEL, OPTUNA_TRIALS, DATA_BALANCE)
#                 all_results.append(results)
                
#                 # Log efficiency
#                 if results["theoretical_kappa"] > 0:
#                     efficiency = results["test_actual_kappa"] / results["theoretical_kappa"] * 100
#                     logger.info(f"Client {client_id} - Efficiency: {efficiency:.1f}%")
#                     mlflow.log_metric("kappa_efficiency", efficiency)
                
#                 logger.info(f"Client {client_id} completed: AUC={results['test_AUC']:.4f}, "
#                            f"AUPRC={results['test_AUPRC']:.4f}")
                
#             except Exception as e:
#                 logger.error(f"Error processing client {client_id}: {e}")
#                 raise
    
#     save_results(all_results, METRICS_CSV, RESULTS_DIR, FORMATTED_CSV)
#     logger.info("Pipeline completed successfully!")


# if __name__ == "__main__":
#     main()

"""Main training script with Optuna hyperparameter tuning, Neural Network."""
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import optuna
import mlflow
from sklearn.metrics import f1_score, log_loss, roc_auc_score, average_precision_score


#custom imports
from base_functions import (save_results, compute_all_metrics, 
                            load_hyperparams, save_hyperparams, 
                            check_or_compute_kappa, log_to_mlflow_local
                            )

from mlp_config import (
    MLFLOW_EXPERIMENT, MLFLOW_SUFFIX,
    METRICS_CSV, FORMATTED_CSV, 
    NN_HPO_SPACE, HYPERPARAMS_FILE_PATTERN, MODEL
)
from general_config import (RANDOM_STATE, USE_EXISTING_HYPERPARAMS, THEORETICAL_KAPPA_FILE_PATTERN, 
                            OPTUNA_TRIALS, RESULTS_DIR, DATA_BALANCE, HYPERPARAMS_DIR)
from mydataloading import prepare_data, get_client_name

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Early stopping on validation log-loss (epochs without improvement)
EARLY_STOPPING_PATIENCE = 10


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
# Neural Network Model
# =============================================================================

class MLP(nn.Module):
    """MLP for binary classification."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        
        hidden2 = max(32, hidden_dim // 2)
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden2, 1),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.net(x).squeeze(-1)




# =============================================================================
# Training Function
# =============================================================================

def train_model(model: nn.Module, X_train: torch.Tensor, y_train: torch.Tensor,
                params: dict, pos_weight: torch.Tensor,
                X_val: torch.Tensor = None, y_val: np.ndarray = None) -> nn.Module:
    """Train a single model with given hyperparameters."""
    # Extract and clamp hyperparameters
    lr = max(1e-5, min(0.3, params.get("learning_rate", 0.001)))
    weight_decay = max(1e-6, min(1e-2, params.get("weight_decay", 1e-4)))
    local_epochs = int(max(10, min(200, params.get("local_epochs", 100))))
    batch_size = int(max(8, min(256, params.get("batch_size", 32))))
    beta1 = max(0.75, min(0.99, params.get("adam_beta1", 0.9)))
    beta2 = max(0.89, min(0.9999, params.get("adam_beta2", 0.999)))
    grad_clip = max(0.1, min(10.0, params.get("grad_clip_norm", 1.0)))
    warmup_epochs = int(max(0, min(20, params.get("lr_warmup_epochs", 0))))
    label_smoothing = max(0.0, min(0.7, params.get("label_smoothing", 0.0)))
    
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=lr, 
        weight_decay=weight_decay,
        betas=(beta1, beta2)
    )
    
    # Scheduler with optional warmup
    if warmup_epochs > 0 and local_epochs > warmup_epochs:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=warmup_epochs
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=local_epochs - warmup_epochs
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, 
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs]
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=local_epochs
        )
    
    # Criterion with label smoothing
    if label_smoothing > 0:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=len(dataset) > batch_size
    )
    
    best_loss, best_epoch, best_state, wait = np.inf, 0, None, 0
    
    for epoch in range(local_epochs):
        model.train()
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            
            optimizer.zero_grad()
            logits = model(X_batch)
            
            # Base loss
            if label_smoothing > 0:
                y_smooth = y_batch * (1 - label_smoothing) + 0.5 * label_smoothing
                loss = criterion(logits, y_smooth).mean()
            else:
                loss = criterion(logits, y_batch)
            
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                max_norm=grad_clip
            )
            optimizer.step()
        
        scheduler.step()
        
        # Early stopping on validation log-loss, keep best weights
        if y_val is not None:
            val_loss = log_loss(y_val, predict_proba(model, X_val))
            if val_loss < best_loss:
                best_loss, best_epoch, wait = val_loss, epoch + 1, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                wait += 1
                if wait >= EARLY_STOPPING_PATIENCE:
                    break
    
    if best_state is not None:
        model.load_state_dict(best_state)
    model.best_epoch = best_epoch if y_val is not None else local_epochs
    
    return model


def predict_proba(model: nn.Module, X: torch.Tensor) -> np.ndarray:
    """Get probability predictions."""
    model.eval()
    with torch.no_grad():
        logits = model(X.to(DEVICE)).cpu()
        y_prob = torch.sigmoid(logits).numpy()
    return np.clip(y_prob, 1e-7, 1 - 1e-7)


# =============================================================================
# Optuna Objective
# =============================================================================

def create_objective(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
                     pos_weight: torch.Tensor, input_dim: int):
    """Create Optuna objective function."""
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    
    def objective(trial):
        # Sample hyperparameters from NN_HPO_SPACE
        params = {}
        
        for param_name, (low, high, param_type) in NN_HPO_SPACE.items():
            if param_name == "threshold":
                # Handle threshold separately (not a training parameter)
                continue
            
            if param_type == "log":
                params[param_name] = trial.suggest_float(param_name, low, high, log=True)
            elif param_type == "int":
                params[param_name] = trial.suggest_int(param_name, low, high)
            elif param_type == "float":
                params[param_name] = trial.suggest_float(param_name, low, high)
        
        # Extract architecture params
        hidden_dim = params.get("hidden_dim", 128)
        dropout = params.get("dropout", 0.2)
        
        # Create and train model
        model = MLP(input_dim, hidden_dim, dropout).to(DEVICE)
        model = train_model(model, X_train_tensor, y_train_tensor, params, pos_weight, X_val_tensor, y_val)
        
        # Predict on validation fold
        val_probs = predict_proba(model, X_val_tensor)
        train_probs = predict_proba(model, X_train_tensor)
        val_logloss = log_loss(y_val, val_probs)
        
        # Log this trial to the active client run; step = trial number
        trial_metrics = {
            "trial_val_logloss": val_logloss,
            "trial_val_AUC": roc_auc_score(y_val, val_probs),
            "trial_val_AUPRC": average_precision_score(y_val, val_probs),
            "trial_train_logloss": log_loss(y_train, train_probs),
            "trial_train_AUC": roc_auc_score(y_train, train_probs),
            "trial_train_AUPRC": average_precision_score(y_train, train_probs),
            "trial_best_epoch": float(model.best_epoch),
            **{f"trial_param_{k}": float(v) for k, v in trial.params.items()},
        }
        mlflow.log_metrics(trial_metrics, step=trial.number)
        
        return val_logloss
    
    return objective



# =============================================================================
# Training Pipeline
# =============================================================================

def train_client(client_id: int, compute_limits: bool = True) -> dict:
    """Train model for a single client with Optuna tuning and 5-fold CV."""
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
    
    # Convert to tensors
    X_train = torch.FloatTensor(X_train_np)
    X_val = torch.FloatTensor(X_val_np)
    X_test = torch.FloatTensor(X_test_np)
    y_train_tensor = torch.FloatTensor(y_train_np)
    
    # Compute class weights
    pos_count = y_train_np.sum()
    neg_count = len(y_train_np) - pos_count
    pos_weight_val = neg_count / max(pos_count, 1)
    pos_weight_val = np.clip(pos_weight_val, 0.5, 10.0)
    pos_weight = torch.tensor([pos_weight_val]).to(DEVICE)
    
    logger.info(f"Client {client_id}: {neg_count:.0f} neg, {pos_count:.0f} pos, weight={pos_weight_val:.2f}")
    
    input_dim = X_train.shape[1]
    
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
            create_objective(X_train_np, y_train_np, X_val_np, y_val_np, pos_weight, input_dim), 
            n_trials=OPTUNA_TRIALS, 
            show_progress_bar=True
        )
        
        best_params = study.best_params
        optuna_best_score = study.best_value
        logger.info(f"Best score: {optuna_best_score:.4f}")
        mlflow.log_metric("optuna_best_trial", float(study.best_trial.number))
        
        # Save hyperparameters
        save_hyperparams(client_name, study.best_params, HYPERPARAMS_FILE_PATTERN, DATA_BALANCE)
    
    # Extract architecture params
    hidden_dim = best_params.get("hidden_dim", 128)
    dropout = best_params.get("dropout", 0.2)
    
    # Train final model on full training set
    logger.info("Training final model on full training set...")
    model = MLP(input_dim, hidden_dim, dropout).to(DEVICE)
    model = train_model(model, X_train, y_train_tensor, best_params, pos_weight, X_val, y_val_np)
    logger.info(f"Early stopping: best epoch {model.best_epoch}")
    
    val_probs = predict_proba(model, X_val)
    grid = np.linspace(0.05, 0.95, 91)
    threshold = float(grid[np.argmax([f1_score(y_val_np, (val_probs > t).astype(int), zero_division=0) for t in grid])])
    best_params["threshold"] = threshold
    
    # Train and validation metrics with the same threshold (diagnostics only)
    train_probs = predict_proba(model, X_train)
    train_metrics = compute_all_metrics(y_train_np, train_probs, threshold, n_boot=0)
    val_metrics = compute_all_metrics(y_val_np, val_probs, threshold, n_boot=0)
    
    # Final test set evaluation (only once on held-out test set)
    test_probs = predict_proba(model, X_test)
    test_metrics = compute_all_metrics(y_test_np, test_probs, threshold)
    
    # Save model
    model_dir = RESULTS_DIR / "models"
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / f"nn_model_client_{client_id}_{DATA_BALANCE}.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': input_dim,
        'hidden_dim': hidden_dim,
        'dropout': dropout,
        'threshold': threshold,
        'hyperparameters': best_params
    }, model_path)
    logger.info(f"Saved model to {model_path}")
    
    # Compile results
    results = {
        "client_id": client_id,
        "client_name": client_name,
        "theoretical_kappa": theoretical_kappa,
        "optuna_best_score": optuna_best_score,
        "best_epoch": model.best_epoch,
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
    logger.info("Starting ML pipeline with Neural Network, Optuna tuning and 5-fold CV")
    
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    all_results = []
    
    for client_id in range(1, 6):
        run_name = f"client_nn_{client_id}_{get_client_name(client_id)}_{MLFLOW_SUFFIX}"
        
        with mlflow.start_run(run_name=run_name):
            try:
                results, model, probs = train_client(client_id, compute_limits=True)
                log_to_mlflow_local(results, MODEL, OPTUNA_TRIALS, DATA_BALANCE)
                log_split_metrics(results)
                mlflow.log_metric("best_epoch", float(results["best_epoch"]))
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