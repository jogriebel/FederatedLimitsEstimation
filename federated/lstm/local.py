# """Main training script with Optuna hyperparameter tuning, LSTM."""
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

# from lstm_config import (
#     MLFLOW_EXPERIMENT, MLFLOW_SUFFIX,
#     METRICS_CSV, FORMATTED_CSV, 
#     LSTM_HPO_SPACE, HYPERPARAMS_FILE_PATTERN, MODEL
# )
# from general_config import (RANDOM_STATE, USE_EXISTING_HYPERPARAMS, THEORETICAL_KAPPA_FILE_PATTERN, 
#                             OPTUNA_TRIALS, RESULTS_DIR, DATA_BALANCE, HYPERPARAMS_DIR)
# from mydataloading import prepare_data, get_client_name

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
# optuna.logging.set_verbosity(optuna.logging.WARNING)

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # =============================================================================
# # Temporal Feature Detection and Reshaping
# # =============================================================================

# def detect_temporal_features(feature_names):
#     """Detect temporal features ending in _1, _2, _3, _4 and group them."""
#     temporal_groups = {}
#     static_features = []
    
#     for feat in feature_names:
#         if feat.endswith(('_1', '_2', '_3', '_4')):
#             # Extract base name and time index
#             base_name = feat.rsplit('_', 1)[0]
#             time_idx = int(feat.rsplit('_', 1)[1]) - 1  # Convert to 0-indexed
            
#             if base_name not in temporal_groups:
#                 temporal_groups[base_name] = [None] * 4
#             temporal_groups[base_name][time_idx] = feat
#         else:
#             static_features.append(feat)
    
#     # Filter complete sequences (all 4 time points present)
#     temporal_groups = {k: v for k, v in temporal_groups.items() if None not in v}
    
#     return temporal_groups, static_features


# def reshape_to_sequences(X_df, feature_names):
#     """Reshape wide-format temporal data to (samples, seq_len, features)."""
#     temporal_groups, static_features = detect_temporal_features(feature_names)
    
#     # Extract temporal data
#     temporal_data = []
#     for base_name in sorted(temporal_groups.keys()):
#         time_series_cols = temporal_groups[base_name]
#         time_series = X_df[time_series_cols].values  # (samples, 4)
#         temporal_data.append(time_series)
    
#     # Stack: (samples, n_temporal_features, 4) -> (samples, 4, n_temporal_features)
#     if temporal_data:
#         temporal_array = np.stack(temporal_data, axis=1)
#         temporal_array = np.transpose(temporal_array, (0, 2, 1))
#     else:
#         temporal_array = None
    
#     # Extract static data
#     static_array = X_df[static_features].values if static_features else None
    
#     logger.info(f"Detected {len(temporal_groups)} temporal features (seq_len=4), "
#                 f"{len(static_features)} static features")
    
#     return temporal_array, static_array, len(temporal_groups), len(static_features)


# # =============================================================================
# # LSTM Model
# # =============================================================================

# class LSTMClassifier(nn.Module):
#     """LSTM for binary classification with temporal and static features."""
    
#     def __init__(self, temporal_dim, static_dim, hidden_dim=64, num_layers=2, dropout=0.2):
#         super().__init__()
#         self.has_temporal = temporal_dim > 0
#         self.has_static = static_dim > 0
        
#         if self.has_temporal:
#             self.lstm = nn.LSTM(
#                 temporal_dim, 
#                 hidden_dim,
#                 num_layers=num_layers,
#                 dropout=dropout if num_layers > 1 else 0,
#                 batch_first=True
#             )
#             combined_dim = hidden_dim
#         else:
#             combined_dim = 0
        
#         if self.has_static:
#             combined_dim += static_dim
        
#         self.fc = nn.Sequential(
#             nn.Linear(combined_dim, max(32, combined_dim // 2)),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(max(32, combined_dim // 2), 1)
#         )
        
#         self._init_weights()
    
#     def _init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0.0)
    
#     def forward(self, temporal=None, static=None):
#         features = []
        
#         if self.has_temporal and temporal is not None:
#             lstm_out, (hidden, _) = self.lstm(temporal)
#             features.append(hidden[-1])  # Use last layer's hidden state
        
#         if self.has_static and static is not None:
#             features.append(static)
        
#         combined = torch.cat(features, dim=1)
#         return self.fc(combined).squeeze(-1)


# # =============================================================================
# # Training Function
# # =============================================================================

# def train_model(model, temporal_train, static_train, y_train, params, pos_weight):
#     """Train a single model with given hyperparameters."""
#     lr = np.clip(params.get("learning_rate", 0.001), 1e-5, 0.1)
#     weight_decay = np.clip(params.get("weight_decay", 1e-4), 1e-6, 1e-2)
#     local_epochs = int(np.clip(params.get("local_epochs", 50), 10, 200))
#     batch_size = int(np.clip(params.get("batch_size", 32), 8, 256))
    
#     model.train()
#     optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=local_epochs)
#     criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
#     # Create dataset based on what features we have
#     tensors = []
#     if temporal_train is not None:
#         tensors.append(temporal_train)
#     if static_train is not None:
#         tensors.append(static_train)
#     tensors.append(y_train)
    
#     dataset = TensorDataset(*tensors)
#     loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
#                        drop_last=len(dataset) > batch_size)
    
#     for epoch in range(local_epochs):
#         for batch in loader:
#             y_batch = batch[-1].to(DEVICE)
#             temporal_batch = batch[0].to(DEVICE) if temporal_train is not None else None
#             static_batch = batch[1].to(DEVICE) if len(batch) == 3 and static_train is not None else None
            
#             optimizer.zero_grad()
#             logits = model(temporal=temporal_batch, static=static_batch)
#             loss = criterion(logits, y_batch)
            
#             if not (torch.isnan(loss) or torch.isinf(loss)):
#                 loss.backward()
#                 optimizer.step()
        
#         scheduler.step()
    
#     return model


# def predict_proba(model, temporal, static):
#     """Get probability predictions."""
#     model.eval()
#     with torch.no_grad():
#         temporal_dev = temporal.to(DEVICE) if temporal is not None else None
#         static_dev = static.to(DEVICE) if static is not None else None
#         logits = model(temporal=temporal_dev, static=static_dev).cpu()
#         y_prob = torch.sigmoid(logits).numpy()
#     return np.clip(y_prob, 1e-7, 1 - 1e-7)


# # =============================================================================
# # Optuna Objective
# # =============================================================================

# def create_objective(temporal_np, static_np, y, temporal_val_np, static_val_np, y_val,
#                      pos_weight, temporal_dim, static_dim):
#     """Create Optuna objective function."""
#     temporal_tensor = torch.FloatTensor(temporal_np) if temporal_np is not None else None
#     static_tensor = torch.FloatTensor(static_np) if static_np is not None else None
#     temporal_val_tensor = torch.FloatTensor(temporal_val_np) if temporal_val_np is not None else None
#     static_val_tensor = torch.FloatTensor(static_val_np) if static_val_np is not None else None
#     y_tensor = torch.FloatTensor(y)
    
#     def objective(trial):
#         params = {}
#         for param_name, (low, high, param_type) in LSTM_HPO_SPACE.items():
#             if param_type == "log":
#                 params[param_name] = trial.suggest_float(param_name, low, high, log=True)
#             elif param_type == "int":
#                 params[param_name] = trial.suggest_int(param_name, low, high)
#             elif param_type == "float":
#                 params[param_name] = trial.suggest_float(param_name, low, high)
        
#         hidden_dim = params.get("hidden_dim", 64)
#         num_layers = params.get("num_layers", 2)
#         dropout = params.get("dropout", 0.2)
        
#         model = LSTMClassifier(temporal_dim, static_dim, hidden_dim, num_layers, dropout).to(DEVICE)
#         model = train_model(model, temporal_tensor, static_tensor, y_tensor, params, pos_weight)
        
#         return log_loss(y_val, predict_proba(model, temporal_val_tensor, static_val_tensor))
    
#     return objective


# # =============================================================================
# # Training Pipeline
# # =============================================================================

# def train_client(client_id, compute_limits=True):
#     """Train model for a single client with Optuna tuning and 5-fold CV."""
#     client_name = get_client_name(client_id)
#     logger.info(f"Training client {client_id} ({client_name})")
    
#     X_train_df, y_train, X_test_df, y_test, X_val_df, y_val, feature_names = prepare_data(client_id)
    
#     # Reshape to sequences
#     temporal_train, static_train, temporal_dim, static_dim = reshape_to_sequences(X_train_df, feature_names)
#     temporal_val, static_val, _, _ = reshape_to_sequences(X_val_df, feature_names)
#     temporal_test, static_test, _, _ = reshape_to_sequences(X_test_df, feature_names)
    
#     # Handle NaN/Inf
#     if temporal_train is not None:
#         temporal_train = np.nan_to_num(temporal_train, nan=0.0, posinf=0.0, neginf=0.0)
#         temporal_val = np.nan_to_num(temporal_val, nan=0.0, posinf=0.0, neginf=0.0)
#         temporal_test = np.nan_to_num(temporal_test, nan=0.0, posinf=0.0, neginf=0.0)
#     if static_train is not None:
#         static_train = np.nan_to_num(static_train, nan=0.0, posinf=0.0, neginf=0.0)
#         static_val = np.nan_to_num(static_val, nan=0.0, posinf=0.0, neginf=0.0)
#         static_test = np.nan_to_num(static_test, nan=0.0, posinf=0.0, neginf=0.0)
    
#     y_train_np = np.asarray(y_train, dtype=np.float32).ravel()
#     y_val_np = np.asarray(y_val, dtype=np.float32).ravel()
#     y_test_np = np.asarray(y_test, dtype=np.float32).ravel()
    
#     # Convert to tensors
#     temporal_train_tensor = torch.FloatTensor(temporal_train) if temporal_train is not None else None
#     static_train_tensor = torch.FloatTensor(static_train) if static_train is not None else None
#     temporal_val_tensor = torch.FloatTensor(temporal_val) if temporal_val is not None else None
#     static_val_tensor = torch.FloatTensor(static_val) if static_val is not None else None
#     temporal_test_tensor = torch.FloatTensor(temporal_test) if temporal_test is not None else None
#     static_test_tensor = torch.FloatTensor(static_test) if static_test is not None else None
#     y_train_tensor = torch.FloatTensor(y_train_np)
    
#     # Compute class weights
#     pos_count = y_train_np.sum()
#     neg_count = len(y_train_np) - pos_count
#     pos_weight_val = np.clip(neg_count / max(pos_count, 1), 0.5, 10.0)
#     pos_weight = torch.tensor([pos_weight_val]).to(DEVICE)
    
#     logger.info(f"Client {client_id}: {neg_count:.0f} neg, {pos_count:.0f} pos, weight={pos_weight_val:.2f}")
    
#     theoretical_kappa = check_or_compute_kappa(compute_limits, client_name, THEORETICAL_KAPPA_FILE_PATTERN, DATA_BALANCE, X_test_df, X_train_df, X_val_df, y_test_np, y_train_np, y_val_np, HYPERPARAMS_DIR)
    
#     # Handle hyperparameters
#     existing_params = load_hyperparams(client_name, HYPERPARAMS_FILE_PATTERN, DATA_BALANCE) if USE_EXISTING_HYPERPARAMS else None
    
#     if existing_params:
#         best_params = existing_params
#         optuna_best_score = None
#         logger.info("Using existing hyperparameters")
#     else:
#         logger.info(f"Running Optuna optimization ({OPTUNA_TRIALS} trials)...")
#         study = optuna.create_study(direction="minimize", 
#                                     sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
#         study.optimize(
#             create_objective(temporal_train, static_train, y_train_np, temporal_val, static_val, y_val_np,
#                            pos_weight, temporal_dim, static_dim), 
#             n_trials=OPTUNA_TRIALS, 
#             show_progress_bar=True
#         )
#         best_params = study.best_params
#         optuna_best_score = study.best_value
#         logger.info(f"Best score: {optuna_best_score:.4f}")
#         save_hyperparams(client_name, best_params, HYPERPARAMS_FILE_PATTERN, DATA_BALANCE)
    
#     # Train final model
#     logger.info("Training final model on full training set...")
#     hidden_dim = best_params.get("hidden_dim", 64)
#     num_layers = best_params.get("num_layers", 2)
#     dropout = best_params.get("dropout", 0.2)
#     model = LSTMClassifier(temporal_dim, static_dim, hidden_dim, num_layers, dropout).to(DEVICE)
#     model = train_model(model, temporal_train_tensor, static_train_tensor, y_train_tensor, 
#                        best_params, pos_weight)
    
#     val_probs = predict_proba(model, temporal_val_tensor, static_val_tensor)
#     grid = np.linspace(0.05, 0.95, 91)
#     threshold = float(grid[np.argmax([f1_score(y_val_np, (val_probs > t).astype(int), zero_division=0) for t in grid])])
#     best_params["threshold"] = threshold
    
#     # Test set evaluation
#     test_probs = predict_proba(model, temporal_test_tensor, static_test_tensor)
#     test_metrics = compute_all_metrics(y_test_np, test_probs, threshold)

#     # Save model
#     model_dir = RESULTS_DIR / "models"
#     model_dir.mkdir(exist_ok=True)
#     model_path = model_dir / f"lstm_model_client_{client_id}_{DATA_BALANCE}.pt"
#     torch.save({
#         'model_state_dict': model.state_dict(),
#         'temporal_dim': temporal_dim,
#         'static_dim': static_dim,
#         'hidden_dim': hidden_dim,
#         'num_layers': num_layers,
#         'dropout': dropout,
#         'threshold': threshold,
#         'hyperparameters': best_params
#     }, model_path)
#     logger.info(f"Saved model to {model_path}")
    
#     results = {
#         "client_id": client_id,
#         "client_name": client_name,
#         "temporal_features": temporal_dim,
#         "static_features": static_dim,
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
#     logger.info("Starting ML pipeline with LSTM, Optuna tuning and 5-fold CV")
    
#     mlflow.set_experiment(MLFLOW_EXPERIMENT)
#     all_results = []
    
#     for client_id in range(1, 6):
#         run_name = f"client_lstm_{client_id}_{get_client_name(client_id)}_{MLFLOW_SUFFIX}"
        
#         with mlflow.start_run(run_name=run_name):
#             try:
#                 results, model, probs = train_client(client_id, compute_limits=True)
#                 log_to_mlflow_local(results, MODEL, OPTUNA_TRIALS, DATA_BALANCE)
#                 all_results.append(results)
                
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

"""Main training script with Optuna hyperparameter tuning, LSTM."""
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

from lstm_config import (
    MLFLOW_EXPERIMENT, MLFLOW_SUFFIX,
    METRICS_CSV, FORMATTED_CSV, 
    LSTM_HPO_SPACE, HYPERPARAMS_FILE_PATTERN, MODEL
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
# Temporal Feature Detection and Reshaping
# =============================================================================

def detect_temporal_features(feature_names):
    """Detect temporal features ending in _1, _2, _3, _4 and group them."""
    temporal_groups = {}
    static_features = []
    
    for feat in feature_names:
        if feat.endswith(('_1', '_2', '_3', '_4')):
            # Extract base name and time index
            base_name = feat.rsplit('_', 1)[0]
            time_idx = int(feat.rsplit('_', 1)[1]) - 1  # Convert to 0-indexed
            
            if base_name not in temporal_groups:
                temporal_groups[base_name] = [None] * 4
            temporal_groups[base_name][time_idx] = feat
        else:
            static_features.append(feat)
    
    # Filter complete sequences (all 4 time points present)
    temporal_groups = {k: v for k, v in temporal_groups.items() if None not in v}
    
    return temporal_groups, static_features


def reshape_to_sequences(X_df, feature_names):
    """Reshape wide-format temporal data to (samples, seq_len, features)."""
    temporal_groups, static_features = detect_temporal_features(feature_names)
    
    # Extract temporal data
    temporal_data = []
    for base_name in sorted(temporal_groups.keys()):
        time_series_cols = temporal_groups[base_name]
        time_series = X_df[time_series_cols].values  # (samples, 4)
        temporal_data.append(time_series)
    
    # Stack: (samples, n_temporal_features, 4) -> (samples, 4, n_temporal_features)
    if temporal_data:
        temporal_array = np.stack(temporal_data, axis=1)
        temporal_array = np.transpose(temporal_array, (0, 2, 1))
    else:
        temporal_array = None
    
    # Extract static data
    static_array = X_df[static_features].values if static_features else None
    
    logger.info(f"Detected {len(temporal_groups)} temporal features (seq_len=4), "
                f"{len(static_features)} static features")
    
    return temporal_array, static_array, len(temporal_groups), len(static_features)


# =============================================================================
# LSTM Model
# =============================================================================

class LSTMClassifier(nn.Module):
    """LSTM for binary classification with temporal and static features."""
    
    def __init__(self, temporal_dim, static_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.has_temporal = temporal_dim > 0
        self.has_static = static_dim > 0
        
        if self.has_temporal:
            self.lstm = nn.LSTM(
                temporal_dim, 
                hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
            combined_dim = hidden_dim
        else:
            combined_dim = 0
        
        if self.has_static:
            combined_dim += static_dim
        
        self.fc = nn.Sequential(
            nn.Linear(combined_dim, max(32, combined_dim // 2)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(32, combined_dim // 2), 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
    
    def forward(self, temporal=None, static=None):
        features = []
        
        if self.has_temporal and temporal is not None:
            lstm_out, (hidden, _) = self.lstm(temporal)
            features.append(hidden[-1])  # Use last layer's hidden state
        
        if self.has_static and static is not None:
            features.append(static)
        
        combined = torch.cat(features, dim=1)
        return self.fc(combined).squeeze(-1)


# =============================================================================
# Training Function
# =============================================================================

def train_model(model, temporal_train, static_train, y_train, params, pos_weight,
                temporal_val=None, static_val=None, y_val=None):
    """Train a single model with given hyperparameters."""
    lr = np.clip(params.get("learning_rate", 0.001), 1e-5, 0.1)
    weight_decay = np.clip(params.get("weight_decay", 1e-4), 1e-6, 1e-2)
    local_epochs = int(np.clip(params.get("local_epochs", 50), 10, 200))
    batch_size = int(np.clip(params.get("batch_size", 32), 8, 256))
    
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=local_epochs)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Create dataset based on what features we have
    tensors = []
    if temporal_train is not None:
        tensors.append(temporal_train)
    if static_train is not None:
        tensors.append(static_train)
    tensors.append(y_train)
    
    dataset = TensorDataset(*tensors)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                       drop_last=len(dataset) > batch_size)
    
    best_loss, best_epoch, best_state, wait = np.inf, 0, None, 0
    
    for epoch in range(local_epochs):
        model.train()
        for batch in loader:
            y_batch = batch[-1].to(DEVICE)
            temporal_batch = batch[0].to(DEVICE) if temporal_train is not None else None
            static_batch = batch[1].to(DEVICE) if len(batch) == 3 and static_train is not None else None
            
            optimizer.zero_grad()
            logits = model(temporal=temporal_batch, static=static_batch)
            loss = criterion(logits, y_batch)
            
            if not (torch.isnan(loss) or torch.isinf(loss)):
                loss.backward()
                optimizer.step()
        
        scheduler.step()
        
        # Early stopping on validation log-loss, keep best weights
        if y_val is not None:
            val_loss = log_loss(y_val, predict_proba(model, temporal_val, static_val))
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


def predict_proba(model, temporal, static):
    """Get probability predictions."""
    model.eval()
    with torch.no_grad():
        temporal_dev = temporal.to(DEVICE) if temporal is not None else None
        static_dev = static.to(DEVICE) if static is not None else None
        logits = model(temporal=temporal_dev, static=static_dev).cpu()
        y_prob = torch.sigmoid(logits).numpy()
    return np.clip(y_prob, 1e-7, 1 - 1e-7)


# =============================================================================
# Optuna Objective
# =============================================================================

def create_objective(temporal_np, static_np, y, temporal_val_np, static_val_np, y_val,
                     pos_weight, temporal_dim, static_dim):
    """Create Optuna objective function."""
    temporal_tensor = torch.FloatTensor(temporal_np) if temporal_np is not None else None
    static_tensor = torch.FloatTensor(static_np) if static_np is not None else None
    temporal_val_tensor = torch.FloatTensor(temporal_val_np) if temporal_val_np is not None else None
    static_val_tensor = torch.FloatTensor(static_val_np) if static_val_np is not None else None
    y_tensor = torch.FloatTensor(y)
    
    def objective(trial):
        params = {}
        for param_name, (low, high, param_type) in LSTM_HPO_SPACE.items():
            if param_type == "log":
                params[param_name] = trial.suggest_float(param_name, low, high, log=True)
            elif param_type == "int":
                params[param_name] = trial.suggest_int(param_name, low, high)
            elif param_type == "float":
                params[param_name] = trial.suggest_float(param_name, low, high)
        
        hidden_dim = params.get("hidden_dim", 64)
        num_layers = params.get("num_layers", 2)
        dropout = params.get("dropout", 0.2)
        
        model = LSTMClassifier(temporal_dim, static_dim, hidden_dim, num_layers, dropout).to(DEVICE)
        model = train_model(model, temporal_tensor, static_tensor, y_tensor, params, pos_weight,
                            temporal_val_tensor, static_val_tensor, y_val)
        
        val_probs = predict_proba(model, temporal_val_tensor, static_val_tensor)
        train_probs = predict_proba(model, temporal_tensor, static_tensor)
        val_logloss = log_loss(y_val, val_probs)
        
        # Log this trial to the active client run; step = trial number
        trial_metrics = {
            "trial_val_logloss": val_logloss,
            "trial_val_AUC": roc_auc_score(y_val, val_probs),
            "trial_val_AUPRC": average_precision_score(y_val, val_probs),
            "trial_train_logloss": log_loss(y, train_probs),
            "trial_train_AUC": roc_auc_score(y, train_probs),
            "trial_train_AUPRC": average_precision_score(y, train_probs),
            "trial_best_epoch": float(model.best_epoch),
            **{f"trial_param_{k}": float(v) for k, v in trial.params.items()},
        }
        mlflow.log_metrics(trial_metrics, step=trial.number)
        
        return val_logloss
    
    return objective


# =============================================================================
# Training Pipeline
# =============================================================================

def train_client(client_id, compute_limits=True):
    """Train model for a single client with Optuna tuning and 5-fold CV."""
    client_name = get_client_name(client_id)
    logger.info(f"Training client {client_id} ({client_name})")
    
    X_train_df, y_train, X_test_df, y_test, X_val_df, y_val, feature_names = prepare_data(client_id)
    
    # Reshape to sequences
    temporal_train, static_train, temporal_dim, static_dim = reshape_to_sequences(X_train_df, feature_names)
    temporal_val, static_val, _, _ = reshape_to_sequences(X_val_df, feature_names)
    temporal_test, static_test, _, _ = reshape_to_sequences(X_test_df, feature_names)
    
    # Handle NaN/Inf
    if temporal_train is not None:
        temporal_train = np.nan_to_num(temporal_train, nan=0.0, posinf=0.0, neginf=0.0)
        temporal_val = np.nan_to_num(temporal_val, nan=0.0, posinf=0.0, neginf=0.0)
        temporal_test = np.nan_to_num(temporal_test, nan=0.0, posinf=0.0, neginf=0.0)
    if static_train is not None:
        static_train = np.nan_to_num(static_train, nan=0.0, posinf=0.0, neginf=0.0)
        static_val = np.nan_to_num(static_val, nan=0.0, posinf=0.0, neginf=0.0)
        static_test = np.nan_to_num(static_test, nan=0.0, posinf=0.0, neginf=0.0)
    
    y_train_np = np.asarray(y_train, dtype=np.float32).ravel()
    y_val_np = np.asarray(y_val, dtype=np.float32).ravel()
    y_test_np = np.asarray(y_test, dtype=np.float32).ravel()
    
    # Convert to tensors
    temporal_train_tensor = torch.FloatTensor(temporal_train) if temporal_train is not None else None
    static_train_tensor = torch.FloatTensor(static_train) if static_train is not None else None
    temporal_val_tensor = torch.FloatTensor(temporal_val) if temporal_val is not None else None
    static_val_tensor = torch.FloatTensor(static_val) if static_val is not None else None
    temporal_test_tensor = torch.FloatTensor(temporal_test) if temporal_test is not None else None
    static_test_tensor = torch.FloatTensor(static_test) if static_test is not None else None
    y_train_tensor = torch.FloatTensor(y_train_np)
    
    # Compute class weights
    pos_count = y_train_np.sum()
    neg_count = len(y_train_np) - pos_count
    pos_weight_val = np.clip(neg_count / max(pos_count, 1), 0.5, 10.0)
    pos_weight = torch.tensor([pos_weight_val]).to(DEVICE)
    
    logger.info(f"Client {client_id}: {neg_count:.0f} neg, {pos_count:.0f} pos, weight={pos_weight_val:.2f}")
    
    theoretical_kappa = check_or_compute_kappa(compute_limits, client_name, THEORETICAL_KAPPA_FILE_PATTERN, DATA_BALANCE, X_test_df, X_train_df, X_val_df, y_test_np, y_train_np, y_val_np, HYPERPARAMS_DIR)
    
    # Handle hyperparameters
    existing_params = load_hyperparams(client_name, HYPERPARAMS_FILE_PATTERN, DATA_BALANCE) if USE_EXISTING_HYPERPARAMS else None
    
    if existing_params:
        best_params = existing_params
        optuna_best_score = None
        logger.info("Using existing hyperparameters")
    else:
        logger.info(f"Running Optuna optimization ({OPTUNA_TRIALS} trials)...")
        study = optuna.create_study(direction="minimize", 
                                    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
        study.optimize(
            create_objective(temporal_train, static_train, y_train_np, temporal_val, static_val, y_val_np,
                           pos_weight, temporal_dim, static_dim), 
            n_trials=OPTUNA_TRIALS, 
            show_progress_bar=True
        )
        best_params = study.best_params
        optuna_best_score = study.best_value
        logger.info(f"Best score: {optuna_best_score:.4f}")
        mlflow.log_metric("optuna_best_trial", float(study.best_trial.number))
        save_hyperparams(client_name, best_params, HYPERPARAMS_FILE_PATTERN, DATA_BALANCE)
    
    # Train final model
    logger.info("Training final model on full training set...")
    hidden_dim = best_params.get("hidden_dim", 64)
    num_layers = best_params.get("num_layers", 2)
    dropout = best_params.get("dropout", 0.2)
    model = LSTMClassifier(temporal_dim, static_dim, hidden_dim, num_layers, dropout).to(DEVICE)
    model = train_model(model, temporal_train_tensor, static_train_tensor, y_train_tensor, 
                       best_params, pos_weight, temporal_val_tensor, static_val_tensor, y_val_np)
    logger.info(f"Early stopping: best epoch {model.best_epoch}")
    
    val_probs = predict_proba(model, temporal_val_tensor, static_val_tensor)
    grid = np.linspace(0.05, 0.95, 91)
    threshold = float(grid[np.argmax([f1_score(y_val_np, (val_probs > t).astype(int), zero_division=0) for t in grid])])
    best_params["threshold"] = threshold
    
    # Train and validation metrics with the same threshold (diagnostics only)
    train_probs = predict_proba(model, temporal_train_tensor, static_train_tensor)
    train_metrics = compute_all_metrics(y_train_np, train_probs, threshold, n_boot=0)
    val_metrics = compute_all_metrics(y_val_np, val_probs, threshold, n_boot=0)
    
    # Test set evaluation
    test_probs = predict_proba(model, temporal_test_tensor, static_test_tensor)
    test_metrics = compute_all_metrics(y_test_np, test_probs, threshold)

    # Save model
    model_dir = RESULTS_DIR / "models"
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / f"lstm_model_client_{client_id}_{DATA_BALANCE}.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'temporal_dim': temporal_dim,
        'static_dim': static_dim,
        'hidden_dim': hidden_dim,
        'num_layers': num_layers,
        'dropout': dropout,
        'threshold': threshold,
        'hyperparameters': best_params
    }, model_path)
    logger.info(f"Saved model to {model_path}")
    
    results = {
        "client_id": client_id,
        "client_name": client_name,
        "temporal_features": temporal_dim,
        "static_features": static_dim,
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
    logger.info("Starting ML pipeline with LSTM, Optuna tuning and 5-fold CV")
    
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    all_results = []
    
    for client_id in range(1, 6):
        run_name = f"client_lstm_{client_id}_{get_client_name(client_id)}_{MLFLOW_SUFFIX}"
        
        with mlflow.start_run(run_name=run_name):
            try:
                results, model, probs = train_client(client_id, compute_limits=True)
                log_to_mlflow_local(results, MODEL, OPTUNA_TRIALS, DATA_BALANCE)
                log_split_metrics(results)
                mlflow.log_metric("best_epoch", float(results["best_epoch"]))
                all_results.append(results)
                
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