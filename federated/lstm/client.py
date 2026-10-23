"""Federated Learning Client with LSTM and full metrics logging."""

import argparse
import logging
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import flwr as fl
import mlflow
from sklearn.metrics import log_loss

#custom imports
from mydataloading import prepare_data, get_client_name
from kappa_limits import compute_theoretical_kappa
from base_functions import load_theoretical_kappa, save_theoretical_kappa, Mixin, set_parameters, get_parameters, log_mlflow_params
from general_config import DATA_BALANCE, HYPERPARAMS_DIR, THEORETICAL_KAPPA_FILE_PATTERN, FL_LOCAL_EPOCHS, FL_BATCH_SIZE
from lstm_config import MLFLOW_EXPERIMENT, MLFLOW_SUFFIX, MODEL

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--server_address", type=str, default="server:8095")
parser.add_argument("--client_id", type=int, default=1)
args = parser.parse_known_args()[0]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# Temporal Feature Detection
# =============================================================================

def detect_temporal_features(feature_names):
    """Detect temporal features ending in _1, _2, _3, _4 and group them."""
    temporal_groups = {}
    static_features = []
    
    for feat in feature_names:
        if feat.endswith(('_1', '_2', '_3', '_4')):
            base_name = feat.rsplit('_', 1)[0]
            time_idx = int(feat.rsplit('_', 1)[1]) - 1
            
            if base_name not in temporal_groups:
                temporal_groups[base_name] = [None] * 4
            temporal_groups[base_name][time_idx] = feat
        else:
            static_features.append(feat)
    
    temporal_groups = {k: v for k, v in temporal_groups.items() if None not in v}
    return temporal_groups, static_features


def reshape_to_sequences(X_df, feature_names):
    """Reshape wide-format temporal data to (samples, seq_len, features)."""
    temporal_groups, static_features = detect_temporal_features(feature_names)
    
    temporal_data = []
    for base_name in sorted(temporal_groups.keys()):
        time_series_cols = temporal_groups[base_name]
        time_series = X_df[time_series_cols].values
        temporal_data.append(time_series)
    
    if temporal_data:
        temporal_array = np.stack(temporal_data, axis=1)
        temporal_array = np.transpose(temporal_array, (0, 2, 1))
    else:
        temporal_array = None
    
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
            features.append(hidden[-1])
        
        if self.has_static and static is not None:
            features.append(static)
        
        combined = torch.cat(features, dim=1)
        return self.fc(combined).squeeze(-1)


# =============================================================================
# FL Client
# =============================================================================

class LSTMClient(Mixin, fl.client.NumPyClient):
    
    def __init__(self, client_id: int, X_train, y_train, X_test, y_test, 
                 X_val, y_val, feature_names, mlflow_run_id: str):
        self.client_id = client_id
        self.client_name = get_client_name(client_id)
        self.feature_names = feature_names
        self.global_params = None
        self.mlflow_run_id = mlflow_run_id
        self.current_round = 0
        self.modelname = MODEL
        
        self.X_train_np, self.y_train_np = self._to_X(X_train), self._to_y(y_train)
        self.X_test_np,  self.y_test_np  = self._to_X(X_test),  self._to_y(y_test)
        self.X_val_np,   self.y_val_np   = self._to_X(X_val),  self. _to_y(y_val)
        
        # Reshape all datasets
        temporal_train, static_train, temporal_dim, static_dim = reshape_to_sequences(X_train, feature_names)
        temporal_test, static_test, _, _ = reshape_to_sequences(X_test, feature_names)
        temporal_val, static_val, _, _ = reshape_to_sequences(X_val, feature_names)
        
        self.temporal_dim = temporal_dim
        self.static_dim = static_dim
        
        # Handle NaN/Inf and store numpy versions
        self.temporal_train_np = np.nan_to_num(temporal_train, nan=0.0, posinf=0.0, neginf=0.0) if temporal_train is not None else None
        self.static_train_np = np.nan_to_num(static_train, nan=0.0, posinf=0.0, neginf=0.0) if static_train is not None else None
        self.temporal_test_np = np.nan_to_num(temporal_test, nan=0.0, posinf=0.0, neginf=0.0) if temporal_test is not None else None
        self.static_test_np = np.nan_to_num(static_test, nan=0.0, posinf=0.0, neginf=0.0) if static_test is not None else None
        self.temporal_val_np = np.nan_to_num(temporal_val, nan=0.0, posinf=0.0, neginf=0.0) if temporal_val is not None else None
        self.static_val_np = np.nan_to_num(static_val, nan=0.0, posinf=0.0, neginf=0.0) if static_val is not None else None
        
        self.y_train_np = np.asarray(y_train, dtype=np.float32).ravel()
        self.y_test_np = np.asarray(y_test, dtype=np.float32).ravel()
        self.y_val_np = np.asarray(y_val, dtype=np.float32).ravel()
        
        # Convert to tensors
        self.temporal_train = torch.FloatTensor(self.temporal_train_np) if self.temporal_train_np is not None else None
        self.static_train = torch.FloatTensor(self.static_train_np) if self.static_train_np is not None else None
        self.temporal_test = torch.FloatTensor(self.temporal_test_np) if self.temporal_test_np is not None else None
        self.static_test = torch.FloatTensor(self.static_test_np) if self.static_test_np is not None else None
        self.temporal_val = torch.FloatTensor(self.temporal_val_np) if self.temporal_val_np is not None else None
        self.static_val = torch.FloatTensor(self.static_val_np) if self.static_val_np is not None else None
        
        self.y_train = torch.FloatTensor(self.y_train_np)
        self.y_test = torch.FloatTensor(self.y_test_np)
        self.y_val = torch.FloatTensor(self.y_val_np)
        
        # Class weights
        pos_count = self.y_train_np.sum()
        neg_count = len(self.y_train_np) - pos_count
        pos_weight_val = np.clip(neg_count / max(pos_count, 1), 0.5, 10.0)
        self.pos_weight = torch.tensor([pos_weight_val]).to(DEVICE)
        
        logger.info(f"Client {client_id}: {neg_count:.0f} neg, {pos_count:.0f} pos, weight={pos_weight_val:.2f}")
        logger.info(f"Client {client_id}: {len(self.y_train)} train, {len(self.y_val)} val, {len(self.y_test)} test")
        
        # Model - LSTM
        self.model = LSTMClassifier(temporal_dim, static_dim, hidden_dim=64, num_layers=2, dropout=0.2).to(DEVICE)
        self.current_params = {"hidden_dim": 64, "num_layers": 2, "dropout": 0.2}
        self.threshold = 0.5
            
        # Theoretical kappa (computed from all data)
        existing_kappa = load_theoretical_kappa(self.client_name, THEORETICAL_KAPPA_FILE_PATTERN, DATA_BALANCE)
        if existing_kappa:
            self.theoretical_kappa = existing_kappa["theoretical_kappa"]
        else:
            X_all = np.concatenate([self.X_train_np, self.X_val_np, self.X_test_np], axis=0)
            y_all = np.concatenate([self.y_train_np, self.y_val_np, self.y_test_np], axis=0)
            kappa_results = compute_theoretical_kappa(X_all, y_all)
            self.theoretical_kappa = kappa_results.get("theoretical_kappa", 0.0)
            save_theoretical_kappa(self.client_name, self.theoretical_kappa, HYPERPARAMS_DIR, THEORETICAL_KAPPA_FILE_PATTERN, DATA_BALANCE)

    def get_parameters(self, config):
        return get_parameters(self.model)

    def _predict(self, temporal_tensor, static_tensor) -> np.ndarray:
        """Get predictions for given input tensors."""
        self.model.eval()
        with torch.no_grad():
            temporal_dev = temporal_tensor.to(DEVICE) if temporal_tensor is not None else None
            static_dev = static_tensor.to(DEVICE) if static_tensor is not None else None
            logits = self.model(temporal=temporal_dev, static=static_dev).cpu()
            y_prob = torch.sigmoid(logits).numpy()
        return np.clip(y_prob, 1e-7, 1 - 1e-7)


    def _rebuild_model_if_needed(self, config: dict) -> bool:
        """Rebuild model if architecture hyperparameters changed."""
        hidden_dim = int(config.get("hidden_dim", 64))
        num_layers = int(config.get("num_layers", 2))
        dropout = float(config.get("dropout", 0.2))
        
        hidden_dim = max(32, min(256, hidden_dim))
        num_layers = max(1, min(3, num_layers))
        dropout = max(0.0, min(0.5, dropout))
        
        needs_rebuild = (
            config.get("reset_params", 0) == 1 or
            hidden_dim != self.current_params.get("hidden_dim") or
            num_layers != self.current_params.get("num_layers") or
            abs(dropout - self.current_params.get("dropout", 0.2)) > 0.05
        )
        
        if needs_rebuild:
            self.model = LSTMClassifier(self.temporal_dim, self.static_dim, hidden_dim, num_layers, dropout).to(DEVICE)
            self.current_params = {"hidden_dim": hidden_dim, "num_layers": num_layers, "dropout": dropout}
            self.global_params = None
            logger.info(f"  Model rebuilt: hidden={hidden_dim}, layers={num_layers}, dropout={dropout:.2f}")
            return True
        return False

    def fit(self, parameters, config):
        global_round = int(config.get("global_round", 0))
        self.current_round = global_round
        
        lr = max(1e-5, min(0.01, float(config.get("learning_rate", 0.001))))
        weight_decay = max(1e-6, min(1e-2, float(config.get("weight_decay", 1e-4))))
        local_epochs = max(10, min(200, int(config.get("local_epochs", FL_LOCAL_EPOCHS))))
        batch_size = max(8, min(256, int(config.get("batch_size", FL_BATCH_SIZE))))
        
        threshold = float(config.get("threshold", self.threshold))
        self.threshold = threshold
        
        use_early_stopping = bool(config.get("use_early_stopping", False))
        patience = int(config.get("early_stopping_patience", 5))
        
        logger.info(f"Client {self.client_id} - Round {global_round}: lr={lr:.6f}, "
                    f"epochs={local_epochs}, bs={batch_size}, thresh={threshold:.2f}")
        
        rebuilt = self._rebuild_model_if_needed(config)
        
        if not rebuilt and parameters is not None and len(parameters) > 0:
            try:
                set_parameters(self.model, parameters)
                self.global_params = [p.clone().detach() for p in self.model.parameters()]
            except Exception as e:
                logger.warning(f"Parameter load failed: {e}")
        
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=local_epochs)
        criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        
        # Create dataset
        tensors = []
        if self.temporal_train is not None:
            tensors.append(self.temporal_train)
        if self.static_train is not None:
            tensors.append(self.static_train)
        tensors.append(self.y_train)
        
        dataset = TensorDataset(*tensors)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                           drop_last=len(dataset) > batch_size)
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None
        
        for epoch in range(local_epochs):
            self.model.train()
            for batch in loader:
                y_batch = batch[-1].to(DEVICE)
                temporal_batch = batch[0].to(DEVICE) if self.temporal_train is not None else None
                static_batch = batch[1].to(DEVICE) if len(batch) == 3 and self.static_train is not None else None
                
                optimizer.zero_grad()
                logits = self.model(temporal=temporal_batch, static=static_batch)
                loss = criterion(logits, y_batch)
                
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    loss.backward()
                    optimizer.step()
            
            scheduler.step()
            
            if use_early_stopping:
                val_prob = self._predict(self.temporal_val, self.static_val)
                val_loss = log_loss(self.y_val_np, val_prob)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(f"  Early stopping at epoch {epoch+1}/{local_epochs}")
                        if best_state is not None:
                            self.model.load_state_dict(best_state)
                        break
        
        return get_parameters(self.model), len(self.y_train), {}


if __name__ == "__main__":
    client_id = args.client_id
    client_name = get_client_name(client_id)
    
    logger.info(f"Starting FL LSTM client {client_id} ({client_name})")
    
    X_train, y_train, X_test, y_test, X_val, y_val, feature_names = prepare_data(client_id)
    
    mlflow.set_experiment(f"{MLFLOW_EXPERIMENT}")
    run_name = f"fl_client_{client_id}_{client_name}_{MLFLOW_SUFFIX}"

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        log_mlflow_params(client_id, client_name, DATA_BALANCE, MODEL, FL_LOCAL_EPOCHS, FL_BATCH_SIZE, X_train, X_val, X_test)
        
        client = LSTMClient(
            client_id,
            X_train, y_train, 
            X_test, y_test,
            X_val, y_val,
            feature_names,
            run_id
        )

        fl.client.start_client(
            server_address=args.server_address,
            client=client.to_client()
        )
    logger.info(f"Client {client_id} finished")