"""Federated Learning Client with Neural Network and full metrics logging."""

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
from mlp_config import MLFLOW_EXPERIMENT, MLFLOW_SUFFIX, MODEL, FL_BOOST_HYPERPARAMS_FILE_PATTERN

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--server_address", type=str, default="server:8095")
parser.add_argument("--client_id", type=int, default=1)
args = parser.parse_known_args()[0]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
# FL Client
# =============================================================================

class NNClient(Mixin, fl.client.NumPyClient):
    
    def __init__(self, client_id: int, X_train, y_train, X_test, y_test, 
                 X_val, y_val, feature_names, mlflow_run_id: str):
        self.client_id = client_id
        self.client_name = get_client_name(client_id)
        self.input_dim = X_train.shape[1]
        self.feature_names = feature_names
        self.global_params = None
        self.mlflow_run_id = mlflow_run_id
        self.current_round = 0
        self.modelname = MODEL
        self.BOOST_HYPERPARAMS_FILE = str(FL_BOOST_HYPERPARAMS_FILE_PATTERN).format(client=self.client_name)
        
        
        self.X_train_np, self.y_train_np = self._to_X(X_train), self._to_y(y_train)
        self.X_test_np,  self.y_test_np  = self._to_X(X_test),  self._to_y(y_test)
        self.X_val_np,   self.y_val_np   = self._to_X(X_val),  self. _to_y(y_val)
        
        # Convert to tensors - Train
        self.X_train = torch.FloatTensor(self.X_train_np)
        self.y_train = torch.FloatTensor(self.y_train_np)
        
        # Convert to tensors - Test
        self.X_test = torch.FloatTensor(self.X_test_np)
        self.y_test = torch.FloatTensor(self.y_test_np)
        
        # Convert to tensors - Validation
        self.X_val = torch.FloatTensor(self.X_val_np)
        self.y_val = torch.FloatTensor(self.y_val_np)
        
        # Class weights (computed from training data only)
        pos_count = self.y_train_np.sum()
        neg_count = len(self.y_train_np) - pos_count
        pos_weight_val = neg_count / max(pos_count, 1)
        pos_weight_val = np.clip(pos_weight_val, 0.5, 10.0)
        self.pos_weight = torch.tensor([pos_weight_val]).to(DEVICE)
        
        logger.info(f"Client {client_id}: {neg_count:.0f} neg, {pos_count:.0f} pos, weight={pos_weight_val:.2f}")
        logger.info(f"Client {client_id}: {len(self.X_train)} train, {len(self.X_val)} val, {len(self.X_test)} test")
        
        # Model - Neural Network
        self.model = MLP(self.input_dim, hidden_dim=128, dropout=0.2).to(DEVICE)
        self.current_params = {"hidden_dim": 128, "dropout": 0.2}
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

    def _predict(self, X_tensor: torch.Tensor) -> np.ndarray:
        """Get predictions for given input tensor."""
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_tensor.to(DEVICE)).cpu()
            y_prob = torch.sigmoid(logits).numpy()
        return np.clip(y_prob, 1e-7, 1 - 1e-7)

    def _rebuild_model_if_needed(self, config: dict) -> bool:
        """Rebuild model if architecture hyperparameters changed."""
        hidden_dim = int(config.get("hidden_dim", 128))
        dropout = float(config.get("dropout", 0.2))
        
        # Clamp values
        hidden_dim = max(32, min(512, hidden_dim))
        dropout = max(0.0, min(0.5, dropout))
        
        needs_rebuild = (
            config.get("reset_params", 0) == 1 or
            hidden_dim != self.current_params.get("hidden_dim") or
            abs(dropout - self.current_params.get("dropout", 0.2)) > 0.05
        )
        
        if needs_rebuild:
            self.model = MLP(self.input_dim, hidden_dim, dropout).to(DEVICE)
            self.current_params = {"hidden_dim": hidden_dim, "dropout": dropout}
            self.global_params = None  # Reset FedProx reference
            logger.info(f"  Model rebuilt: hidden={hidden_dim}, dropout={dropout:.2f}")
            return True
        return False

    def fit(self, parameters, config):
        global_round = int(config.get("global_round", 0))
        self.current_round = global_round
        
        # Get and clamp hyperparameters
        lr = float(config.get("learning_rate", 0.001))
        lr = max(1e-5, min(0.01, lr))
        
        weight_decay = float(config.get("weight_decay", 1e-4))
        weight_decay = max(1e-6, min(1e-2, weight_decay))
        
        local_epochs = int(config.get("local_epochs", FL_LOCAL_EPOCHS))
        local_epochs = max(10, min(200, local_epochs))
        
        batch_size = int(config.get("batch_size", FL_BATCH_SIZE))
        batch_size = max(8, min(256, batch_size))
        
        # Optimizer parameters
        beta1 = float(config.get("adam_beta1", 0.9))
        beta1 = max(0.8, min(0.99, beta1))
        
        beta2 = float(config.get("adam_beta2", 0.999))
        beta2 = max(0.99, min(0.9999, beta2))
        
        # Training stability
        grad_clip = float(config.get("grad_clip_norm", 1.0))
        grad_clip = max(0.1, min(10.0, grad_clip))
        
        warmup_epochs = int(config.get("lr_warmup_epochs", 0))
        warmup_epochs = max(0, min(20, warmup_epochs))
        
        label_smoothing = float(config.get("label_smoothing", 0.0))
        label_smoothing = max(0.0, min(0.2, label_smoothing))
        
        # FedProx: Get proximal_mu (from HPO)
        proximal_mu = float(config.get("proximal_mu", 0.0))
        proximal_mu = max(0.0, min(10.0, proximal_mu))
        
        # Threshold (from server)
        threshold = float(config.get("threshold", self.threshold))
        self.threshold = threshold
        
        # Early stopping config (uses validation set)
        use_early_stopping = bool(config.get("use_early_stopping", False))
        patience = int(config.get("early_stopping_patience", 5))
        
        logger.info(f"Client {self.client_id} - Round {global_round}: lr={lr:.6f}, "
                    f"epochs={local_epochs}, bs={batch_size}, μ={proximal_mu:.4f}, thresh={threshold:.2f}")
        
        # Check if model needs rebuilding
        rebuilt = self._rebuild_model_if_needed(config)
        
        if not rebuilt and parameters is not None and len(parameters) > 0:
            try:
                set_parameters(self.model, parameters)
                self.global_params = [p.clone().detach() for p in self.model.parameters()]
            except Exception as e:
                logger.warning(f"Parameter load failed: {e}")
        
        # Training
        self.model.train()
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
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
            criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight, reduction='none')
        else:
            criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        
        dataset = TensorDataset(self.X_train, self.y_train)
        loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            drop_last=len(dataset) > batch_size
        )
        
        # Early stopping state (monitored on VALIDATION set)
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None
        
        for epoch in range(local_epochs):
            self.model.train()
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                
                optimizer.zero_grad()
                logits = self.model(X_batch)
                
                # Base loss
                if label_smoothing > 0:
                    y_smooth = y_batch * (1 - label_smoothing) + 0.5 * label_smoothing
                    base_loss = criterion(logits, y_smooth).mean()
                else:
                    base_loss = criterion(logits, y_batch)
                
                # FedProx: Add proximal term
                if proximal_mu > 0 and self.global_params is not None:
                    proximal_term = 0.0
                    for local_param, global_param in zip(self.model.parameters(), self.global_params):
                        proximal_term += torch.sum((local_param - global_param) ** 2)
                    loss = base_loss + (proximal_mu / 2) * proximal_term
                else:
                    loss = base_loss
                
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    max_norm=grad_clip
                )
                optimizer.step()
            
            scheduler.step()
            
            # Early stopping check using VALIDATION set
            if use_early_stopping:
                val_prob = self._predict(self.X_val)
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
        
        return get_parameters(self.model), len(self.X_train), {}



if __name__ == "__main__":
    client_id = args.client_id
    client_name = get_client_name(client_id)
    
    logger.info(f"Starting FL client {client_id} ({client_name})")
    
    # Load data with train/test/val split
    X_train, y_train, X_test, y_test, X_val, y_val, feature_names = prepare_data(client_id)
    
    mlflow.set_experiment(f"{MLFLOW_EXPERIMENT}")
    run_name = f"fl_client_{client_id}_{client_name}_{MLFLOW_SUFFIX}"

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        log_mlflow_params(client_id, client_name, DATA_BALANCE, MODEL, FL_LOCAL_EPOCHS, FL_BATCH_SIZE, X_train, X_val, X_test)
        client = NNClient(
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