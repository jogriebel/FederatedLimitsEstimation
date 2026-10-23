import pandas as pd
import logging
import torch
import torch.nn as nn
import mlflow
import os
import numpy as np
import json
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from base_functions import compute_all_metrics
from general_config import FL_NUM_ROUNDS, RESULTS_DIR, FL_METRICS_CSV_PATTERN

def log_mlflow_params(client_id, client_name, DATA_BALANCE, MODEL, FL_LOCAL_EPOCHS, FL_BATCH_SIZE, X_train, X_val, X_test):
    mlflow.log_param("client_id", client_id)
    mlflow.log_param("client_name", client_name)
    mlflow.log_param("data_balance", DATA_BALANCE)
    mlflow.log_param("model_type", MODEL)
    mlflow.log_param("fl_local_epochs", FL_LOCAL_EPOCHS)
    mlflow.log_param("fl_batch_size", FL_BATCH_SIZE)
    mlflow.log_param("train_size", len(X_train))
    mlflow.log_param("val_size", len(X_val))
    mlflow.log_param("test_size", len(X_test))
    

def set_parameters(model: nn.Module, parameters: list):
    """Set model parameters from numpy arrays."""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)

def get_parameters(model: nn.Module) -> list:
    """Get model parameters as numpy arrays."""
    return [val.cpu().numpy() for val in model.state_dict().values()]

class Mixin:
    def _to_X(self, X):
        return np.nan_to_num(np.asarray(X, dtype=np.float32),
                         nan=0.0, posinf=0.0, neginf=0.0)

    def _to_y(self, y):
        return np.asarray(y, dtype=np.float32).ravel()
    
    THRESHOLD_GRID = np.linspace(0.05, 0.95, 91)
    BOOST_HYPERPARAMS_FILE = None

    def _cm_grid(self, y, p):
        """(TP, FP, FN) counts at each grid threshold."""
        p, y = p[None, :] > self.THRESHOLD_GRID[:, None], np.asarray(y).astype(bool)
        return np.stack([(p & y).sum(1), (p & ~y).sum(1), (~p & y).sum(1)]).astype(np.int64)

    def _probs(self, split):
        if self.modelname == "LSTM":
            return self._predict(getattr(self, f"temporal_{split}"), getattr(self, f"static_{split}"))
        return self._predict(getattr(self, f"X_{split}"))

    def _boost_fit(self, parameters, config):
        self.fit(parameters, config)

    def _local_boost(self, parameters, config, y_prob):
        """Test metrics after one local training pass from the global model, threshold re-tuned on local val."""
        self._boost_fit(parameters, config)
        tp, fp, fn = self._cm_grid(self.y_val_np, self._probs("val"))
        f1 = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros(len(tp)), where=(2 * tp + fp + fn) > 0)
        boost = compute_all_metrics(self.y_test_np, self._probs("test"), float(self.THRESHOLD_GRID[np.argmax(f1)]),
                                    ref=(y_prob, self.threshold))
        if self.BOOST_HYPERPARAMS_FILE:
            with open(self.BOOST_HYPERPARAMS_FILE, "w") as f:
                json.dump({"threshold": boost["threshold"]}, f, indent=2)
        return {f"boost_{k}": v for k, v in boost.items()}

    def evaluate(self, parameters, config):
        if parameters is not None and len(parameters) > 0:
            try:
                set_parameters(self.model, parameters)
            except Exception as e:
                logger.warning(f"Parameter load failed: {e}")
                raise
        
        current_round = int(config.get("global_round", self.current_round))
        
        # Update threshold from config if provided
        threshold = float(config.get("threshold", self.threshold))
        self.threshold = threshold
        
        # Check if we're in HPO phase or final evaluation
        # HPO phase: use validation set | Final phase: use test set
        is_hpo_phase = bool(config.get("hpo_phase", True))
        
        if is_hpo_phase:
            # =====================================================
            # HPO PHASE: Evaluate on VALIDATION set only
            # =====================================================
            if self.modelname == "LSTM":
                y_prob = self._predict(self.temporal_val, self.static_val)
            else:    
                y_prob = self._predict(self.X_val)
            y_pred = (y_prob > self.threshold).astype(int)
            
            metrics = compute_all_metrics(self.y_val_np, y_prob, self.threshold, n_boot=0)
            metrics["theoretical_kappa"] = float(self.theoretical_kappa)
            metrics["threshold"] = float(self.threshold)
            metrics["eval_set"] = "validation"
            metrics["cm_grid"] = self._cm_grid(self.y_val_np, y_prob).tobytes()
            

            
            eval_size = len(self.y_val)
            logger.info(f"[HPO] Round {current_round}: VAL AUC={metrics['AUC']:.4f}, "
                    f"AUPRC={metrics['AUPRC']:.4f}, F1={metrics['F1']:.4f}")
        else:
            # =====================================================
            # FINAL PHASE: Evaluate on TEST set (after HPO complete)
            # =====================================================
            if self.modelname == "LSTM":
                y_prob = self._predict(self.temporal_test, self.static_test)
            else:    
                y_prob = self._predict(self.X_test)
            y_pred = (y_prob > self.threshold).astype(int)
            
            metrics = compute_all_metrics(self.y_test_np, y_prob, self.threshold)
            metrics["theoretical_kappa"] = float(self.theoretical_kappa)
            metrics["threshold"] = float(self.threshold)
            metrics["eval_set"] = "test"
            
            
            # Also include validation metrics for comparison
            
            
            if self.modelname == "LSTM":
                val_prob = self._predict(self.temporal_val, self.static_val)
            else:    
                val_prob = self._predict(self.X_val)
            
            
            val_metrics = compute_all_metrics(self.y_val_np, val_prob, self.threshold, n_boot=0)
            metrics["val_AUC"] = val_metrics["AUC"]
            metrics["val_AUPRC"] = val_metrics["AUPRC"]
            metrics["val_F1"] = val_metrics["F1"]
            metrics["val_log_loss"] = val_metrics["log_loss"]
            metrics["val_Precision"] = val_metrics["Precision"]
            metrics["val_Recall"] = val_metrics["Recall"]
            metrics["val_Sensitivity"] = val_metrics["Sensitivity"]
            metrics["val_Specificity"] = val_metrics["Specificity"]
            metrics["val_Brier"] = val_metrics["Brier"]
            metrics["val_actual_kappa"] = val_metrics["actual_kappa"]
            metrics.update(self._local_boost(parameters, config, y_prob))
            
            if getattr(self, "BOOST_HYPERPARAMS_FILE", None):
                with open(self.BOOST_HYPERPARAMS_FILE, "w") as f:
                    json.dump({"threshold": float(metrics["boost_threshold"])}, f, indent=2)
            
            eval_size = len(self.y_test)
            logger.info(f"[FINAL] Round {current_round}: TEST AUC={metrics['AUC']:.4f}, "
                    f"AUPRC={metrics['AUPRC']:.4f}, F1={metrics['F1']:.4f}")
        
        # MLflow logging
        metrics_to_log = {
            "AUC": metrics["AUC"],
            "AUPRC": metrics["AUPRC"],
            "Precision": metrics["Precision"],
            "Recall": metrics["Recall"],
            "F1": metrics["F1"],
            "Brier": metrics["Brier"],
            "log_loss": metrics["log_loss"],
            "Sensitivity": metrics["Sensitivity"],
            "Specificity": metrics["Specificity"],
            "actual_kappa": metrics["actual_kappa"],
            "theoretical_kappa": metrics["theoretical_kappa"],
            "threshold": metrics["threshold"],
        }
        
        # Add validation metrics if in final phase
        if not is_hpo_phase:
            metrics_to_log["val_AUC"] = metrics["val_AUC"]
            metrics_to_log["val_AUPRC"] = metrics["val_AUPRC"]
            metrics_to_log["val_F1"] = metrics["val_F1"]
            metrics_to_log["val_log_loss"] = metrics["val_log_loss"]
            metrics_to_log["val_Precision"] = metrics["val_Precision"]
            metrics_to_log["val_Recall"] = metrics["val_Recall"]
            metrics_to_log["val_Sensitivity"] = metrics["val_Sensitivity"]
            metrics_to_log["val_Specificity"] = metrics["val_Specificity"]
            metrics_to_log["val_Brier"] = metrics["val_Brier"]
            metrics_to_log["val_actual_kappa"] = metrics["val_actual_kappa"]
        metrics_to_log.update({k: v for k, v in metrics.items() if "_ci_" in k or k.startswith("boost_")})
        
        mlflow.log_metrics(metrics_to_log, step=current_round)
        logger.info(f"Logged metrics to MLflow run {self.mlflow_run_id} at step {current_round}")
        
        # Save to CSV
        RESULTS_DIR.mkdir(exist_ok=True)
        csv_path = str(FL_METRICS_CSV_PATTERN).format(client_id=self.client_id, modelname=self.modelname)
        if not is_hpo_phase: csv_path = csv_path.replace(".csv", "_test.csv")
        
        # Base row with all core metrics
        row = {
            "client_id": self.client_id,
            "round": current_round,
            "eval_set": metrics["eval_set"],
            "AUC": metrics["AUC"],
            "AUPRC": metrics["AUPRC"],
            "F1": metrics["F1"],
            "Precision": metrics["Precision"],
            "Recall": metrics["Recall"],
            "Sensitivity": metrics["Sensitivity"],
            "Specificity": metrics["Specificity"],
            "Brier": metrics["Brier"],
            "log_loss": metrics["log_loss"],
            "actual_kappa": metrics["actual_kappa"],
            "theoretical_kappa": metrics["theoretical_kappa"],
            "threshold": metrics["threshold"],
        }

        
        # Add validation metrics if in final phase (test set evaluation)
        if not is_hpo_phase:
            row["val_AUC"] = metrics["val_AUC"]
            row["val_AUPRC"] = metrics["val_AUPRC"]
            row["val_F1"] = metrics["val_F1"]
            row["val_log_loss"] = metrics["val_log_loss"]
            row["val_Precision"] = metrics["val_Precision"]
            row["val_Recall"] = metrics["val_Recall"]
            row["val_Sensitivity"] = metrics["val_Sensitivity"]
            row["val_Specificity"] = metrics["val_Specificity"]
            row["val_Brier"] = metrics["val_Brier"]
            row["val_actual_kappa"] = metrics["val_actual_kappa"]
        row.update({k: v for k, v in metrics.items() if "_ci_" in k or k.startswith("boost_")})
        
        pd.DataFrame([row]).to_csv(
            csv_path, mode='a', header=not os.path.exists(csv_path), index=False
        )
        
        return metrics["log_loss"], eval_size, metrics