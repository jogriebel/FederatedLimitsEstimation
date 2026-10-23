"""
Federated XGBoost Client - Histogram with HPO support.
Client computes gradient histograms, server builds trees.
Updated with proper train/val/test split, HPO phase handling, and SHAP.
"""

import os, json, argparse, logging, warnings
from typing import Dict, Optional, List
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import flwr as fl
import mlflow
from flwr.common import Code, EvaluateIns, EvaluateRes, FitIns, FitRes, GetParametersIns, GetParametersRes, Parameters, Status


#custom imports
from mydataloading import prepare_data, get_client_name
from kappa_limits import compute_theoretical_kappa
from base_functions import load_theoretical_kappa, save_theoretical_kappa, Mixin, log_mlflow_params, compute_all_metrics
from general_config import DATA_BALANCE, HYPERPARAMS_DIR, THEORETICAL_KAPPA_FILE_PATTERN, FL_LOCAL_EPOCHS, FL_BATCH_SIZE, FL_NUM_ROUNDS, FL_METRICS_CSV_PATTERN
from xgb_config import MLFLOW_EXPERIMENT, MLFLOW_SUFFIX, MODEL, FL_HYPERPARAMS_FILE, FL_BOOST_HYPERPARAMS_FILE_PATTERN, XGB_NUM_BINS, RESULTS_DIR, XGB_BOOST_TREES
from server import FederatedTreeBuilder


warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--server_address", type=str, default="server:8095")
parser.add_argument("--client_id", type=int, default=1)
parser.add_argument("--num_bins", type=int, default=XGB_NUM_BINS)
args = parser.parse_known_args()[0]


@dataclass
class GradientHistogram:
    gradient_sums: np.ndarray
    hessian_sums: np.ndarray
    bin_counts: np.ndarray
    bin_edges: np.ndarray


@dataclass
class ClientHistogramData:
    num_bins: int
    num_features: int
    feature_histograms: Dict[int, GradientHistogram] = field(default_factory=dict)
    bin_edges: Dict[int, np.ndarray] = field(default_factory=dict)
    target_sum: float = 0.0
    sample_count: int = 0

    def to_bytes(self) -> bytes:
        return json.dumps({
            'num_bins': self.num_bins, 'num_features': self.num_features,
            'target_sum': self.target_sum, 'sample_count': self.sample_count,
            'bin_edges': {str(k): v.tolist() for k, v in self.bin_edges.items()},
            'feature_histograms': {str(k): {'gradient_sums': v.gradient_sums.tolist(), 'hessian_sums': v.hessian_sums.tolist(),
                                            'bin_counts': v.bin_counts.tolist(), 'bin_edges': v.bin_edges.tolist()}
                                   for k, v in self.feature_histograms.items()}
        }).encode()


@dataclass
class TreeNode:
    is_leaf: bool = False
    feature_idx: int = -1
    threshold: float = 0.0
    leaf_weight: float = 0.0
    left_child: Optional['TreeNode'] = None
    right_child: Optional['TreeNode'] = None


@dataclass
class FederatedTreeEnsemble:
    trees: List[TreeNode] = field(default_factory=list)
    base_score: float = 0.5
    learning_rate: float = 0.1
    num_features: int = 0

    def predict_raw(self, X: np.ndarray) -> np.ndarray:
        preds = np.full(X.shape[0], np.log(self.base_score / (1 - self.base_score + 1e-10)))
        for tree in self.trees:
            preds += self.learning_rate * self._predict_tree(tree, X)
        return preds

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(self.predict_raw(X), -500, 500)))

    def _predict_tree(self, node: TreeNode, X: np.ndarray) -> np.ndarray:
        if node.is_leaf: return np.full(X.shape[0], node.leaf_weight)
        preds = np.zeros(X.shape[0])
        mask = X[:, node.feature_idx] < node.threshold
        if node.left_child and mask.any(): preds[mask] = self._predict_tree(node.left_child, X[mask])
        if node.right_child and (~mask).any(): preds[~mask] = self._predict_tree(node.right_child, X[~mask])
        return preds

    @classmethod
    def from_bytes(cls, data: bytes) -> 'FederatedTreeEnsemble':
        p = json.loads(data.decode())
        ens = cls(base_score=p['base_score'], learning_rate=p['learning_rate'], num_features=p['num_features'])
        def des(d):
            if d['is_leaf']: return TreeNode(is_leaf=True, leaf_weight=d['leaf_weight'])
            n = TreeNode(is_leaf=False, feature_idx=d['feature_idx'], threshold=d['threshold'])
            if d.get('left'): n.left_child = des(d['left'])
            if d.get('right'): n.right_child = des(d['right'])
            return n
        ens.trees = [des(t) for t in p['trees']]
        return ens


class HistogramBuilder:
    def __init__(self, num_bins=255):
        self.num_bins = num_bins
        self.bin_edges: Dict[int, np.ndarray] = {}

    def build_sketch(self, X: np.ndarray):
        for i in range(X.shape[1]):
            vals = X[:, i][~np.isnan(X[:, i])]
            self.bin_edges[i] = np.unique(np.percentile(vals, np.linspace(0, 100, self.num_bins + 1))) if len(vals) > 0 else np.array([0.0, 1.0])

    def compute_histogram(self, X: np.ndarray, grads: np.ndarray, hess: np.ndarray) -> ClientHistogramData:
        if not self.bin_edges: self.build_sketch(X)
        hist = ClientHistogramData(self.num_bins, X.shape[1], bin_edges={k: v.copy() for k, v in self.bin_edges.items()})
        for i in range(X.shape[1]):
            edges = self.bin_edges.get(i, np.array([0.0, 1.0]))
            if len(edges) < 2:
                hist.feature_histograms[i] = GradientHistogram(np.array([grads.sum()]), np.array([hess.sum()]), np.array([len(grads)]), edges)
                continue
            bins = np.digitize(X[:, i], edges[1:-1])
            n = len(edges) - 1
            g, h, c = np.zeros(n), np.zeros(n), np.zeros(n, dtype=np.int64)
            for b in range(n):
                m = bins == b
                g[b], h[b], c[b] = grads[m].sum(), hess[m].sum(), m.sum()
            hist.feature_histograms[i] = GradientHistogram(g, h, c, edges)
        return hist



def load_threshold(name):
    try:
        with open(FL_HYPERPARAMS_FILE) as f:
            return float(json.load(f).get("threshold", 0.5))
    except:
        return 0.5


class XgbHistogramHPOClient(Mixin, fl.client.Client):
    
    def __init__(self, client_id, X_train, y_train, X_test, y_test, X_val, y_val,
                 feature_names, mlflow_run_id: str, num_bins=255, threshold=0.5):
        self.client_id = client_id
        self.client_name = get_client_name(client_id)
        self.feature_names = feature_names
        self.mlflow_run_id = mlflow_run_id
        self.current_round = 0
        self.threshold = threshold
        self.modelname = MODEL
        
        # Store numpy versions - Train
        self.X_train_np = np.asarray(X_train, dtype=np.float32)
        self.y_train_np = np.asarray(y_train, dtype=np.float32).ravel()
        
        # Store numpy versions - Test (used only after HPO)
        self.X_test_np = np.asarray(X_test, dtype=np.float32)
        self.y_test_np = np.asarray(y_test, dtype=np.float32).ravel()
        
        # Store numpy versions - Validation (used during HPO)
        self.X_val_np = np.asarray(X_val, dtype=np.float32)
        self.y_val_np = np.asarray(y_val, dtype=np.float32).ravel()
        
        # Handle NaN/Inf
        self.X_train_np = np.nan_to_num(self.X_train_np, nan=0.0, posinf=0.0, neginf=0.0)
        self.X_test_np = np.nan_to_num(self.X_test_np, nan=0.0, posinf=0.0, neginf=0.0)
        self.X_val_np = np.nan_to_num(self.X_val_np, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Histogram builder
        self.hist_builder = HistogramBuilder(num_bins)
        self.hist_builder.build_sketch(self.X_train_np)
        self.global_ensemble = None
        
        logger.info(f"Client {client_id}: {len(self.X_train_np)} train, {len(self.X_val_np)} val, {len(self.X_test_np)} test")
        
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


    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        return GetParametersRes(status=Status(code=Code.OK, message="OK"), 
                               parameters=Parameters(tensor_type="", tensors=[]))

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """Get probability predictions for given input."""
        if self.global_ensemble and self.global_ensemble.trees:
            y_prob = self.global_ensemble.predict_proba(X)
        else:
            y_prob = np.full(len(X), 0.5)
        return np.clip(y_prob, 1e-7, 1 - 1e-7)

    def _model_predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Wrapper for SHAP - returns probability predictions."""
        return self._predict(X)

    def _probs(self, split):
        return self._predict(getattr(self, f"X_{split}_np"))

    def _boost_fit(self, parameters, config):
        """Grow XGB_BOOST_TREES trees on local gradients only."""
        builder = FederatedTreeBuilder(**{k: float(config[k]) for k in ("min_child_weight", "reg_lambda", "reg_gamma") if k in config})
        for _ in range(XGB_BOOST_TREES):
            self.global_ensemble.trees.append(builder.build_tree(self._histogram().feature_histograms))

    def _histogram(self) -> ClientHistogramData:
        """Gradient histogram of the current ensemble on the training data."""
        # Compute gradients
        if self.global_ensemble and self.global_ensemble.trees:
            raw = self.global_ensemble.predict_raw(self.X_train_np)
        else:
            raw = np.zeros(len(self.y_train_np))
        prob = 1.0 / (1.0 + np.exp(-np.clip(raw, -500, 500)))
        grads = prob - self.y_train_np
        hess = np.maximum(prob * (1 - prob), 1e-6)

        # Build histogram
        hist = self.hist_builder.compute_histogram(self.X_train_np, grads, hess)
        hist.target_sum = float(self.y_train_np.sum())
        hist.sample_count = len(self.y_train_np)
        return hist


    def fit(self, ins: FitIns) -> FitRes:
        global_round = int(ins.config.get("global_round", 1))
        self.current_round = global_round
        reset = int(ins.config.get("reset_params", 0))
        
        # Update threshold from config if provided
        threshold = float(ins.config.get("threshold", self.threshold))
        self.threshold = threshold
        
        logger.info(f"Client {self.client_id} - Round {global_round} (reset={reset}, thresh={threshold:.2f})")

        # Load model (will be reset/empty on new HPO trial)
        if reset == 1:
            self.global_ensemble = None
            logger.info(f"  Model reset for new HPO trial")
        elif ins.parameters.tensors:
            try:
                self.global_ensemble = FederatedTreeEnsemble.from_bytes(ins.parameters.tensors[0])
            except Exception as e:
                logger.warning(f"Parameter load failed: {e}")
                self.global_ensemble = None

        hist = self._histogram()

        return FitRes(
            status=Status(code=Code.OK, message="OK"),
            parameters=Parameters(tensor_type="", tensors=[hist.to_bytes()]),
            num_examples=len(self.y_train_np),
            metrics={"client_id": self.client_id}
        )
        
    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        current_round = int(ins.config.get("global_round", self.current_round))
        
        # Update threshold from config if provided
        threshold = float(ins.config.get("threshold", self.threshold))
        self.threshold = threshold
        
        # Load ensemble from parameters
        if ins.parameters.tensors:
            try:
                self.global_ensemble = FederatedTreeEnsemble.from_bytes(ins.parameters.tensors[0])
            except Exception as e:
                logger.warning(f"Parameter load failed: {e}")
                return EvaluateRes(status=Status(code=Code.OK, message="OK"),
                                  loss=1.0, num_examples=len(self.y_test_np), metrics={})
        
        if not self.global_ensemble:
            return EvaluateRes(status=Status(code=Code.OK, message="OK"),
                              loss=1.0, num_examples=len(self.y_test_np), metrics={})
        
        # Check if we're in HPO phase or final evaluation
        is_hpo_phase = bool(ins.config.get("hpo_phase", True))
        print("-------------------------------------------------------", FL_NUM_ROUNDS)
        
        if is_hpo_phase:
            print("This is the val (no test) speaking")
            # =====================================================
            # HPO PHASE: Evaluate on VALIDATION set only
            # =====================================================
            y_prob = self._predict(self.X_val_np)
            
            metrics = compute_all_metrics(self.y_val_np, y_prob, self.threshold, n_boot=0)
            metrics["theoretical_kappa"] = float(self.theoretical_kappa)
            metrics["threshold"] = float(self.threshold)
            metrics["num_trees"] = len(self.global_ensemble.trees)
            metrics["eval_set"] = "validation"
            metrics["cm_grid"] = self._cm_grid(self.y_val_np, y_prob).tobytes()
            
            eval_size = len(self.y_val_np)
            logger.info(f"[HPO] Round {current_round}: VAL AUC={metrics['AUC']:.4f}, "
                       f"AUPRC={metrics['AUPRC']:.4f}, F1={metrics['F1']:.4f}")
            
        else:
            print("Made it to test round")
            # =====================================================
            # FINAL PHASE: Evaluate on TEST set (after HPO complete)
            # =====================================================
            y_prob = self._predict(self.X_test_np)
            
            metrics = compute_all_metrics(self.y_test_np, y_prob, self.threshold)
            metrics["theoretical_kappa"] = float(self.theoretical_kappa)
            metrics["threshold"] = float(self.threshold)
            metrics["num_trees"] = len(self.global_ensemble.trees)
            metrics["eval_set"] = "test"
            
            
            # Also include validation metrics for comparison
            val_prob = self._predict(self.X_val_np)
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
            metrics.update(self._local_boost(ins.parameters, ins.config, y_prob))
            
            #store hyperparameters
            with open(str(FL_BOOST_HYPERPARAMS_FILE_PATTERN).format(client=self.client_name), "w") as f:
                json.dump({"threshold": float(metrics["boost_threshold"]), "boost_trees": XGB_BOOST_TREES}, f, indent=2)
        
            
            eval_size = len(self.y_test_np)
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
            "num_trees": metrics["num_trees"],
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
        csv_path = str(FL_METRICS_CSV_PATTERN).format(client_id=self.client_id, balance=DATA_BALANCE, modelname=self.modelname)
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
            "num_trees": metrics["num_trees"],
        }
        
        # Add validation metrics if in final phase
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
        
        return EvaluateRes(
            status=Status(code=Code.OK, message="OK"),
            loss=metrics["log_loss"],
            num_examples=eval_size,
            metrics=metrics
        )




if __name__ == "__main__":
    client_id = args.client_id
    client_name = get_client_name(client_id)
    
    logger.info(f"Starting FL XGBoost client {client_id} ({client_name})")
    
    # Load data with train/test/val split
    X_train, y_train, X_test, y_test, X_val, y_val, feature_names = prepare_data(client_id)
    
    threshold = load_threshold(client_name)
    
    mlflow.set_experiment(f"{MLFLOW_EXPERIMENT}")
    run_name = f"xgb_client_{client_id}_{client_name}_{MLFLOW_SUFFIX}"

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id

        log_mlflow_params(client_id, client_name, DATA_BALANCE, MODEL, FL_LOCAL_EPOCHS, FL_BATCH_SIZE, X_train, X_val, X_test)

        client = XgbHistogramHPOClient(
            client_id,
            X_train, y_train,
            X_test, y_test,
            X_val, y_val,
            feature_names,
            run_id,
            args.num_bins,
            threshold
        )

        fl.client.start_client(
            server_address=args.server_address,
            client=client
        )
    
    logger.info(f"Client {client_id} finished")