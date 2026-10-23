"""
Federated XGBoost Server - Histogram Aggregation with Optuna HPO.
Combines Jones et al. histogram approach with Parra-Ullauri et al. HPO.
"""

import logging, json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np
import pandas as pd
import flwr as fl
from flwr.common import Parameters, FitIns, EvaluateIns, EvaluateRes, FitRes
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy
import mlflow, optuna

from xgb_config import XGB_HPO_SPACE, RESULTS_DIR, MLFLOW_EXPERIMENT, XGB_NUM_BINS, MLFLOW_SUFFIX, DATA_BALANCE, FL_HYPERPARAMS_FILE
from general_config import FL_NUM_ROUNDS, FL_HPO_FACTOR, FL_HPO_RATE, POOL_SIZE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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

    @classmethod
    def from_bytes(cls, data: bytes) -> 'ClientHistogramData':
        p = json.loads(data.decode('utf-8'))
        obj = cls(p['num_bins'], p['num_features'], target_sum=p.get('target_sum', 0.0), sample_count=p.get('sample_count', 0))
        obj.bin_edges = {int(k): np.array(v) for k, v in p.get('bin_edges', {}).items()}
        for k, v in p.get('feature_histograms', {}).items():
            obj.feature_histograms[int(k)] = GradientHistogram(np.array(v['gradient_sums']), np.array(v['hessian_sums']), np.array(v['bin_counts']), np.array(v['bin_edges']))
        return obj


@dataclass
class TreeNode:
    is_leaf: bool = False
    feature_idx: int = -1
    threshold: float = 0.0
    gain: float = 0.0
    leaf_weight: float = 0.0
    left_child: Optional['TreeNode'] = None
    right_child: Optional['TreeNode'] = None
    node_id: int = 0


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
        return 1.0 / (1.0 + np.exp(-self.predict_raw(X)))

    def _predict_tree(self, node: TreeNode, X: np.ndarray) -> np.ndarray:
        if node.is_leaf:
            return np.full(X.shape[0], node.leaf_weight)
        preds = np.zeros(X.shape[0])
        mask = X[:, node.feature_idx] < node.threshold
        if node.left_child: preds[mask] = self._predict_tree(node.left_child, X[mask])
        if node.right_child: preds[~mask] = self._predict_tree(node.right_child, X[~mask])
        return preds

    def to_bytes(self) -> bytes:
        def ser(n):
            if n.is_leaf: return {'is_leaf': True, 'leaf_weight': n.leaf_weight, 'node_id': n.node_id}
            return {'is_leaf': False, 'feature_idx': n.feature_idx, 'threshold': n.threshold, 'node_id': n.node_id,
                    'left': ser(n.left_child) if n.left_child else None, 'right': ser(n.right_child) if n.right_child else None}
        return json.dumps({'base_score': self.base_score, 'learning_rate': self.learning_rate,
                          'num_features': self.num_features, 'trees': [ser(t) for t in self.trees]}).encode()

    @classmethod
    def from_bytes(cls, data: bytes) -> 'FederatedTreeEnsemble':
        p = json.loads(data.decode())
        ens = cls(base_score=p['base_score'], learning_rate=p['learning_rate'], num_features=p['num_features'])
        def des(d):
            if d['is_leaf']: return TreeNode(is_leaf=True, leaf_weight=d['leaf_weight'], node_id=d['node_id'])
            n = TreeNode(is_leaf=False, feature_idx=d['feature_idx'], threshold=d['threshold'], node_id=d['node_id'])
            if d.get('left'): n.left_child = des(d['left'])
            if d.get('right'): n.right_child = des(d['right'])
            return n
        ens.trees = [des(t) for t in p['trees']]
        return ens


class FederatedTreeBuilder:
    def __init__(self, max_depth=1, min_child_weight=1.0, reg_lambda=1.0, reg_gamma=0.0):
        self.max_depth, self.min_child_weight, self.reg_lambda, self.reg_gamma = max_depth, min_child_weight, reg_lambda, reg_gamma

    def build_tree(self, histograms: Dict[int, GradientHistogram]) -> TreeNode:
        if not histograms:
            return TreeNode(is_leaf=True, leaf_weight=0.0)
        G_tot = sum(h.gradient_sums.sum() for h in histograms.values())
        H_tot = sum(h.hessian_sums.sum() for h in histograms.values())
        best = {'gain': -np.inf, 'feat': -1, 'thresh': 0.0, 'w_L': 0.0, 'w_R': 0.0}
        
        for feat, hist in histograms.items():
            if len(hist.gradient_sums) == 0: continue
            G_L, H_L = 0.0, 0.0
            for i in range(len(hist.gradient_sums) - 1):
                G_L += hist.gradient_sums[i]; H_L += hist.hessian_sums[i]
                G_R, H_R = G_tot - G_L, H_tot - H_L
                if H_L < self.min_child_weight or H_R < self.min_child_weight: continue
                gain = 0.5 * (G_L**2/(H_L+self.reg_lambda) + G_R**2/(H_R+self.reg_lambda) - (G_L+G_R)**2/(H_L+H_R+self.reg_lambda)) - self.reg_gamma
                if gain > best['gain']:
                    best = {'gain': gain, 'feat': feat, 'thresh': float(hist.bin_edges[min(i+1, len(hist.bin_edges)-1)]),
                            'w_L': -G_L/(H_L+self.reg_lambda), 'w_R': -G_R/(H_R+self.reg_lambda)}
        
        if best['feat'] < 0 or best['gain'] <= 0:
            return TreeNode(is_leaf=True, leaf_weight=-G_tot/(H_tot+self.reg_lambda) if H_tot+self.reg_lambda != 0 else 0.0)
        
        root = TreeNode(is_leaf=False, feature_idx=best['feat'], threshold=best['thresh'], gain=best['gain'], node_id=0)
        root.left_child = TreeNode(is_leaf=True, leaf_weight=best['w_L'], node_id=1)
        root.right_child = TreeNode(is_leaf=True, leaf_weight=best['w_R'], node_id=2)
        return root


def merge_histograms(client_data: List[ClientHistogramData]) -> Dict[int, GradientHistogram]:
    if not client_data: return {}
    all_feats = set().union(*[d.feature_histograms.keys() for d in client_data])
    merged = {}
    for feat in all_feats:
        hists = [d.feature_histograms[feat] for d in client_data if feat in d.feature_histograms]
        if not hists: continue
        n = len(hists[0].gradient_sums)
        g, h, c = np.zeros(n), np.zeros(n), np.zeros(n)
        for hist in hists:
            m = min(n, len(hist.gradient_sums))
            g[:m] += hist.gradient_sums[:m]; h[:m] += hist.hessian_sums[:m]; c[:m] += hist.bin_counts[:m]
        merged[feat] = GradientHistogram(g, h, c, hists[0].bin_edges)
    return merged


class FedXgbHistogramHPO(Strategy):
    """Histogram XGBoost with Optuna HPO. Resets ensemble at each new HPO trial."""

    def __init__(self, num_rounds: int, **kwargs):
        super().__init__()
        self.num_rounds = num_rounds
        self.hpo_rounds = int(num_rounds * FL_HPO_FACTOR)
        self.fraction_fit = kwargs.get('fraction_fit', 1.0)
        self.fraction_evaluate = kwargs.get('fraction_evaluate', 1.0)
        self.min_fit_clients = kwargs.get('min_fit_clients', POOL_SIZE)
        self.min_available_clients = kwargs.get('min_available_clients', POOL_SIZE)

        # Optuna
        self.study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
        self.current_trial, self.current_params, self.best_params = None, {}, {}

        # XGB state
        self.global_ensemble: Optional[FederatedTreeEnsemble] = None
        self.base_score = 0.5
        self.num_features = 0
        self.tree_builder = FederatedTreeBuilder()
        self.metrics_history = []
        self.threshold = 0.5

        logger.info(f"HPO rounds 1-{self.hpo_rounds}, Training {self.hpo_rounds+1}-{num_rounds-1}, Rate={FL_HPO_RATE}")

    def _suggest_params(self) -> Dict:
        self.current_trial = self.study.ask()
        params = {}
        for name, (lo, hi, ptype) in XGB_HPO_SPACE.items():
            if ptype == "log": params[name] = self.current_trial.suggest_float(name, lo, hi, log=True)
            elif ptype == "int": params[name] = self.current_trial.suggest_int(name, lo, hi)
            else: params[name] = self.current_trial.suggest_float(name, lo, hi)
        return params

    def _reset_ensemble(self, params: Dict):
        """Reset ensemble and tree builder with new hyperparameters."""
        self.tree_builder = FederatedTreeBuilder(
            max_depth=int(params.get('max_depth', 1)),
            min_child_weight=float(params.get('min_child_weight', 1.0)),
            reg_lambda=float(params.get('reg_lambda', 1.0)),
            reg_gamma=float(params.get('reg_gamma', 0.0)))
        lr = float(params.get('learning_rate', 0.3))
        self.colsample = float(params.get('colsample_bytree', 1.0))
        # Reset ensemble - start fresh with new params
        self.global_ensemble = FederatedTreeEnsemble(base_score=self.base_score, learning_rate=lr, num_features=self.num_features)
        logger.info(f"  Reset ensemble: lr={lr:.4f}, depth={params.get('max_depth')}, λ={params.get('reg_lambda'):.4f}")

    def initialize_parameters(self, client_manager): return Parameters(tensor_type="", tensors=[])

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager: ClientManager):
        if server_round == self.num_rounds: return []  # final round: test evaluation only
        config = {"global_round": server_round, "num_bins": XGB_NUM_BINS, "reset_params": 0}

        if server_round <= self.hpo_rounds:
            # New HPO trial every HPO_RATE rounds
            if (server_round - 1) % FL_HPO_RATE == 0:
                self.current_params = self._suggest_params()
                config["reset_params"] = 1
                logger.info(f"Round {server_round} [HPO NEW TRIAL]: {self.current_params}")
            for k, v in self.current_params.items(): config[k] = v
        else:
            if server_round == self.hpo_rounds + 1:
                self._reset_ensemble(self.best_params)
                logger.info(f"Round {server_round} [TRAINING]: Best params {self.best_params}")
            for k, v in self.best_params.items(): config[k] = v

        clients = client_manager.sample(num_clients=max(int(self.fraction_fit * client_manager.num_available()), self.min_fit_clients), min_num_clients=self.min_available_clients)
        params = Parameters(tensor_type="", tensors=[self.global_ensemble.to_bytes()]) if self.global_ensemble else Parameters(tensor_type="", tensors=[])
        return [(c, FitIns(params, config)) for c in clients]

    def aggregate_fit(self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures):
        if not results: return None, {}

        # Parse histograms
        client_hists, target_sum, sample_count = [], 0.0, 0
        for _, res in results:
            if res.parameters.tensors:
                try:
                    h = ClientHistogramData.from_bytes(res.parameters.tensors[0])
                    client_hists.append(h)
                    target_sum += h.target_sum; sample_count += h.sample_count
                    self.num_features = max(self.num_features, h.num_features)
                except: pass

        if not client_hists: return None, {}

        # Initialize or reset ensemble on new HPO trial
        if server_round <= self.hpo_rounds and (server_round - 1) % FL_HPO_RATE == 0:
            self.base_score = np.clip(target_sum / max(sample_count, 1), 0.01, 0.99)
            self._reset_ensemble(self.current_params)

        # First round initialization
        if self.global_ensemble is None:
            self.base_score = np.clip(target_sum / max(sample_count, 1), 0.01, 0.99)
            self.global_ensemble = FederatedTreeEnsemble(base_score=self.base_score, learning_rate=0.3, num_features=self.num_features)

        # Merge and build tree
        merged = merge_histograms(client_hists)
        if self.colsample < 1.0 and len(merged) > 1:
            sel = np.random.choice(list(merged.keys()), max(1, int(len(merged) * self.colsample)), replace=False)
            merged = {k: v for k, v in merged.items() if k in sel}

        tree = self.tree_builder.build_tree(merged)
        self.global_ensemble.trees.append(tree)
        logger.info(f"Round {server_round}: {len(self.global_ensemble.trees)} trees, split=f{tree.feature_idx}" if not tree.is_leaf else f"Round {server_round}: leaf")

        return Parameters(tensor_type="", tensors=[self.global_ensemble.to_bytes()]), {"num_trees": len(self.global_ensemble.trees)}

    def configure_evaluate(self, server_round: int, parameters: Parameters, client_manager: ClientManager):
        if not self.global_ensemble: return []
        clients = client_manager.sample(num_clients=max(int(self.fraction_evaluate * client_manager.num_available()), self.min_fit_clients), min_num_clients=self.min_available_clients)
        config = {**self.best_params, "global_round": server_round, "threshold": self.threshold, "hpo_phase": server_round < self.num_rounds}
        return [(c, EvaluateIns(Parameters(tensor_type="", tensors=[self.global_ensemble.to_bytes()]), config)) for c in clients]

    def aggregate_evaluate(self, server_round: int, results: List[Tuple[ClientProxy, EvaluateRes]], failures):
        if not results: return None, {}

        total = sum(r.num_examples for _, r in results)
        agg = defaultdict(float)
        for _, r in results:
            w = r.num_examples / total
            for k, v in r.metrics.items():
                if isinstance(v, (int, float)): agg[k] += v * w
        loss = sum(r.loss * r.num_examples for _, r in results) / total
        auprc = agg.get("AUPRC", 0)

        # Threshold maximizing pooled F1 on validation set
        cms = [np.frombuffer(r.metrics["cm_grid"], dtype=np.int64).reshape(3, -1) for _, r in results if "cm_grid" in r.metrics]
        if cms:
            tp, fp, fn = sum(cms)
            f1 = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros(len(tp)), where=(2 * tp + fp + fn) > 0)
            self.threshold = float(np.linspace(0.05, 0.95, 91)[np.argmax(f1)])

        # HPO: tell Optuna
        if server_round <= self.hpo_rounds and server_round % FL_HPO_RATE == 0 and self.current_trial:
            self.study.tell(self.current_trial, loss)
            logger.info(f"  HPO tell: log_loss={loss:.4f}")
            if self.study.best_trial: self.best_params = self.study.best_params.copy()
            self.current_trial = None

        if server_round == self.hpo_rounds:
            logger.info(f"{'='*50}\nHPO DONE: Best log_loss={self.study.best_value:.4f}\nParams: {self.best_params}\n{'='*50}")
            mlflow.log_params({f"best_{k}": v for k, v in self.best_params.items()})

        for k, v in agg.items(): mlflow.log_metric(k, v, step=server_round)
        self.metrics_history.append({"round": server_round, "phase": "HPO" if server_round <= self.hpo_rounds else "TRAIN", **dict(agg)})
        logger.info(f"Round {server_round}: AUC={agg.get('AUC',0):.4f}, AUPRC={auprc:.4f}, F1={agg.get('F1',0):.4f}")
        return loss, dict(agg)

    def evaluate(self, server_round, parameters): return None


def main():
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    mlflow.start_run(run_name=f"histogram_hpo_server_{MLFLOW_SUFFIX}")
    mlflow.log_params({"num_rounds": FL_NUM_ROUNDS, "hpo_factor": FL_HPO_FACTOR, "hpo_rate": FL_HPO_RATE, "data_balance": DATA_BALANCE})

    strategy = FedXgbHistogramHPO(num_rounds=FL_NUM_ROUNDS)
    strategy.colsample = 1.0

    fl.server.start_server(server_address="0.0.0.0:8097", config=fl.server.ServerConfig(num_rounds=FL_NUM_ROUNDS), strategy=strategy)
    with open(FL_HYPERPARAMS_FILE, "w") as f:
        json.dump({**strategy.best_params, "threshold": strategy.threshold}, f, indent=2)

    RESULTS_DIR.mkdir(exist_ok=True)
    pd.DataFrame(strategy.metrics_history).to_csv(RESULTS_DIR / f"FL_histogram_hpo_history_{DATA_BALANCE}.csv", index=False)
    strategy.study.trials_dataframe().to_csv(RESULTS_DIR / f"optuna_trials_{DATA_BALANCE}_xgb.csv", index=False)
    mlflow.end_run()
    logger.info("Server finished")


if __name__ == "__main__":
    main()