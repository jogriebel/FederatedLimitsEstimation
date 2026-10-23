"""Federated Learning Server with Logistic Regression and Optuna HPO (FedProx Implementation)."""

from typing import Dict, List, Tuple
import flwr as fl
from flwr.server.strategy import FedProx
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager
from flwr.common import FitIns, Parameters, Scalar
import logging
import optuna
import mlflow
import numpy as np
import pandas as pd
import os
import json

from lr_config import LR_HPO_SPACE, RESULTS_DIR, MLFLOW_EXPERIMENT, FL_HYPERPARAMS_FILE
from general_config import FL_NUM_ROUNDS, FL_HPO_FACTOR, FL_HPO_RATE, POOL_SIZE, DATA_BALANCE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_metrics_aggregation(eval_metrics):
    """Return weighted aggregated metrics."""
    total_num = sum([num for _, (_, num, _) in eval_metrics])
    metrics_aggregated = {}
    metric_names = ["AUC", "AUPRC", "Precision", "Recall", "F1", 
                    "Sensitivity", "Specificity", "Brier", "actual_kappa"]
    for metric in metric_names:
        metrics_aggregated[metric] = sum(
            [m.get(metric, 0) * n for _, (_, n, m) in eval_metrics]
        ) / total_num
    return metrics_aggregated


class FedProxWithHPO(FedProx):
    """
    FedProx with Optuna HPO
    
    HPO Phase: First FL_HPO_FACTOR * num_rounds
    Training Phase: Remaining rounds with best params
    """
    
    def __init__(self, num_rounds: int, **kwargs):
        # Start with a default proximal_mu (will be overridden by HPO)
        if 'proximal_mu' not in kwargs:
            kwargs['proximal_mu'] = 0.1
        
        super().__init__(**kwargs)
        self.num_rounds = num_rounds
        self.hpo_rounds = int(num_rounds * FL_HPO_FACTOR)
        
        # Optuna study
        self.study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        self.current_trial = None
        self.current_params = {}
        self.best_params = {}
        self.threshold = 0.5
        
        # Metrics
        self.metrics_history = []
        self.client_metrics_by_round = {}
        
        # MLflow run reference
        self.mlflow_run = None
        
        logger.info(f"FedProx with HPO for Logistic Regression")
        logger.info(f"HPO: rounds 1-{self.hpo_rounds}, Training: rounds {self.hpo_rounds+1}-{num_rounds-1}")

    def _suggest_params(self) -> Dict:
        """Get hyperparameters from Optuna (paper: ask())."""
        self.current_trial = self.study.ask()
        params = {}
        for name, spec in LR_HPO_SPACE.items():
            low, high, ptype = spec
            if ptype == "log":
                params[name] = self.current_trial.suggest_float(name, low, high, log=True)
            elif ptype == "int":
                params[name] = self.current_trial.suggest_int(name, low, high)
            else:
                params[name] = self.current_trial.suggest_float(name, low, high)
        return params

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, fl.common.EvaluateIns]]:
        """Configure evaluation with global_round in config."""
        
        config: Dict[str, Scalar] = {
            "global_round": server_round,
        }
        
        # Add current hyperparameters
        if server_round <= self.hpo_rounds:
            for k, v in self.current_params.items():
                config[k] = v
        else:
            for k, v in self.best_params.items():
                config[k] = v
        config["threshold"] = self.threshold
        config["hpo_phase"] = server_round < self.num_rounds
        
        sample_size = int(self.fraction_evaluate * client_manager.num_available())
        clients = client_manager.sample(
            num_clients=max(sample_size, self.min_evaluate_clients),
            min_num_clients=self.min_available_clients
        )
        
        return [(client, fl.common.EvaluateIns(parameters, config)) for client in clients]

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        if server_round == self.num_rounds: return []  # final round: test evaluation only
        
        config: Dict[str, Scalar] = {
            "global_round": server_round,
        }
        
        # HPO Phase
        if server_round <= self.hpo_rounds:
            if (server_round - 1) % FL_HPO_RATE == 0:
                self.current_params = self._suggest_params()
                config["reset_params"] = 1
            else:
                config["reset_params"] = 0
            
            # Add all hyperparameters from current trial (including proximal_mu)
            for k, v in self.current_params.items():
                config[k] = v
        else:
            # Training phase - use best params
            if server_round == self.hpo_rounds + 1:
                logger.info(f"Round {server_round} [TRAINING START]: {self.best_params}")
            config["reset_params"] = 0
            
            # Add all best hyperparameters (including proximal_mu)
            for k, v in self.best_params.items():
                config[k] = v
        
        sample_size = int(self.fraction_fit * client_manager.num_available())
        clients = client_manager.sample(
            num_clients=max(sample_size, self.min_fit_clients),
            min_num_clients=self.min_available_clients
        )
        
        return [(client, FitIns(parameters, config)) for client in clients]

    def aggregate_evaluate(self, server_round: int, results, failures):
        if not results:
            return None, {}
        
        step = int(server_round)
        
        # Build eval_metrics structure
        eval_metrics = [
            (proxy, (res.loss, res.num_examples, res.metrics))
            for proxy, res in results
        ]
        
        aggregated = evaluate_metrics_aggregation(eval_metrics)
        auprc = aggregated["AUPRC"]
        
        # Aggregate loss (weighted average)
        total_examples = sum([num for _, (_, num, _) in eval_metrics])
        aggregated_loss = sum([loss * num for _, (loss, num, _) in eval_metrics]) / total_examples
        
        # Threshold maximizing pooled F1 on validation set
        cms = [np.frombuffer(res.metrics["cm_grid"], dtype=np.int64).reshape(3, -1) for _, res in results if "cm_grid" in res.metrics]
        if cms:
            tp, fp, fn = sum(cms)
            f1 = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros(len(tp)), where=(2 * tp + fp + fn) > 0)
            self.threshold = float(np.linspace(0.05, 0.95, 91)[np.argmax(f1)])
        
        # HPO: Report to Optuna (paper: tell())
        if server_round <= self.hpo_rounds:
            if server_round % FL_HPO_RATE == 0 and self.current_trial is not None:
                self.study.tell(self.current_trial, aggregated_loss)
                logger.info(f"  → log_loss={aggregated_loss:.4f} reported to Optuna")
                
                if self.study.best_trial:
                    self.best_params = self.study.best_params.copy()
                self.current_trial = None
        
        # Log to MLflow with explicit step
        aggregated["Round"] = server_round
        mlflow.log_metric("server_AUC", aggregated["AUC"], step=step)
        mlflow.log_metric("server_AUPRC", aggregated["AUPRC"], step=step)
        mlflow.log_metric("server_Precision", aggregated["Precision"], step=step)
        mlflow.log_metric("server_Recall", aggregated["Recall"], step=step)
        mlflow.log_metric("server_F1Score", aggregated["F1"], step=step)
        mlflow.log_metric("server_Sensitivity", aggregated["Sensitivity"], step=step)
        mlflow.log_metric("server_Specificity", aggregated["Specificity"], step=step)
        mlflow.log_metric("server_Brier", aggregated["Brier"], step=step)
        mlflow.log_metric("server_actual_kappa", aggregated["actual_kappa"], step=step)
        mlflow.log_metric("server_loss", aggregated_loss, step=step)
        
        # Log current HPO params
        if server_round <= self.hpo_rounds and self.current_params:
            for k, v in self.current_params.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(f"hpo_{k}", v, step=step)
        
        # Phase transitions
        if server_round == self.hpo_rounds:
            logger.info("=" * 50)
            logger.info(f"HPO COMPLETE: Best log_loss={self.study.best_value:.4f}")
            logger.info(f"Best params: {self.best_params}")
            logger.info("=" * 50)
            mlflow.log_params({f"best_{k}": v for k, v in self.best_params.items()})
        
        # Save aggregated metrics
        RESULTS_DIR.mkdir(exist_ok=True)
        csv_path = RESULTS_DIR / "federated_metrics_aggregated.csv"
        row = {
            "round": server_round,
            "phase": "HPO" if server_round <= self.hpo_rounds else "TRAIN",
            **aggregated
        }
        pd.DataFrame([row]).to_csv(
            csv_path, mode='a', header=not os.path.exists(csv_path), index=False
        )
        
        # Store client metrics
        for proxy, res in results:
            client_id = proxy.cid
            if client_id not in self.client_metrics_by_round:
                self.client_metrics_by_round[client_id] = []
            self.client_metrics_by_round[client_id].append({
                "Round": server_round,
                **{k: v for k, v in res.metrics.items() if isinstance(v, (int, float))}
            })
        
        return aggregated_loss, aggregated


def main():
    # MLflow setup
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    
    with mlflow.start_run(run_name="server"):
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("num_rounds", FL_NUM_ROUNDS)
        mlflow.log_param("hpo_factor", FL_HPO_FACTOR)
        mlflow.log_param("hpo_rate", FL_HPO_RATE)
        mlflow.log_param("pool_size", POOL_SIZE)
        
        strategy = FedProxWithHPO(
            num_rounds=FL_NUM_ROUNDS,
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=POOL_SIZE,
            min_evaluate_clients=POOL_SIZE,
            min_available_clients=POOL_SIZE,
        )
        
        logger.info("Starting Federated Learning Server with FedProx + HPO (Logistic Regression)")
        fl.server.start_server(
            server_address="0.0.0.0:8097",
            config=fl.server.ServerConfig(num_rounds=FL_NUM_ROUNDS),
            strategy=strategy,
        )
        
        # Save results
        RESULTS_DIR.mkdir(exist_ok=True)
        strategy.study.trials_dataframe().to_csv(RESULTS_DIR / f"optuna_trials_{DATA_BALANCE}_lr.csv", index=False)
        with open(FL_HYPERPARAMS_FILE, "w") as f:
            json.dump({**strategy.best_params, "threshold": strategy.threshold}, f, indent=2)
        
        mlflow.end_run()
        logger.info("Server finished")


if __name__ == "__main__":
    main()